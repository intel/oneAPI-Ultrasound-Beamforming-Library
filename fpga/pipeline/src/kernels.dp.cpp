// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <cmath>
#include "BeamForming.h"
#include "HilbertFirEnvelope.h"
#include "LogCompressor.h"
#include "ScanConverter.h"

using bf_pipe = sycl::ext::intel::pipe<class BF_HE_PIPE, float, 128>;
using he_pipe = sycl::ext::intel::pipe<class HE_LC_PIPE, float, 128>;
using lc_pipe = sycl::ext::intel::pipe<class LC_SC_PIPE, float, 128>;

#define SHOW_IMG_WIDTH 350
#define SHOW_IMG_HEIGHT 400
#define SHOW_IMG_CHANNELS 1

template <typename T>
inline T computeAperture_D(T F, T z) {
  return z / (2 * F);
}

template <typename T>
inline T computeDelayDTSPACE_D(T dirX, T dirY, T dirZ, T x_element, T x, T z) {
  return sycl::sqrt((x_element - (x + dirX * z)) *
                        (x_element - (x + dirX * z)) +
                    (dirY * z) * (dirY * z)) +
         z;
}

// Shuffle the input order
template <typename RFType>
void shuffle_RF_order(RFType *shuffle_RF, const RFType *RF,
                      size_t numTransducer = 128, size_t numChannel = 64,
                      size_t numTimeSteps = 2337) {
  for (int i = 0; i < numTransducer; i++) {
    uint32_t transducerIndex = i * numChannel * numTimeSteps;
    for (int j = 0; j < numChannel; j++) {
      uint32_t channelIndex = j * numTimeSteps;
      for (int k = 0; k < numTimeSteps; k++) {
        shuffle_RF[transducerIndex + k * numChannel + j] =
            RF[transducerIndex + channelIndex + k];
      }
    }
  }
}

// Shuffle the output image order
template <typename ImageType>
void shuffle_image(ImageType *outputimg, ImageType *inputimg,
                   size_t row_size = 255, size_t col_size = 2000) {
  for (size_t i = 0; i < row_size; i++) {
    int r = i * col_size;
    for (size_t j = 0; j < col_size; j++) {
      outputimg[j * row_size + i] = inputimg[r + j];
    }
  }
}
std::ostream &operator<<(std::ostream &os,
                         const ScanlineRxParameters3D &params) {
  os << std::setprecision(9) << params.position.x << " " << std::setprecision(9)
     << params.position.y << " " << std::setprecision(9) << params.position.z
     << " " << std::setprecision(9) << params.direction.x << " "
     << std::setprecision(9) << params.direction.y << " "
     << std::setprecision(9) << params.direction.z << " "
     << params.maxElementDistance.x << " " << params.maxElementDistance.y
     << " ";
  for (size_t k = 0; k < std::extent<decltype(params.txWeights)>::value; k++) {
    os << params.txParameters[k].firstActiveElementIndex.x << " "
       << params.txParameters[k].firstActiveElementIndex.y << " "
       << params.txParameters[k].lastActiveElementIndex.x << " "
       << params.txParameters[k].lastActiveElementIndex.y << " "
       << params.txParameters[k].txScanlineIdx << " " << std::setprecision(9)
       << params.txParameters[k].initialDelay << " " << std::setprecision(9)
       << params.txWeights[k] << " ";
  }
  return os;
}

std::istream &operator>>(std::istream &is, ScanlineRxParameters3D &params) {
  is >> params.position.x >> params.position.y >> params.position.z >>
      params.direction.x >> params.direction.y >> params.direction.z >>
      params.maxElementDistance.x >> params.maxElementDistance.y;
  for (size_t k = 0; k < std::extent<decltype(params.txWeights)>::value; k++) {
    is >> params.txParameters[k].firstActiveElementIndex.x >>
        params.txParameters[k].firstActiveElementIndex.y >>
        params.txParameters[k].lastActiveElementIndex.x >>
        params.txParameters[k].lastActiveElementIndex.y >>
        params.txParameters[k].txScanlineIdx >>
        params.txParameters[k].initialDelay >> params.txWeights[k];
  }
  return is;
}
void convertToDtSpace(double dt, double &speedOfSoundMMperS,
                      size_t numTransducerElements, int numRxScanlines,
                      int rxNumDepths,
                      vector<vector<ScanlineRxParameters3D>> &cP,
                      float *pRxDepths, float *pRxElementXs) {
  double tspeedOfSoundMMperS = speedOfSoundMMperS;
  double oldFactor = 1;
  double oldFactorTime = 1;

  double factor = 1 / oldFactor / (tspeedOfSoundMMperS * dt);
  double factorTime = 1 / oldFactorTime / dt;

  for (size_t i = 0; i < numRxScanlines; i++) {
    cP[i][0].position.x = cP[i][0].position.x * factor;
    cP[i][0].position.y = cP[i][0].position.y * factor;
    cP[i][0].position.z = cP[i][0].position.z * factor;
    for (size_t k = 0; k < std::extent<decltype(cP[i][0].txWeights)>::value;
         k++) {
      cP[i][0].txParameters[k].initialDelay *= factorTime;
    }
    cP[i][0].maxElementDistance.x = cP[i][0].maxElementDistance.x * factor;
    cP[i][0].maxElementDistance.y = cP[i][0].maxElementDistance.y * factor;
  }

  for (size_t i = 0; i < rxNumDepths; i++) {
    pRxDepths[i] = pRxDepths[i] * factor;
  }

  for (size_t i = 0; i < numTransducerElements; i++) {
    pRxElementXs[i] = pRxElementXs[i] * factor;
  }

  speedOfSoundMMperS = tspeedOfSoundMMperS;
}

template <typename T>
T windowFunction(const WindowType &type, const T &relativeIndex,
                 const T &windowParameter) {
  switch (type) {
    case WindowRectangular:
      return 1.0;
    case WindowHann:
      return (1 - windowParameter) *
                 (0.5f - 0.5f * std::cos(2 * static_cast<T>(M_PI) *
                                         ((relativeIndex + 1) * 0.5f))) +
             windowParameter;
    case WindowHamming:
      return (1 - windowParameter) *
                 (0.54f - 0.46f * std::cos(2 * static_cast<T>(M_PI) *
                                           ((relativeIndex + 1) * 0.5f))) +
             windowParameter;
    case WindowGauss:
      return static_cast<T>(
          1.0 / (windowParameter * sqrt(2.0 * M_PI)) *
          exp((-1.0 / 2.0) * (relativeIndex / windowParameter) *
              (relativeIndex / windowParameter)));
    default:
      return 0;
  }
}

void WindowData(WindowType type, float windowParameter,
                size_t numEntriesPerFunction, float *m_data) {
  float maxValue = std::numeric_limits<float>::min();

  for (size_t entryIdx = 0; entryIdx < numEntriesPerFunction; entryIdx++) {
    float relativeEntryIdx =
        (static_cast<float>(entryIdx) / (numEntriesPerFunction - 1)) * 2 - 1;
    float value = windowFunction(type, relativeEntryIdx, windowParameter);
    m_data[entryIdx] = value;
    maxValue = std::max(maxValue, value);
  }

  // normalize window
  for (size_t entryIdx = 0; entryIdx < numEntriesPerFunction; entryIdx++) {
    m_data[entryIdx] /= maxValue;
  }
}

static constexpr double m_skewnessTestThreshold = 1e-6;
static constexpr int m_mappingMaxIterations = 1000;
static constexpr double m_mappingDistanceThreshold = 1e-5;
vec3 m_bbMin = {0, 0, 0};
vec3 m_bbMax = {0, 0, 0};

double barycentricCoordinate2D(const vec2 &a, const vec2 &b, const vec2 &c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}
bool pointInsideTriangle(const vec2 &a, const vec2 &b, const vec2 &c,
                         const vec2 &p) {
  double w0 = barycentricCoordinate2D(b, c, p);
  double w1 = barycentricCoordinate2D(c, a, p);
  double w2 = barycentricCoordinate2D(a, b, p);

  // Test if p is on or inside all edges
  return (w0 >= 0 && w1 >= 0 && w2 >= 0);
}

vec3 pointLineConnection(const vec3 &a, const vec3 &da, const vec3 &x) {
  vec3 conn = x - a;
  vec3 r = conn - dot(da, conn) * da;
  return r;
}

vec2 mapToParameters2D(const vec3 &a, const vec3 &b, const vec3 &da,
                       const vec3 &db, double startDepth, double endDepth,
                       const vec3 &x) {
  // find t via binary search
  double lowT = 0;
  double highT = 1;
  vec3 lowConn = pointLineConnection(a, da, x);
  vec3 highConn = pointLineConnection(b, db, x);
  double lowDist = norm(lowConn);
  double highDist = norm(highConn);

  if (highConn.x == 0 && highConn.y == 0 && highConn.z == 0) {
    double t = highT;
    double d = norm(x - b);
    return {t, d};
  } else if (lowConn.x == 0 && lowConn.y == 0 && lowConn.z == 0) {
    double t = lowT;
    double d = norm(x - a);
    return {t, d};
  }

  assert(dot(lowConn, highConn) < 0);

  double dist = 1e10;
  double t = (highT - lowT) / 2 + lowT;
  vec3 lineBase;
  for (size_t numIter = 0;
       numIter < m_mappingMaxIterations && dist > m_mappingDistanceThreshold;
       numIter++) {
    t = (1 - highDist / (highDist + lowDist)) * highT +
        (1 - lowDist / (highDist + lowDist)) * lowT;

    lineBase = (1 - t) * a + t * b;
    vec3 lineDir = slerp3(da, db, t);

    vec3 conn = pointLineConnection(lineBase, lineDir, x);
    dist = norm(conn);

    if (dot(lowConn, conn) < 0) {
      highT = t;
      highConn = conn;
      highDist = dist;
    } else {
      lowT = t;
      lowConn = conn;
      lowDist = dist;
    }
  }
  double d = norm(x - lineBase);

  return {t, d};
}

// Calculate the params for ScanConvertor
void updateInternals(vector<uint8_t> &m_mask, vector<float> &m_weightX,
                     vector<float> &m_weightY, vector<uint32_t> &m_sampleIdx,
                     vec3s &m_imageSize, vec2s layout, double endDepth,
                     vector<vector<ScanlineRxParameters3D>> &scanlines,
                     int numSamples, int NumScanlines = 255,
                     double resolution = 0.022511) {
  // Check the scanline configuration for validity
  double startDepth = 0;
  double SampleDistance = endDepth / (numSamples - 1);
  resolution = SampleDistance;

  bool scanlinesGood = true;

  if (scanlinesGood) {
    for (size_t scanlineIdxY = 0; scanlineIdxY < layout.y; scanlineIdxY++) {
      for (size_t scanlineIdxX = 0; scanlineIdxX < layout.x; scanlineIdxX++) {
        if (scanlineIdxX > 0) {
          vec3 start =
              scanlines[scanlineIdxX][scanlineIdxY].getPoint(startDepth);
          vec3 startbefore =
              scanlines[scanlineIdxX - 1][scanlineIdxY].getPoint(startDepth);
          vec3 end = scanlines[scanlineIdxX][scanlineIdxY].getPoint(endDepth);
          vec3 endbefore =
              scanlines[scanlineIdxX - 1][scanlineIdxY].getPoint(endDepth);

          // scanline start points are increasing in x
          scanlinesGood = scanlinesGood && start.x >= startbefore.x;
          if (!scanlinesGood) {
            scanlinesGood = true;
          }
          // scanline end points are increasing in x, that means scanlines do
          // not intersect
          scanlinesGood = scanlinesGood && end.x >= endbefore.x;
          if (!scanlinesGood) {
            scanlinesGood = true;
          }
          // scanlines can not be identical
          scanlinesGood =
              scanlinesGood && (start.x > startbefore.x || end.x > endbefore.x);
          if (!scanlinesGood) {
            scanlinesGood = true;
          }
          // scanlines are not skew
          scanlinesGood = scanlinesGood &&
                          abs(det(start - endbefore, startbefore - endbefore,
                                  end - endbefore)) < m_skewnessTestThreshold;
          if (!scanlinesGood) {
            scanlinesGood = true;
          }
        }

        if (scanlineIdxY > 0) {
          vec3 start =
              scanlines[scanlineIdxX][scanlineIdxY].getPoint(startDepth);
          vec3 startbefore =
              scanlines[scanlineIdxX][scanlineIdxY - 1].getPoint(startDepth);
          vec3 end = scanlines[scanlineIdxX][scanlineIdxY].getPoint(endDepth);
          vec3 endbefore =
              scanlines[scanlineIdxX][scanlineIdxY - 1].getPoint(endDepth);

          // scanline start points are increasing in z
          scanlinesGood = scanlinesGood && start.z >= startbefore.z;
          if (!scanlinesGood) {
            scanlinesGood = true;
          }
          // scanline end points are increasing in z, that means scanlines do
          // not intersect
          scanlinesGood = scanlinesGood && end.z >= endbefore.z;
          if (!scanlinesGood) {
            scanlinesGood = true;
          }
          // scanlines can not be identical
          scanlinesGood =
              scanlinesGood && (start.z > startbefore.z || end.z > endbefore.z);
          if (!scanlinesGood) {
            scanlinesGood = true;
          }
          // scanlines are not skew
          scanlinesGood = scanlinesGood &&
                          abs(det(start - endbefore, startbefore - endbefore,
                                  end - endbefore)) < m_skewnessTestThreshold;
          if (!scanlinesGood) {
            scanlinesGood = true;
          }
        }
      }
    }
  }

  if (scanlinesGood) {
    // find scan bounding box
    vec3 bbMin{numeric_limits<double>::max(), numeric_limits<double>::max(),
               numeric_limits<double>::max()};
    vec3 bbMax{-numeric_limits<double>::max(), -numeric_limits<double>::max(),
               -numeric_limits<double>::max()};
    for (size_t scanlineIdxY = 0; scanlineIdxY < layout.y; scanlineIdxY++) {
      for (size_t scanlineIdxX = 0; scanlineIdxX < layout.x; scanlineIdxX++) {
        vec3 p1 = scanlines[scanlineIdxX][scanlineIdxY].getPoint(startDepth);
        vec3 p2 = scanlines[scanlineIdxX][scanlineIdxY].getPoint(endDepth);
        bbMin = {std::min(bbMin.x, p1.x), std::min(bbMin.y, p1.y),
                 std::min(bbMin.z, p1.z)};
        bbMax = {std::max(bbMax.x, p1.x), std::max(bbMax.y, p1.y),
                 std::max(bbMax.z, p1.z)};
        bbMin = {std::min(bbMin.x, p2.x), std::min(bbMin.y, p2.y),
                 std::min(bbMin.z, p2.z)};
        bbMax = {std::max(bbMax.x, p2.x), std::max(bbMax.y, p2.y),
                 std::max(bbMax.z, p2.z)};
      }
    }
    m_bbMin = bbMin;
    m_bbMax = bbMax;

    // compute image size
    m_imageSize = static_cast<vec3s>(ceil((bbMax - bbMin) / resolution)) + 1;
    m_imageSize.x = std::max(m_imageSize.x, (size_t)1);
    m_imageSize.y = std::max(m_imageSize.y, (size_t)1);
    m_imageSize.z = std::max(m_imageSize.z, (size_t)1);

    // create buffers
    size_t numelBuffers = m_imageSize.x * m_imageSize.y * m_imageSize.z;

    m_mask.resize(numelBuffers);
    m_sampleIdx.resize(numelBuffers);
    m_weightX.resize(numelBuffers);
    m_weightY.resize(numelBuffers);

    if (1) {
      // 2D is computed on the cpu at the moment -> copy

      vec2 bb2DMin{m_bbMin.x, m_bbMin.y};
      assert(layout.x > 1);
      // From now on, we assume everything is in the xy-plane
      // -----------------------------------------
      for (size_t scanlineIdxY = 0; scanlineIdxY < layout.y; scanlineIdxY++) {
        //#pragma omp parallel for schedule(dynamic, 8)
        for (int scanlineIdxX = 0; scanlineIdxX < layout.x - 1;
             scanlineIdxX++) {
          vec3 start3 =
              scanlines[scanlineIdxX][scanlineIdxY].getPoint(startDepth);
          vec3 startN3 =
              scanlines[scanlineIdxX + 1][scanlineIdxY].getPoint(startDepth);
          vec3 end3 = scanlines[scanlineIdxX][scanlineIdxY].getPoint(endDepth);
          vec3 endN3 =
              scanlines[scanlineIdxX + 1][scanlineIdxY].getPoint(endDepth);
          vec2 start = {start3.x, start3.y};
          vec2 startN = {startN3.x, startN3.y};
          vec2 end = {end3.x, end3.y};
          vec2 endN = {endN3.x, endN3.y};

          // find bounding box of the two scanlines
          vec2 quadMinWorld = {
              std::min(std::min(std::min(start.x, startN.x), end.x), endN.x),
              std::min(std::min(std::min(start.y, startN.y), end.y), endN.y)};
          vec2 quadMaxWorld = {
              std::max(std::max(std::max(start.x, startN.x), end.x), endN.x),
              std::max(std::max(std::max(start.y, startN.y), end.y), endN.y)};

          vec2s quadMinPixel =
              static_cast<vec2s>(floor((quadMinWorld - bb2DMin) / resolution));
          vec2s quadMaxPixel =
              static_cast<vec2s>(ceil((quadMaxWorld - bb2DMin) / resolution));

          // check the pixels in the quad bounding box and mark the inside ones
          vec2s pixel;
          for (pixel.x = quadMinPixel.x; pixel.x <= quadMaxPixel.x; pixel.x++) {
            for (pixel.y = quadMinPixel.y; pixel.y <= quadMaxPixel.y;
                 pixel.y++) {
              vec2 pixelPos = static_cast<vec2>(pixel) * resolution + bb2DMin;
              if (pointInsideTriangle(endN, end, start, pixelPos) ||
                  pointInsideTriangle(start, startN, endN, pixelPos)) {
                m_mask[pixel.x + pixel.y * m_imageSize.x] = 1;

                vec2 params = mapToParameters2D(
                    scanlines[scanlineIdxX][scanlineIdxY].position,
                    scanlines[scanlineIdxX + 1][scanlineIdxY].position,
                    scanlines[scanlineIdxX][scanlineIdxY].direction,
                    scanlines[scanlineIdxX + 1][scanlineIdxY].direction,
                    startDepth, endDepth, {pixelPos.x, pixelPos.y, 0.0});
                double t = params.x;
                double d = params.y;

                uint32_t sampleIdxScanline =
                    static_cast<uint32_t>(std::floor(d / SampleDistance));
                float weightY = static_cast<float>(
                    d - (sampleIdxScanline * SampleDistance));
                float weightX = static_cast<float>(t);

                uint32_t sampleIdx = static_cast<uint32_t>(
                    sampleIdxScanline * NumScanlines + scanlineIdxX +
                    scanlineIdxY * layout.x);

                m_sampleIdx[pixel.x + pixel.y * m_imageSize.x] = sampleIdx;
                m_weightX[pixel.x + pixel.y * m_imageSize.x] = weightX;
                m_weightY[pixel.x + pixel.y * m_imageSize.x] = weightY;
              }
            }
          }
        }
      }
    }
  }
}

Beamforming2D::Beamforming2D() {
  Width = 0;
  Height = 0;

  rxScanlines = NULL;
  rxDepths = NULL;
  rxElementXs = NULL;
  rxElementYs = NULL;
  window_data = NULL;
  RFdata = NULL;

  dt = 1.0 / 40000000.0;
  additionalOffset = 0;
  fNumber = 1;
  speedOfSound = 0;
  interpolateRFlines = true;
  interpolateBetweenTransmits = false;

  windowParameter = 0.5;
  numEntriesPerFunction = 64;
  window_scale = 31.5;
}

Beamforming2D::~Beamforming2D() {
  if (rxScanlines) free(rxScanlines);
  if (rxDepths) free(rxDepths);
  if (rxElementXs) free(rxElementXs);
  if (rxElementYs) free(rxElementYs);
  if (window_data) free(window_data);
  if (RFdata) free(RFdata);
  if (s) free(s);
  if (s_tmp) free(s_tmp);
  if (RFdata_shuffle) free(RFdata_shuffle);
  if (RFdata_shuffle_s1) free(RFdata_shuffle_s1);
  if (RFdata_shuffle_s2) free(RFdata_shuffle_s2);
  if (rxScanlines_dev) sycl::free(rxScanlines_dev, q);
  if (rxDepths_dev) sycl::free(rxDepths_dev, q);
  if (rxElementXs_dev) sycl::free(rxElementXs_dev, q);
  if (rxElementYs_dev) sycl::free(rxElementYs_dev, q);
  if (window_data_dev) sycl::free(window_data_dev, q);

  if (m_mask) sycl::free(m_mask, q);
  if (m_sampleIdx) sycl::free(m_sampleIdx, q);
  if (m_weightX) sycl::free(m_weightX, q);
  if (m_weightY) sycl::free(m_weightY, q);
  if (RFdata_dev) sycl::free(RFdata_dev, q);
  if (s_dev) sycl::free(s_dev, q);
}

int Beamforming2D::GetInputImage(const char *Paramfilename,
                                 const char *Inputfilename,
                                 BeamformingType type) {
  dt = 1.0 / 40000000.0;
  additionalOffset = 0;
  fNumber = 1;
  speedOfSound = 0;
  interpolateRFlines = true;
  interpolateBetweenTransmits = false;

  windowParameter = 0.5;
  numEntriesPerFunction = 64;
  // BF type: { DelayAndSum, DelayAndStddev, TestSignal }
  mBeamformingType = type;

  // read parameters file
  std::ifstream f(Paramfilename);
  std::string dummy;
  int version;
  if (f.is_open()) {
    f >> dummy;
    f >> dummy;
    f >> version;

    f >> numElements;
    f >> elementLayout.x;
    f >> elementLayout.y;
    f >> numReceivedChannels;
    f >> numSamples;
    f >> numTxScanlines;
    f >> scanlineLayout.x;
    f >> scanlineLayout.y;
    f >> depth;
    f >> samplingFrequency;
    f >> rxNumDepths;
    f >> speedOfSoundMMperS;
  } else {
    printf("input file open fail\n");
    return -1;
  }

  // read input data file
  ifstream rf(Inputfilename, ios::in | ios::binary);
  int len = numReceivedChannels * numSamples * numTxScanlines;
  int valid_bytes = len * sizeof(int16_t);
  RFdata = (int16_t *)malloc(valid_bytes * 8);
  rf.read((char *)RFdata, valid_bytes * 8);
  rf.close();

  // generate window data
  window_data = (float *)malloc(numEntriesPerFunction * sizeof(float));
  for (uint i = 0; i < numEntriesPerFunction; i++) window_data[i] = 0;
  WindowData(WindowType::WindowHamming, windowParameter, numEntriesPerFunction,
             window_data);

  // set output width and height
  numRxScanlines = scanlineLayout.x * scanlineLayout.y;
  Width = numRxScanlines;
  Height = rxNumDepths;
  speedOfSound = speedOfSoundMMperS;

  rxScanlines = (ScanlineRxParameters3D *)malloc(
      numRxScanlines * sizeof(ScanlineRxParameters3D));
  vector<vector<ScanlineRxParameters3D>> origin_rxScanlines(
      scanlineLayout.x, vector<ScanlineRxParameters3D>(scanlineLayout.y));

  rxDepths = (float *)malloc(rxNumDepths * sizeof(float));
  rxElementXs = (float *)malloc(numElements * sizeof(float));
  rxElementYs = (float *)malloc(numElements * sizeof(float));
  for (uint i = 0; i < rxNumDepths; i++) rxDepths[i] = 0;
  for (uint i = 0; i < numElements; i++) rxElementXs[i] = 0;
  for (uint i = 0; i < numElements; i++) rxElementYs[i] = 0;

  vector<vector<ScanlineRxParameters3D>> local_rxScanlines(
      scanlineLayout.x, vector<ScanlineRxParameters3D>(scanlineLayout.y));

  for (size_t idxY = 0; idxY < scanlineLayout.y; idxY++) {
    for (size_t idxX = 0; idxX < scanlineLayout.x; idxX++) {
      ScanlineRxParameters3D params;

      f >> params;
      local_rxScanlines[idxX][idxY] = params;
      origin_rxScanlines[idxX][idxY] = params;
    }
  }

  for (size_t idx = 0; idx < rxNumDepths; idx++) {
    float val;
    f >> val;
    rxDepths[idx] = val;
  }
  for (size_t idx = 0; idx < numElements; idx++) {
    float val;
    f >> val;
    rxElementXs[idx] = val;
  }
  dt = 2.5e-08;

  convertToDtSpace(dt, speedOfSoundMMperS, numElements, numRxScanlines,
                   rxNumDepths, local_rxScanlines, rxDepths, rxElementXs);

  size_t scanlineIdx = 0;
  for (size_t idxY = 0; idxY < scanlineLayout.y; idxY++) {
    for (size_t idxX = 0; idxX < scanlineLayout.x; idxX++) {
      rxScanlines[scanlineIdx].position.x =
          local_rxScanlines[idxX][idxY].position.x;
      rxScanlines[scanlineIdx].position.y =
          local_rxScanlines[idxX][idxY].position.y;
      rxScanlines[scanlineIdx].position.z =
          local_rxScanlines[idxX][idxY].position.z;

      rxScanlines[scanlineIdx].direction.x =
          local_rxScanlines[idxX][idxY].direction.x;
      rxScanlines[scanlineIdx].direction.y =
          local_rxScanlines[idxX][idxY].direction.y;
      rxScanlines[scanlineIdx].direction.z =
          local_rxScanlines[idxX][idxY].direction.z;

      rxScanlines[scanlineIdx].txWeights[0] =
          local_rxScanlines[idxX][idxY].txWeights[0];
      rxScanlines[scanlineIdx].txWeights[1] =
          local_rxScanlines[idxX][idxY].txWeights[1];
      rxScanlines[scanlineIdx].txWeights[2] =
          local_rxScanlines[idxX][idxY].txWeights[2];
      rxScanlines[scanlineIdx].txWeights[3] =
          local_rxScanlines[idxX][idxY].txWeights[3];

      rxScanlines[scanlineIdx].txParameters[0].firstActiveElementIndex.x =
          local_rxScanlines[idxX][idxY]
              .txParameters[0]
              .firstActiveElementIndex.x;
      rxScanlines[scanlineIdx].txParameters[0].firstActiveElementIndex.y =
          local_rxScanlines[idxX][idxY]
              .txParameters[0]
              .firstActiveElementIndex.y;
      rxScanlines[scanlineIdx].txParameters[0].lastActiveElementIndex.x =
          local_rxScanlines[idxX][idxY]
              .txParameters[0]
              .lastActiveElementIndex.x;
      rxScanlines[scanlineIdx].txParameters[0].lastActiveElementIndex.y =
          local_rxScanlines[idxX][idxY]
              .txParameters[0]
              .lastActiveElementIndex.y;
      rxScanlines[scanlineIdx].txParameters[0].txScanlineIdx =
          local_rxScanlines[idxX][idxY].txParameters[0].txScanlineIdx;
      rxScanlines[scanlineIdx].txParameters[0].initialDelay =
          local_rxScanlines[idxX][idxY].txParameters[0].initialDelay;

      rxScanlines[scanlineIdx].txParameters[1].firstActiveElementIndex.x =
          local_rxScanlines[idxX][idxY]
              .txParameters[1]
              .firstActiveElementIndex.x;
      rxScanlines[scanlineIdx].txParameters[1].firstActiveElementIndex.y =
          local_rxScanlines[idxX][idxY]
              .txParameters[1]
              .firstActiveElementIndex.y;
      rxScanlines[scanlineIdx].txParameters[1].lastActiveElementIndex.x =
          local_rxScanlines[idxX][idxY]
              .txParameters[1]
              .lastActiveElementIndex.x;
      rxScanlines[scanlineIdx].txParameters[1].lastActiveElementIndex.y =
          local_rxScanlines[idxX][idxY]
              .txParameters[1]
              .lastActiveElementIndex.y;
      rxScanlines[scanlineIdx].txParameters[1].txScanlineIdx =
          local_rxScanlines[idxX][idxY].txParameters[1].txScanlineIdx;
      rxScanlines[scanlineIdx].txParameters[1].initialDelay =
          local_rxScanlines[idxX][idxY].txParameters[1].initialDelay;

      rxScanlines[scanlineIdx].txParameters[2].firstActiveElementIndex.x =
          local_rxScanlines[idxX][idxY]
              .txParameters[2]
              .firstActiveElementIndex.x;
      rxScanlines[scanlineIdx].txParameters[2].firstActiveElementIndex.y =
          local_rxScanlines[idxX][idxY]
              .txParameters[2]
              .firstActiveElementIndex.y;
      rxScanlines[scanlineIdx].txParameters[2].lastActiveElementIndex.x =
          local_rxScanlines[idxX][idxY]
              .txParameters[2]
              .lastActiveElementIndex.x;
      rxScanlines[scanlineIdx].txParameters[2].lastActiveElementIndex.y =
          local_rxScanlines[idxX][idxY]
              .txParameters[2]
              .lastActiveElementIndex.y;
      rxScanlines[scanlineIdx].txParameters[2].txScanlineIdx =
          local_rxScanlines[idxX][idxY].txParameters[2].txScanlineIdx;
      rxScanlines[scanlineIdx].txParameters[2].initialDelay =
          local_rxScanlines[idxX][idxY].txParameters[2].initialDelay;

      rxScanlines[scanlineIdx].txParameters[3].firstActiveElementIndex.x =
          local_rxScanlines[idxX][idxY]
              .txParameters[3]
              .firstActiveElementIndex.x;
      rxScanlines[scanlineIdx].txParameters[3].firstActiveElementIndex.y =
          local_rxScanlines[idxX][idxY]
              .txParameters[3]
              .firstActiveElementIndex.y;
      rxScanlines[scanlineIdx].txParameters[3].lastActiveElementIndex.x =
          local_rxScanlines[idxX][idxY]
              .txParameters[3]
              .lastActiveElementIndex.x;
      rxScanlines[scanlineIdx].txParameters[3].lastActiveElementIndex.y =
          local_rxScanlines[idxX][idxY]
              .txParameters[3]
              .lastActiveElementIndex.y;
      rxScanlines[scanlineIdx].txParameters[3].txScanlineIdx =
          local_rxScanlines[idxX][idxY].txParameters[3].txScanlineIdx;
      rxScanlines[scanlineIdx].txParameters[3].initialDelay =
          local_rxScanlines[idxX][idxY].txParameters[3].initialDelay;

      rxScanlines[scanlineIdx].maxElementDistance.x =
          local_rxScanlines[idxX][idxY].maxElementDistance.x;
      rxScanlines[scanlineIdx].maxElementDistance.y =
          local_rxScanlines[idxX][idxY].maxElementDistance.y;

      scanlineIdx++;
    }
  }

  std::cout << "Get Input Finished. " << std::endl;

  vector<uint8_t> mask;
  vector<float> weightX;
  vector<float> weightY;
  vector<uint32_t> sampleIdx;
  m_imageSize = {0, 0, 0};

  updateInternals(mask, weightX, weightY, sampleIdx, m_imageSize,
                  scanlineLayout, depth, origin_rxScanlines, rxNumDepths,
                  numRxScanlines);

  m_mask = (uint8_t *)sycl::malloc_device(mask.size() * sizeof(uint8_t), q);
  m_sampleIdx =
      (uint32_t *)sycl::malloc_device(sampleIdx.size() * sizeof(uint32_t), q);
  m_weightX = (float *)sycl::malloc_device(weightX.size() * sizeof(float), q);
  m_weightY = (float *)sycl::malloc_device(weightY.size() * sizeof(float), q);

  uint8_t *mask_shuffle = (uint8_t *)malloc(mask.size() * sizeof(uint8_t));
  float *weightX_shuffle = (float *)malloc(weightX.size() * sizeof(float));
  float *weightY_shuffle = (float *)malloc(weightY.size() * sizeof(float));

  shuffle_image(mask_shuffle, mask.data(), 2000, 1667);
  shuffle_image(weightX_shuffle, weightX.data(), 2000, 1667);
  shuffle_image(weightY_shuffle, weightY.data(), 2000, 1667);

  q.memcpy(m_mask, mask_shuffle, mask.size() * sizeof(uint8_t)).wait();
  q.memcpy(m_sampleIdx, sampleIdx.data(), sampleIdx.size() * sizeof(uint32_t))
      .wait();
  q.memcpy(m_weightX, weightX_shuffle, weightX.size() * sizeof(float)).wait();
  q.memcpy(m_weightY, weightY_shuffle, weightY.size() * sizeof(float)).wait();

  delete mask_shuffle;
  delete weightX_shuffle;
  delete weightY_shuffle;

  s_dev = (float *)sycl::malloc_device(
      rxNumDepths * numRxScanlines * sizeof(float), q);
  s = (float *)malloc(rxNumDepths * numRxScanlines * sizeof(float));

  RFdata_dev = (unsigned char *)sycl::malloc_device(
      numReceivedChannels * numSamples * numTxScanlines * sizeof(unsigned char),
      q);
  RFdata_dev1 = (unsigned char *)sycl::malloc_device(
      numReceivedChannels * numSamples * numTxScanlines * sizeof(unsigned char),
      q);
  RFdata_shuffle = (int16_t *)malloc(len * sizeof(int16_t));
  RFdata_shuffle_s1 = (unsigned char *)malloc(len * sizeof(unsigned char));
  RFdata_shuffle_s2 = (unsigned char *)malloc(len * sizeof(unsigned char));

  s_tmp = (float *)malloc(rxNumDepths * numRxScanlines * sizeof(float));

  return 1;
}

int Beamforming2D::copy_data2dev() {
  rxDepths_dev = (float *)sycl::malloc_device(rxNumDepths * sizeof(float), q);
  if (rxDepths_dev == nullptr) {
    malloc_mem_log(std::string("rxDepths_dev"));
  }

  rxElementXs_dev =
      (float *)sycl::malloc_device(numElements * sizeof(float), q);
  rxElementYs_dev =
      (float *)sycl::malloc_device(numElements * sizeof(float), q);
  if (rxElementXs_dev == nullptr || rxElementYs_dev == nullptr) {
    malloc_mem_log(std::string("rxElementXs_dev"));
  }

  window_data_dev =
      (float *)sycl::malloc_device(numEntriesPerFunction * sizeof(float), q);
  if (window_data_dev == nullptr) {
    malloc_mem_log(std::string("window_data_dev"));
  }

  rxScanlines_dev = (ScanlineRxParameters3D *)sycl::malloc_device(
      numRxScanlines * sizeof(ScanlineRxParameters3D), q);
  if (rxScanlines_dev == nullptr) {
    malloc_mem_log(std::string("rxScanlines_dev"));
  }

  q.memcpy(rxDepths_dev, rxDepths, rxNumDepths * sizeof(float)).wait();
  q.memcpy(rxElementXs_dev, rxElementXs, numElements * sizeof(float)).wait();
  q.memcpy(rxElementYs_dev, rxElementYs, numElements * sizeof(float)).wait();
  q.memcpy(window_data_dev, window_data, numEntriesPerFunction * sizeof(float))
      .wait();
  q.memcpy(rxScanlines_dev, rxScanlines,
           numRxScanlines * sizeof(ScanlineRxParameters3D))
      .wait();

  return 1;
}

int Beamforming2D::read_one_frame2dev(int16_t *raw_ptr, size_t len) {
  if (len == 0) {
    std::cout << "Error raw data frame size.\n";
    return 0;
  }
  shuffle_RF_order<int16_t>(RFdata_shuffle, raw_ptr);

  for (size_t i = 0; i < len; i++) {
    RFdata_shuffle_s1[i] = (unsigned char)((RFdata_shuffle[i] >> 8) & 0xff);
    RFdata_shuffle_s2[i] = (unsigned char)(RFdata_shuffle[i] & 0xff);
  }

  q.memcpy(RFdata_dev, RFdata_shuffle_s1, len * sizeof(unsigned char)).wait();
  q.memcpy(RFdata_dev1, RFdata_shuffle_s2, len * sizeof(unsigned char)).wait();

  return 1;
}

sycl::queue Beamforming2D::getStream() { return q; }

float *Beamforming2D::getRes() { return s_tmp; }

float *Beamforming2D::getResHost() {
  q.memcpy(s_tmp, s_dev, rxNumDepths * numRxScanlines * sizeof(float)).wait();

  shuffle_image<float>(s, s_tmp);
  return s;
}

HilbertFirEnvelope::HilbertFirEnvelope(sycl::queue hq, float *input_addr)
    : q(hq), input_dev(input_addr) {
  m_numScanlines = 255;
  m_numSamples = 2000;
  m_filterLength = 65;
  output = (float *)malloc(m_numScanlines * m_numSamples * sizeof(float));
  output_dev = (float *)sycl::malloc_device(
      m_numScanlines * m_numSamples * sizeof(float), q);
  output_tmp = (float *)malloc(2000 * 255 * sizeof(float));
  prepareFilter();
}

HilbertFirEnvelope::HilbertFirEnvelope(sycl::queue hq) : q(hq) {
  m_numScanlines = 255;
  m_numSamples = 2000;
  m_filterLength = 65;
  output = (float *)malloc(m_numScanlines * m_numSamples * sizeof(float));
  output_dev = (float *)sycl::malloc_device(
      m_numScanlines * m_numSamples * sizeof(float), q);
  output_tmp = (float *)malloc(2000 * 255 * sizeof(float));

  prepareFilter();
}

void HilbertFirEnvelope::prepareFilter() {
  m_hilbertFilter = FirFilterFactory::createFilter<float>(
      m_filterLength, FirFilterFactory::FilterTypeHilbertTransformer,
      FirFilterFactory::FilterWindowHamming);

  m_hilbertFilter_dev =
      (float *)sycl::malloc_device(m_filterLength * sizeof(float), q);

  q.memcpy(m_hilbertFilter_dev, m_hilbertFilter, m_filterLength * sizeof(float))
      .wait();
}

float *HilbertFirEnvelope::getRes() { return output_dev; }

float *HilbertFirEnvelope::getResHost() {
  q.memcpy(output_tmp, output_dev, 2000 * 255 * sizeof(float)).wait();
  shuffle_image(output, output_tmp);
  return output;
}

HilbertFirEnvelope::~HilbertFirEnvelope()
{
  if(m_hilbertFilter) free(m_hilbertFilter);
  if(m_hilbertFilter_dev) sycl::free(m_hilbertFilter_dev, q);
  if(output_dev) sycl::free(output_dev, q);
  if(output_tmp) free(output_tmp);
  if(output) free(output);
}
template <typename In, typename Out, typename WorkType>
struct thrustLogcompress {
  WorkType _inScale;
  WorkType _scaleOverDenominator;

  // Thrust functor that computes
  // signal = log10(1 + a*signal)./log10(1 + a)
  // of the downscaled (_inMax) input signal
  thrustLogcompress(double dynamicRange, In inMax, Out outMax, double scale)
      : _inScale(static_cast<WorkType>(dynamicRange / inMax)),
        _scaleOverDenominator(static_cast<WorkType>(
            scale * outMax / sycl::log10(dynamicRange + 1))){};

  Out operator()(const In &a) const {
    WorkType val = sycl::log10(std::abs(static_cast<WorkType>(a)) * _inScale +
                               (WorkType)1) *
                   _scaleOverDenominator;
    return clampCast<Out>(val);
  }
};

LogCompressor::LogCompressor(float *input, sycl::queue in_q)
    : q(in_q), input_dev(input) {
  output = (float *)malloc(2000 * 255 * sizeof(float));
  output_dev =
      (float *)sycl::malloc_device(2000 * 255 * sizeof(float), q);
  output_tmp = (float *)malloc(2000 * 255 * sizeof(float));
}

LogCompressor::LogCompressor(sycl::queue in_q) : q(in_q) {
  output = (float *)malloc(2000 * 255 * sizeof(float));
  output_dev =
      (float *)sycl::malloc_device(2000 * 255 * sizeof(float), q);
  output_tmp = (float *)malloc(2000 * 255 * sizeof(float));
}

LogCompressor::~LogCompressor() {
  if (output_dev) sycl::free(output_dev, q);
  if (output) free(output);
  if (output_tmp) free(output_tmp);
}

float *LogCompressor::getRes() { return output_dev; }

float *LogCompressor::getResHost() {
  q.memcpy(output_tmp, output_dev, 2000 * 255 * sizeof(float)).wait();
  shuffle_image(output, output_tmp);

  return output;
}

ScanConverter::ScanConverter(sycl::queue hq, uint8_t* mask,
                uint32_t* sampleIdx, float* weightX, float* weightY,
                vec3s imageSize)
    : q(hq),
      m_mask(mask),
      m_sampleIdx(sampleIdx),
      m_weightX(weightX),
      m_weightY(weightY),
      m_imageSize(imageSize) {
  output_tmp = (float *)malloc(2000 * 1667 * sizeof(float));
  output_dev = (float *)sycl::malloc_device(
      m_imageSize.x * m_imageSize.y * m_imageSize.z * sizeof(float), q);
  output = (float *)malloc(m_imageSize.x * m_imageSize.y * m_imageSize.z *
                                sizeof(float));
  output_tmp = (float *)malloc(m_imageSize.x * m_imageSize.y * m_imageSize.z *
                                sizeof(float));
}

uint8_t *ScanConverter::getMask() { return m_mask; }

void ScanConverter::SubmitKernel() { convert2D<float, float>(); }

float *ScanConverter::getRes() { return output_dev; }
float *ScanConverter::getResHost() 
{
  q.memcpy(output_tmp, output_dev, 2000 * 1667 * sizeof(float)).wait();
  shuffle_image(output, output_tmp, 1667, 2000);

  return output;
}

ScanConverter::~ScanConverter(){
  if (output_dev) free(output_dev, q);
  if (output) free(output);
  if (output_tmp) free(output_tmp);
}

/////////////////// Kernel Function ////////////////////////

#define MAX_SAMPLES 4096
#define MAX_RECEIVED_CHANNELS 64
#define MAX_TX_SCANLINES 128
#define MAX_X_ELEMSDT 128
#define MAX_RX_SCANLINES 256
#define MAX_TIMESTEPS 2337

template <bool interpolateRFlines, bool interpolateBetweenTransmits,
          typename RFType, typename ResultType,
          typename LocationType>
void rxBeamformingDTSPACEKernel(
    size_t numTransducerElements,
    size_t numReceivedChannels,
    size_t numTimesteps,
    unsigned char *__restrict__ RF, unsigned char *__restrict__ RF1,
    size_t numTxScanlines,
    size_t numRxScanlines,
    ScanlineRxParameters3D *__restrict__ scanlinesDT,
    size_t numDs,
    LocationType *__restrict__ dsDT, LocationType *__restrict__ xElemsDT,
    LocationType speedOfSound, LocationType dt, uint32_t additionalOffset,
    LocationType F, float *windowAcc, const float windowFunctionScale,
    ResultType *__restrict__ s) {
  ScanlineRxParameters3D localScanlinesDT[MAX_RX_SCANLINES];

  for (int i = 0; i < numRxScanlines; i++) {
    localScanlinesDT[i] = scanlinesDT[i];
  }

  LocationType localDsDT[MAX_SAMPLES];
  for (int i = 0; i < numDs; i++) {
    localDsDT[i] = dsDT[i];
  }

  float localWindowFunction[64];
  for (int i = 0; i < 64; i++) {
    localWindowFunction[i] = windowAcc[i];
  }

  float local_x_elemsDT[MAX_X_ELEMSDT];
  for (int i = 0; i < MAX_X_ELEMSDT; i++) {
    local_x_elemsDT[i] = xElemsDT[i];
  }

  [[
    intel::numbanks(MAX_RECEIVED_CHANNELS), intel::bankwidth(2),
    intel::singlepump
  ]] RFType rf1[MAX_SAMPLES * MAX_RECEIVED_CHANNELS];
  [[
    intel::numbanks(MAX_RECEIVED_CHANNELS), intel::bankwidth(2),
    intel::singlepump
  ]] RFType rf2[MAX_SAMPLES * MAX_RECEIVED_CHANNELS];

  bool rf1Used = false;
#ifndef FAKEDATA
  // local initial data
  // assume that the data is interleaved across the channels
  for (uint32_t i = 0; i < numTimesteps; i++) {
    int baseAddr = i * MAX_RECEIVED_CHANNELS;
    int rfbaseAddr = i * MAX_RECEIVED_CHANNELS;
#pragma unroll
    for (int j = 0; j < MAX_RECEIVED_CHANNELS; j++) {
      // still load data even the j is bigger than numReceviedChannels
      // even we won't use them at all
      int16_t rf_tmp = RF[baseAddr + j];
      rf_tmp = (rf_tmp << 8) | RF1[baseAddr + j];
      rf1[rfbaseAddr + j] = rf_tmp;
    }
  }
#endif

  int sampleIndex = 0;

  for (uint32_t i = 0; i < numTimesteps * numRxScanlines; i++) {
    uint32_t r = i % MAX_TIMESTEPS;
    uint32_t scanlineIdx = i / MAX_TIMESTEPS;

    uint32_t nextScanlineIdx = scanlineIdx + 1;
    uint32_t nextTxScanlinIdx =
        localScanlinesDT[nextScanlineIdx].txParameters[0].txScanlineIdx;

    // reverse rf at the beginning of each scanline
    rf1Used = r == 0 ? (!rf1Used) : rf1Used;

    // load the next scanline data
    int baseAddr =
        (nextTxScanlinIdx * numTimesteps + r) * MAX_RECEIVED_CHANNELS;

    RFType RF_tmp[MAX_RECEIVED_CHANNELS];

    // Fake data is for testing the performance without DDR bandwidth limit.
    // As in ultrasound applications, FPGA will be connected with ultrasound probe and fetching DDR to get input data is not necessary.
#ifdef FAKEDATA
#pragma unroll
    for (int j = 0; j < MAX_RECEIVED_CHANNELS; j++) {
      RF_tmp[j] = j % 2 + 1;
    }
#else
#pragma unroll
    for (int j = 0; j < MAX_RECEIVED_CHANNELS; j++) {
      int16_t rf_tmp = RF[baseAddr + j];
      rf_tmp = (rf_tmp << 8) | RF1[baseAddr + j];
      RF_tmp[j] = rf_tmp;
    }
#endif
    if (rf1Used) {
#pragma unroll
      for (int j = 0; j < MAX_RECEIVED_CHANNELS; j++) {
        rf2[r * MAX_RECEIVED_CHANNELS + j] = RF_tmp[j];
      }
    } else {
#pragma unroll
      for (int j = 0; j < MAX_RECEIVED_CHANNELS; j++) {
        rf1[r * MAX_RECEIVED_CHANNELS + j] = RF_tmp[j];
      }
    }

    if (r < numDs) {
      LocationType d = localDsDT[r];
      LocationType aDT =
          computeAperture_D(F, d * dt * speedOfSound) / speedOfSound / dt;
      ScanlineRxParameters3D scanline = localScanlinesDT[scanlineIdx];
      LocationType scanlineX = scanline.position.x;
      LocationType dirX = scanline.direction.x;
      LocationType dirY = scanline.direction.y;
      LocationType dirZ = scanline.direction.z;
      LocationType maxElementDistance =
          static_cast<LocationType>(scanline.maxElementDistance.x);
      LocationType invMaxElementDistance =
          1 / sycl::fmin(aDT, maxElementDistance);

      float sInterp = 0.0f;

      ScanlineRxParameters3D::TransmitParameters txParams =
          scanline.txParameters[0];
      float sLocal = 0.0f;

      float sample = 0.0f;
      float weightAcum = 0.0f;
      int numAdds = 0;
      LocationType initialDelay = txParams.initialDelay;

#pragma unroll
      for (int j = 0; j < MAX_RECEIVED_CHANNELS; j++) {
        int32_t elemIdxX_possible = j + numReceivedChannels;
        int32_t elemIdxX = j;

        if (elemIdxX_possible >= txParams.firstActiveElementIndex.x &&
            elemIdxX_possible < txParams.lastActiveElementIndex.x) {
          elemIdxX = elemIdxX_possible;
        }

        bool valid_elem = true;

        if (elemIdxX < txParams.firstActiveElementIndex.x ||
            elemIdxX >= txParams.lastActiveElementIndex.x) {
          valid_elem = false;
          elemIdxX = 0;
        }

        LocationType xElem = local_x_elemsDT[elemIdxX];

        float relativeIndex = (xElem - scanlineX) * invMaxElementDistance;
        float relativeIndexClamped =
            sycl::fmin(sycl::fmax(relativeIndex, -1.0f), 1.0f);
        float absoluteIndex =
            windowFunctionScale * (relativeIndexClamped + 1.0f);
        uint32_t absoluteIndex_int = static_cast<uint32_t>(absoluteIndex);
        absoluteIndex_int = absoluteIndex_int > 63 ? 63 : absoluteIndex_int;
        float weight = localWindowFunction[absoluteIndex_int];

        LocationType delayf =
            initialDelay +
            computeDelayDTSPACE_D(dirX, dirY, dirZ, xElem, scanlineX, d) +
            additionalOffset;
        int32_t delay = static_cast<int32_t>(sycl::floor(delayf));
        delayf -= delay;
        delay = delay & (MAX_SAMPLES - 1);

        RFType rfData = 0.0;
        RFType rfData2 = 0.0;

        if (rf1Used) {
          rfData = rf1[delay * MAX_RECEIVED_CHANNELS + j];
        } else {
          rfData = rf2[delay * MAX_RECEIVED_CHANNELS + j];
        }

        if (sycl::fabs(xElem - scanlineX) <= aDT && valid_elem) {
          if (delay < numTimesteps) {
            sample += weight * rfData;
          }
          weightAcum += weight;
          numAdds++;
        }
      }

      if (numAdds > 0) {
        sLocal = sample / weightAcum * numAdds;
      }

      sInterp += sLocal;

      float res = clampCast<float>(sInterp);
#ifdef STORE
      s[sampleIndex] = res;
#endif

      bf_pipe::write(res);
      sampleIndex = sampleIndex + 1;
    }
  }
}

void Beamforming2D::SubmitKernel() {
  size_t p_numElements = numElements;
  size_t p_numReceivedChannels = numReceivedChannels;
  size_t p_numSamples = numSamples;
  size_t p_numTxScanlines = numTxScanlines;
  size_t p_numRxScanlines = numRxScanlines;
  ScanlineRxParameters3D *p_rxScanlines_dev = rxScanlines_dev;
  size_t p_rxNumDepths = rxNumDepths;

  float *p_rxDepths_dev = rxDepths_dev;
  float *p_rxElementXs_dev = rxElementXs_dev;

  float p_speedOfSoundMMperS = speedOfSoundMMperS;
  float p_dt = dt;
  float p_additionalOffset = additionalOffset;
  float p_fNumber = fNumber;
  float *p_window_data_dev = window_data_dev;
  float p_window_scale = window_scale;
  float *p_s_dev = s_dev;
  unsigned char *p_RFdata_dev = RFdata_dev;
  unsigned char *p_RFdata_dev1 = RFdata_dev1;

  e = q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class BeamForming>([=]()[[intel::kernel_args_restrict]] {
      rxBeamformingDTSPACEKernel<true, false, int16_t, float, float>(
          p_numElements, p_numReceivedChannels, p_numSamples, p_RFdata_dev,
          p_RFdata_dev1, p_numTxScanlines, p_numRxScanlines, p_rxScanlines_dev,
          p_rxNumDepths, p_rxDepths_dev, p_rxElementXs_dev,
          p_speedOfSoundMMperS, p_dt, p_additionalOffset, p_fNumber,
          p_window_data_dev, p_window_scale, p_s_dev);
    });
  });
}

#define ROW 2000
#define COL 255
#define HALF_LENGTH 32
#define NUMSCANLINES 255
#define FILTER_LENGTH 65
const int H_VEC_SIZE = 4;

template <typename InputType, typename OutputType>
void kernelFilterDemodulation(
    const HilbertFirEnvelope::WorkType *__restrict__ filter,
    OutputType *__restrict__ out, const int numSamples, const int numScanlines,
    const int filterLength) {
  float last_window_register;
  float sample_list;
  float filterElement;
  float accumulator;
  float signalValue;
  float local_filter[FILTER_LENGTH];

  float last_window_pp[FILTER_LENGTH - 1];

  for (int i = 0; i < FILTER_LENGTH; i++) {
    local_filter[i] = filter[i];
  }

  for (int kl = 0; kl < COL; kl++) {
    for (int i = 0; i < HALF_LENGTH; i++) {
      last_window_pp[i] = 0;
      last_window_pp[HALF_LENGTH + i] = bf_pipe::read();
    }
    for (int kh = 0; kh < ROW; kh++) {
      accumulator = 0;

#pragma unroll
      for (int j = 0; j < FILTER_LENGTH - 1; j++) {
        sample_list = last_window_pp[j];
        filterElement = local_filter[j];
        accumulator += sample_list * filterElement;
      }

#pragma unroll
      for (int j = 0; j < FILTER_LENGTH - 2; j++) {
        last_window_pp[j] = last_window_pp[j + 1];
      }

      if ((kh + HALF_LENGTH) < ROW) {
        last_window_register = bf_pipe::read();
      } else {
        last_window_register = 0;
      }

      last_window_pp[FILTER_LENGTH - 2] = last_window_register;

      accumulator += last_window_register * local_filter[FILTER_LENGTH - 1];

      signalValue = last_window_pp[FILTER_LENGTH / 2];
      float res =
          sycl::sqrt(signalValue * signalValue + accumulator * accumulator);
#ifdef STORE
      out[kl * 2000 + kh] = res;
#endif
      he_pipe::write(res);
    }
  }
}

void HilbertFirEnvelope::SubmitKernel() {
  e = q.submit([&](sycl::handler &cgh) {
    auto m_hilbertFilter_get_ct0 = m_hilbertFilter_dev;
    auto pEnv_get_ct1 = output_dev;
    auto numSamples_ct2 = m_numSamples;
    auto numScanlines_ct3 = m_numScanlines;
    auto m_filterLength_ct4 = (int)m_filterLength;

    cgh.single_task<class HilbertEnvelope>([=
    ]()[[intel::kernel_args_restrict]] {
      kernelFilterDemodulation<float, float>(
          m_hilbertFilter_get_ct0, pEnv_get_ct1, numSamples_ct2, numScanlines_ct3,
          m_filterLength_ct4);
    });
  });
}

#define PARA_NUM 8
void LogCompressor::compress(vec3s size, double dynamicRange, double scale,
                             double inMax) {
  const float *inImageData = input_dev;

  float outMax;
  if (std::is_integral<float>::value) {
    outMax = std::numeric_limits<float>::max();
  } else if (std::is_floating_point<float>::value) {
    outMax = static_cast<float>(255.0);
  }

  thrustLogcompress<float, float, WorkType> c(
      sycl::pow<double>(10, (dynamicRange / 20)), static_cast<float>(inMax),
      outMax, scale);

  auto pComprGpu_t = output_dev;

  static std::chrono::duration<double, std::milli> log_total_duration(0);

  e = q.submit([&](sycl::handler &h) {
    h.single_task<class LogCompress>([=]()[[intel::kernel_args_restrict]] {
      const int reg = 2000 * 255;
      for (int i = 0; i < reg; i++) {
        float inData = he_pipe::read();
        float res_tmp = c(inData);
#ifdef STORE
        pComprGpu_t[i] = res_tmp;
#endif
        lc_pipe::write(res_tmp);
      }
    });
  });
}

void LogCompressor::SubmitKernel() {
  vec3s m_input_size;
  m_input_size.x = 255;
  m_input_size.y = 2000;
  m_input_size.z = 1;

  double m_dynamicRange = 80;

  double m_scale = 1;

  double inMax = 32600;

  compress(m_input_size, m_dynamicRange, m_scale, inMax);
}

#define BUFFER_SIZE 2000
const int OUTSIZE_COL = 1667;
const int OUTSIZE_ROW = 2000;
template <typename InputType, typename OutputType, typename WeightType,
          typename IndexType>
void scanConvert2D(uint32_t numScanlines,
                   uint32_t numSamples,
                   uint32_t width,
                   uint32_t height,
                   const uint8_t *__restrict__ mask,
                   const IndexType *__restrict__ sampleIdx,
                   const WeightType *__restrict__ weightX,
                   const WeightType *__restrict__ weightY,
                   OutputType *__restrict__ image)
{
  float buf[BUFFER_SIZE * 2];

  for (int i = 0; i < BUFFER_SIZE * 2; i++) buf[i] = lc_pipe::read();

  int init_pos = 0;

  int s_idx = 0;

  float buf0[4];
  buf0[0] = buf[0];
  buf0[1] = buf[1];
  buf0[2] = buf[BUFFER_SIZE];
  buf0[3] = buf[BUFFER_SIZE + 1];

  for(int row = 0; row < OUTSIZE_COL - 1; row++)
  {
    if(s_idx != sampleIdx[row] && s_idx < 253){
      for (int i = 0; i < BUFFER_SIZE; i++) {
        buf[i] = buf[i + BUFFER_SIZE];
        buf[i + BUFFER_SIZE] = lc_pipe::read();
      }
      s_idx++;
    }

    int bias = OUTSIZE_ROW * row;
#pragma unroll 16
    for (int col = 0; col < OUTSIZE_ROW; col++) {
      float wX = weightX[bias + col];
      float wY = weightY[bias + col];
      uint8_t s_mask = mask[bias + col];

      float val =
          s_mask * ((1 - wY) * ((1 - wX) * buf[col] + wX * buf[col + BUFFER_SIZE]) +
          wY * ((1 - wX) * buf[col + 1] + wX * buf[col + 1 + BUFFER_SIZE]));
      image[bias + col] = clampCast<OutputType>(val);
    }
  }

  int bias = OUTSIZE_ROW * (OUTSIZE_COL - 1);

  // After analysis of sampleIdx, we find that the calculation of last row of output depends on buf0.
#pragma unroll 16
  for(int i = 0; i < OUTSIZE_ROW; i++)
  {
    float wX = weightX[bias + i];
    float wY = weightY[bias + i];
    uint8_t s_mask = mask[bias + i];

    float val =
        s_mask * ((1 - wY) * ((1 - wX) * buf0[0] + wX * buf0[2]) +
        wY * ((1 - wX) * buf0[1] + wX * buf0[3]));
    image[bias + i] = clampCast<OutputType>(val);
  }
}

template <typename InputType, typename OutputType>
void ScanConverter::convert2D() {
  uint32_t numScanlines = 255;
  vec2s scanlineLayout = {255, 1};
  uint32_t numSamples = 2000;
  m_imageSize = {1667, 2000, 1};

  auto pConv = output_dev;

  static long scan_call_count = 0;

  e = q.submit([&](sycl::handler &cgh) {
    auto m_numScanlines_ct0 = numScanlines;
    auto m_numSamples_ct1 = numSamples;
    auto m_imageSize_x_ct2 = (uint32_t)m_imageSize.x;
    auto m_imageSize_y_ct3 = (uint32_t)m_imageSize.y;
    auto m_mask_get_ct4 = m_mask;
    auto m_sampleIdx_get_ct5 = m_sampleIdx;
    auto m_weightX_get_ct6 = m_weightX;
    auto m_weightY_get_ct7 = m_weightY;
    auto pConv_get_ct8 = pConv;

    cgh.single_task<class ScanConvert>([=]()[[intel::kernel_args_restrict]] {
      scanConvert2D<float, float, float, uint32_t>(
          m_numScanlines_ct0, m_numSamples_ct1, m_imageSize_x_ct2, m_imageSize_y_ct3,
          m_mask_get_ct4, m_sampleIdx_get_ct5, m_weightX_get_ct6,
          m_weightY_get_ct7, pConv_get_ct8);
    });
  });
}
