// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "BeamForming.h"

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

template <typename RFType>
void shuffle_RF_order(RFType *shuffle_RF, const RFType *RF,
                      size_t numTransducer = 128, size_t numChannel = 64,
                      size_t numTimeSteps = 2337) {
  for (int i = 0; i < numTransducer; i++) {
    uint32_t transducerIndex = i * numChannel * numTimeSteps;  // i * 64 * 2337;
    for (int j = 0; j < numChannel; j++) {
      uint32_t channelIndex = j * numTimeSteps;  // j * 2337
      for (int k = 0; k < numTimeSteps; k++) {
        shuffle_RF[transducerIndex + k * numChannel + j] =
            RF[transducerIndex + channelIndex + k];
      }
    }
  }
}

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

#define MAX_SAMPLES 4096
#define MAX_RECEIVED_CHANNELS 64
#define MAX_TX_SCANLINES 128
#define MAX_X_ELEMSDT 128
#define MAX_RX_SCANLINES 256
#define MAX_TIMESTEPS 2337

template <bool interpolateRFlines, bool interpolateBetweenTransmits,
          typename RFType, typename ResultType, typename LocationType>
void rxBeamformingDTSPACEKernel(
    size_t numTransducerElements, size_t numReceivedChannels,
    size_t numTimesteps, RFType *__restrict__ RF, size_t numTxScanlines,
    size_t numRxScanlines, ScanlineRxParameters3D *__restrict__ scanlinesDT,
    size_t numDs, LocationType *__restrict__ dsDT,
    LocationType *__restrict__ xElemsDT, LocationType speedOfSound,
    LocationType dt, uint32_t additionalOffset, LocationType F,
    float *windowAcc, const float windowFunctionScale,
    ResultType *__restrict__ s) {
  numReceivedChannels = MAX_RECEIVED_CHANNELS;
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

  // local initial data
  // assume that the data is interleaved across the channels
  for (uint32_t i = 0; i < numTimesteps; i++) {
    int baseAddr = i * numReceivedChannels;
    int rfbaseAddr = i * MAX_RECEIVED_CHANNELS;
#pragma unroll
    for (int j = 0; j < MAX_RECEIVED_CHANNELS; j++) {
      // still load data even the j is bigger than numReceviedChannels
      // even we won't use them at all
      rf1[rfbaseAddr + j] = RF[baseAddr + j];
    }
  }

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
    int baseAddr = nextTxScanlinIdx * numTimesteps * numReceivedChannels +
                   r * numReceivedChannels;

    RFType RF_tmp[MAX_RECEIVED_CHANNELS];
#pragma unroll
    for (int j = 0; j < MAX_RECEIVED_CHANNELS; j++) {
      RF_tmp[j] = RF[baseAddr + j];
    }

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

    if (r >= numDs) continue;
    {
      LocationType d = localDsDT[r];
      // TODO should this also depend on the angle?
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
      uint32_t txScanlineIdx = txParams.txScanlineIdx;
      float sLocal = 0.0f;

      float sample = 0.0f;
      float weightAcum = 0.0f;
      int numAdds = 0;
      LocationType initialDelay = txParams.initialDelay;
      // uint32_t txScanlineIdx = txParams.txScanlineIdx;

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
        int32_t delay1 = (delay + 1) & (MAX_SAMPLES - 1);

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

      s[sampleIndex] = clampCast<ResultType>(sInterp);
      sampleIndex = sampleIndex + 1;
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

  std::cout << "set some solid param \n";

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

  std::cout << "read in params \n";

  // read input data file
  ifstream rf(Inputfilename, ios::in | ios::binary);
  rf.seekg(0, ios::end);
  size_t filelength = rf.tellg();
  rf.seekg(0);
  size_t valid_bytes =
      numReceivedChannels * numSamples * numTxScanlines * sizeof(int16_t);
  uint32_t numFrames = filelength / valid_bytes;

  RFdata = (int16_t *)malloc(filelength);
  rf.read((char *)RFdata, filelength);
  rf.close();

  std::cout << "read in raw data \n";

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
  // origin_rxScanlines = (ScanlineRxParameters3D
  // *)sycl::malloc_device(numRxScanlines * sizeof(ScanlineRxParameters3D), q);
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

  std::cout << "scanlineLayout.x " << scanlineLayout.x << ", scanlineLayout.y "
            << scanlineLayout.y << std::endl;

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

  q.memcpy(m_mask, mask.data(), mask.size() * sizeof(uint8_t)).wait();
  q.memcpy(m_sampleIdx, sampleIdx.data(), sampleIdx.size() * sizeof(uint32_t))
      .wait();
  q.memcpy(m_weightX, weightX.data(), weightX.size() * sizeof(float)).wait();
  q.memcpy(m_weightY, weightY.data(), weightY.size() * sizeof(float)).wait();

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

  s_dev = (float *)sycl::malloc_device(
      rxNumDepths * numRxScanlines * sizeof(float), q);
  s_tmp = (float *)malloc(rxNumDepths * numRxScanlines * sizeof(float));
  s = (float *)malloc(rxNumDepths * numRxScanlines * sizeof(float));

  RFdata_shuffle = (int16_t *)malloc(numReceivedChannels * numSamples * numTxScanlines * sizeof(int16_t));

  RFdata_dev = (int16_t *)sycl::malloc_device(
      numReceivedChannels * numSamples * numTxScanlines * sizeof(int16_t), q);

  q.memcpy(rxDepths_dev, rxDepths, rxNumDepths * sizeof(float)).wait();
  q.memcpy(rxElementXs_dev, rxElementXs, numElements * sizeof(float)).wait();
  q.memcpy(rxElementYs_dev, rxElementYs, numElements * sizeof(float)).wait();
  q.memcpy(window_data_dev, window_data, numEntriesPerFunction * sizeof(float))
      .wait();
  q.memcpy(rxScanlines_dev, rxScanlines,
           numRxScanlines * sizeof(ScanlineRxParameters3D)).wait();

  return 1;
}

void Beamforming2D::SubmitKernel(int16_t *raw_ptr, size_t len) {
  window_scale = 0;
  size_t p_numElements = numElements;
  size_t p_numReceivedChannels = numReceivedChannels;
  size_t p_numSamples = numSamples;
  size_t p_numTxScanlines = numTxScanlines;
  size_t p_numRxScanlines = numRxScanlines;
  struct ScanlineRxParameters3D *p_rxScanlines_dev = rxScanlines_dev;
  size_t p_rxNumDepths = rxNumDepths;

  float *p_rxDepths_dev = rxDepths_dev;
  float *p_rxElementXs_dev = rxElementXs_dev;
  float *p_rxElementYs_dev = rxElementYs_dev;

  float p_speedOfSoundMMperS = speedOfSoundMMperS;
  float p_dt = dt;
  float p_additionalOffset = additionalOffset;
  float p_fNumber = fNumber;
  float *p_window_data_dev = window_data_dev;
  float p_window_scale = window_scale;
  float *p_s_dev = s_dev;
  int16_t *p_RFdata_dev = RFdata_dev;

  q.wait();

  sycl::event e = q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class BeamForming>([=]()[[intel::kernel_args_restrict]] {
      rxBeamformingDTSPACEKernel<true, false, int16_t, float, float>(
          p_numElements, p_numReceivedChannels, p_numSamples, p_RFdata_dev,
          p_numTxScanlines, p_numRxScanlines, p_rxScanlines_dev, p_rxNumDepths,
          p_rxDepths_dev, p_rxElementXs_dev, p_speedOfSoundMMperS, p_dt,
          p_additionalOffset, p_fNumber, p_window_data_dev, p_window_scale,
          p_s_dev);
    });
  });

  e.wait();
  q.wait();

  Report_time(std::string("beamforming kernel: "), e);
}

int Beamforming2D::read_one_frame2dev(int16_t *raw_ptr, size_t len) {
  if (len == 0) {
    std::cout << "Error raw data frame size.\n";
    return 0;
  }
  shuffle_RF_order<int16_t>(RFdata_shuffle, raw_ptr);
  q.memcpy(RFdata_dev, RFdata_shuffle, len * sizeof(int16_t)).wait();
  return 1;
}

sycl::queue Beamforming2D::getStream() { return q; }

float *Beamforming2D::getRes() { return s_dev; }

float *Beamforming2D::getResHost() {
  q.memcpy(s_tmp, s_dev, rxNumDepths * numRxScanlines * sizeof(float)).wait();

  shuffle_image<float>(s, s_tmp);
  q.memcpy(s_dev, s, rxNumDepths * numRxScanlines * sizeof(float)).wait();
  return s; }