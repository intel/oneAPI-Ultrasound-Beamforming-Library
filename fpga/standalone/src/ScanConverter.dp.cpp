// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include <CL/sycl.hpp>
#include <cassert>
#include <cmath>
#include "ScanConverter.h"
#include "utility.hpp"

using namespace std;

class ScanConverterInternals {
 public:
  typedef ScanConverter::IndexType IndexType;
  typedef ScanConverter::WeightType WeightType;

  static constexpr double m_tetrahedronTestDistanceThreshold = 1e-9;
  static constexpr int m_mappingMaxIterations =
      ScanConverter::m_mappingMaxIterations;
  static constexpr double m_mappingDistanceThreshold =
      ScanConverter::m_mappingDistanceThreshold;

  template <typename Tf, typename Ti>
  static void computeParametersVoxel3D(
      const Tf &sampleDistance, const vec2T<Ti> &scanlineLayout,
      const int &scanlineIdxX, const int &scanlineIdxY, const vec3T<Tf> &s1,
      const vec3T<Tf> &e1, const vec3T<Tf> &s2, const vec3T<Tf> &e2,
      const vec3T<Tf> &s3, const vec3T<Tf> &e3, const vec3T<Tf> &s4,
      const vec3T<Tf> &e4, const vec3T<Tf> &scanline1Pos,
      const vec3T<Tf> &scanline1Dir, const vec3T<Tf> &scanline2Pos,
      const vec3T<Tf> &scanline2Dir, const vec3T<Tf> &scanline3Pos,
      const vec3T<Tf> &scanline3Dir, const vec3T<Tf> &scanline4Pos,
      const vec3T<Tf> &scanline4Dir, const Tf &startDepth, const Tf &endDepth,
      const vec3T<Ti> &imageSize, const vec3T<Ti> &voxel,
      const vec3T<Tf> &voxelPos, uint8_t *__restrict__ maskBuf,
      uint32_t *__restrict__ sampleIdxBuf, float *__restrict__ weightXBuf,
      float *__restrict__ weightYBuf, float *__restrict__ weightZBuf) {
    if (pointInsideTetrahedron(s1, s2, s3, e1, voxelPos) ||
        pointInsideTetrahedron(s2, s4, s3, e4, voxelPos) ||
        pointInsideTetrahedron(s2, e1, e2, e4, voxelPos) ||
        pointInsideTetrahedron(s3, e3, e1, e4, voxelPos) ||
        pointInsideTetrahedron(s2, s3, e1, e4, voxelPos)) {
      std::pair<vec3T<Tf>, bool> params = mapToParameters3D<Tf, Ti>(
          scanline1Pos, scanline2Pos, scanline3Pos, scanline4Pos, scanline1Dir,
          scanline2Dir, scanline3Dir, scanline4Dir, startDepth, endDepth,
          voxelPos);

      if (params.second) {
        size_t voxelIndex = voxel.x + voxel.y * imageSize.x +
                            voxel.z * imageSize.x * imageSize.y;
        maskBuf[voxelIndex] = 1;

        Tf t1 = params.first.x;
        Tf t2 = params.first.y;
        Tf d = params.first.z + 0;

        IndexType sampleIdxScanline =
            static_cast<IndexType>(sycl::floor(d / sampleDistance));
        WeightType weightY =
            static_cast<WeightType>(d / sampleDistance - sampleIdxScanline);
        WeightType weightX = static_cast<WeightType>(t1);
        WeightType weightZ = static_cast<WeightType>(t2);

        IndexType sampleIdx = static_cast<IndexType>(
            sampleIdxScanline * scanlineLayout.x * scanlineLayout.y +
            scanlineIdxX + scanlineIdxY * scanlineLayout.x);

        sampleIdxBuf[voxelIndex] = sampleIdx;
        weightXBuf[voxelIndex] = weightX;
        weightYBuf[voxelIndex] = weightY;
        weightZBuf[voxelIndex] = weightZ;
      }
    }
  }

  /**
   * Tests whether point p lies within the tetrahedron defined by a, b, c, d.
   *
   * For the test, the barycentric coordinates of p are computed and checked for
   * equal sign.
   */
  template <typename Tf>
  static bool pointInsideTetrahedron(const vec3T<Tf> &a, const vec3T<Tf> &b,
                                     const vec3T<Tf> &c, const vec3T<Tf> &d,
                                     const vec3T<Tf> &p) {
    Tf w0 = barycentricCoordinate3D(a, b, c, d);

    Tf w1 = barycentricCoordinate3D(p, b, c, d);
    Tf w2 = barycentricCoordinate3D(a, p, c, d);
    Tf w3 = barycentricCoordinate3D(a, b, p, d);
    Tf w4 = barycentricCoordinate3D(a, b, c, p);

    return w0 > 0 && w1 >= -m_tetrahedronTestDistanceThreshold &&
           w2 >= -m_tetrahedronTestDistanceThreshold &&
           w3 >= -m_tetrahedronTestDistanceThreshold &&
           w4 >= -m_tetrahedronTestDistanceThreshold;
  }

  template <typename Tf>
  static Tf barycentricCoordinate3D(const vec3T<Tf> &a, const vec3T<Tf> &b,
                                    const vec3T<Tf> &c, const vec3T<Tf> &p) {
    // computes the determinant of
    //[a_x, a_y, a_z, 1]
    //[b_x, b_y, b_z, 1]
    //[c_x, c_y, c_z, 1]
    //[p_x, p_y, p_z, 1]

    // reducing 12 multiplications per compute
    const Tf axby = a.x * b.y;
    const Tf cypz = c.y * p.z;
    const Tf axbz = a.x * b.z;
    const Tf czpy = c.z * p.y;
    const Tf aybx = a.y * b.x;
    const Tf cxpz = c.x * p.z;
    const Tf aybz = a.y * b.z;
    const Tf czpx = c.z * p.x;
    const Tf azbx = a.z * b.x;
    const Tf cxpy = c.x * p.y;
    const Tf azby = a.z * b.y;
    const Tf cypx = c.y * p.x;

    return (axby - aybx) * (c.z - p.z) + (aybz - azby) * (c.x - p.x) +
           (azbx - axbz) * (c.y - p.y) + (cypz - czpy) * (a.x - b.x) -
           (cxpz - czpx) * (a.y - b.y) + (cxpy - cypx) * (a.z - b.z);
    // reducing 18 multiplications with the updated return statement per compute
  }

  template <typename Tf>
  static vec3T<Tf> pointPlaneConnection(const vec3T<Tf> &a, const vec3T<Tf> &na,
                                        const vec3T<Tf> &x) {
    return dot(na, (x - a)) * na;
  }

  template <typename Tf, typename Ti>
  static std::pair<vec3T<Tf>, bool> mapToParameters3D(
      const vec3T<Tf> &a, const vec3T<Tf> &ax, const vec3T<Tf> &ay,
      const vec3T<Tf> &axy, const vec3T<Tf> &da, const vec3T<Tf> &dax,
      const vec3T<Tf> &day, const vec3T<Tf> &daxy, Tf startDepth, Tf endDepth,
      const vec3T<Tf> &x) {
    vec3T<Tf> normalXLow = normalize(cross(da, (ay + day) - a));
    vec3T<Tf> normalYLow = normalize(cross((ax + dax) - a, da));
    vec3T<Tf> normalXHigh = normalize(cross(dax, (axy + daxy) - ax));
    vec3T<Tf> normalYHigh = normalize(cross((axy + daxy) - ay, day));

    // find t via binary search
    vec2T<Tf> lowT = {0, 0};
    vec2T<Tf> highT = {1, 1};
    vec3T<Tf> lowConnX = pointPlaneConnection(a, normalXLow, x);
    vec3T<Tf> highConnX = pointPlaneConnection(ax, normalXHigh, x);
    vec3T<Tf> lowConnY = pointPlaneConnection(a, normalYLow, x);
    vec3T<Tf> highConnY = pointPlaneConnection(ay, normalYHigh, x);
    vec2T<Tf> lowDist = {norm(lowConnX), norm(lowConnY)};
    vec2T<Tf> highDist = {norm(highConnX), norm(highConnY)};

    if (dot(lowConnX, highConnX) > 0 || dot(lowConnY, highConnY) > 0) {
      return std::pair<vec3T<Tf>, bool>(vec3T<Tf>{0, 0, 0}, false);
    }

    vec2T<Tf> dist = {1e10, 1e10};
    vec2T<Tf> t = (highT - lowT) / 2 + lowT;
    vec3T<Tf> planeBaseX1;
    vec3T<Tf> planeBaseY1;
    vec3T<Tf> planeBaseX2;
    vec3T<Tf> planeBaseY2;
    for (int numIter = 0; numIter < m_mappingMaxIterations &&
                          (dist.x > m_mappingDistanceThreshold ||
                           dist.y > m_mappingDistanceThreshold);
         numIter++) {
      t = (1 - highDist / (highDist + lowDist)) * highT +
          (1 - lowDist / (highDist + lowDist)) * lowT;

      planeBaseX1 = (1 - t.x) * a + t.x * ax;
      planeBaseX2 = (1 - t.x) * ay + t.x * axy;
      planeBaseY1 = (1 - t.y) * a + t.y * ay;
      planeBaseY2 = (1 - t.y) * ax + t.y * axy;
      vec3T<Tf> dir = slerp3(slerp3(da, dax, t.x), slerp3(day, daxy, t.x), t.y);
      vec3T<Tf> normal_x = normalize(cross(dir, planeBaseX2 - planeBaseX1));
      vec3T<Tf> normal_y = normalize(cross(planeBaseY2 - planeBaseY1, dir));

      vec3T<Tf> connX = pointPlaneConnection(planeBaseX1, normal_x, x);
      vec3T<Tf> connY = pointPlaneConnection(planeBaseY1, normal_y, x);

      dist.x = norm(connX);
      dist.y = norm(connY);

      if (dot(highConnX, connX) > M_EPS) {
        highT.x = t.x;
        highConnX = connX;
        highDist.x = dist.x;
      } else if (dot(lowConnX, connX) > M_EPS) {
        lowT.x = t.x;
        lowConnX = connX;
        lowDist.x = dist.x;
      }

      if (dot(highConnY, connY) > M_EPS) {
        highT.y = t.y;
        highConnY = connY;
        highDist.y = dist.y;
      } else if (dot(lowConnY, connY) > M_EPS) {
        lowT.y = t.y;
        lowConnY = connY;
        lowDist.y = dist.y;
      }
    }

    vec3T<Tf> lineBase = (1 - t.y) * planeBaseX1 + t.y * planeBaseX2;
    Tf d = norm(x - lineBase);

    return std::pair<vec3T<Tf>, bool>(vec3T<Tf>{t.x, t.y, d}, true);
  }
};

template <typename Tf, typename Ti>
void
    
    computeParameterBB3D(
        const Tf sampleDistance,
        const vec2T<Ti> scanlineLayout,
        const int scanlineIdxX,
        const int scanlineIdxY,
        const vec3T<Tf> s1,
        const vec3T<Tf> e1,
        const vec3T<Tf> s2,
        const vec3T<Tf> e2,
        const vec3T<Tf> s3,
        const vec3T<Tf> e3,
        const vec3T<Tf> s4,
        const vec3T<Tf> e4,
        const vec3T<Tf> scanline1Pos,
        const vec3T<Tf> scanline1Dir,
        const vec3T<Tf> scanline2Pos,
        const vec3T<Tf> scanline2Dir,
        const vec3T<Tf> scanline3Pos,
        const vec3T<Tf> scanline3Dir,
        const vec3T<Tf> scanline4Pos,
        const vec3T<Tf> scanline4Dir,
        const Tf startDepth,
        const Tf endDepth,
        const vec3T<Ti> imageSize,
        const vec3T<Tf> bbMin,
        const vec3T<Ti> tetMinVoxel,
        const vec3T<Ti> tetMaxVoxel,
        const Tf resolution,
        uint8_t* __restrict__ maskBuf,
        uint32_t* __restrict__ sampleIdxBuf,
        float* __restrict__ weightXBuf,
        float* __restrict__ weightYBuf,
        float* __restrict__ weightZBuf
    ,
        sycl::nd_item<3> item_ct1)
{
  vec3T<Ti> voxel = vec3T<Ti>{
      static_cast<Ti>(item_ct1.get_local_range().get(2) *
                          item_ct1.get_group(2) +
                      item_ct1.get_local_id(2)),
      static_cast<Ti>(item_ct1.get_local_range().get(1) *
                          item_ct1.get_group(1) +
                      item_ct1.get_local_id(1)),
      static_cast<Ti>(
          item_ct1.get_local_range().get(0) * item_ct1.get_group(0) +
          item_ct1.get_local_id(0))};  //@suppress("Symbol is not resolved")
                                       //@suppress("Field cannot be resolved")
  voxel = voxel + tetMinVoxel;

  if (voxel.x <= tetMaxVoxel.x && voxel.y <= tetMaxVoxel.y &&
      voxel.z <= tetMaxVoxel.z) {
    vec3T<Tf> voxelPos = static_cast<vec3T<Tf>>(voxel) * resolution + bbMin;
    ScanConverterInternals::computeParametersVoxel3D(
        sampleDistance, scanlineLayout, scanlineIdxX, scanlineIdxY, s1, e1, s2,
        e2, s3, e3, s4, e4, scanline1Pos, scanline1Dir, scanline2Pos,
        scanline2Dir, scanline3Pos, scanline3Dir, scanline4Pos, scanline4Dir,
        startDepth, endDepth, imageSize, voxel, voxelPos, maskBuf, sampleIdxBuf,
        weightXBuf, weightYBuf, weightZBuf);
  }
}

#define BUFFER_SIZE 256
#define STRIDE_SIZE 4
const uint32_t INPUT_SIZE = 2000 * 255;
const uint32_t OUTPUT_SIZE = 2000 * 1667;
template <typename InputType, typename OutputType, typename WeightType,
          typename IndexType>
void scanConvert2D(uint32_t numScanlines, uint32_t numSamples, uint32_t width,
                   uint32_t height, const uint8_t *__restrict__ mask,
                   const IndexType *__restrict__ sampleIdx,
                   const WeightType *__restrict__ weightX,
                   const WeightType *__restrict__ weightY,
                   const InputType *__restrict__ scanlines,
                   OutputType *__restrict__ image) {
  uint32_t sIdx;
  float wX;
  float wY;
  float val;
  int32_t temp;
  uint32_t buffer_index;
  uint32_t buffer_1;
  uint32_t buffer_2;
  uint32_t buffer_3;
  int Index_e = 0;

  [[ intel::max_replicates(4), intel::doublepump ]] float buf1[BUFFER_SIZE];
  [[ intel::max_replicates(4), intel::doublepump ]] float buf2[BUFFER_SIZE];
  [[ intel::max_replicates(4), intel::doublepump ]] float buf3[BUFFER_SIZE];

  int buf_index = -1;
  int Index = 0;

  for (uint32_t i = 0; i < BUFFER_SIZE; i++) {
    buf1[i] = scanlines[i];
    buf2[i] = scanlines[i + numScanlines];
  }

  int kl = 0;
  int kh = -1;

  [[intel::ivdep]] for (uint32_t i = 0; i < (OUTPUT_SIZE / STRIDE_SIZE); i++) {
    if (kl == (width / STRIDE_SIZE + 2) * STRIDE_SIZE) kl = 0;
    if (kl == 0) {
      buf_index = buf_index + 1;
      if (buf_index == 3) {
        buf_index = 0;
      }
      kh = kh + 1;
    }

    int scanlinesPos = kl / STRIDE_SIZE;

#pragma unroll
    for (uint32_t j = 0; j < STRIDE_SIZE; j++) {
      uint32_t large_index = kh * width + kl + j;

      sIdx = sampleIdx[large_index];
      wX = weightX[large_index];
      wY = weightY[large_index];
      uint8_t kernel_mask = mask[large_index];

      buffer_1 = sIdx / BUFFER_SIZE * BUFFER_SIZE;
      buffer_2 = sIdx / numScanlines * numScanlines;
      buffer_3 = sIdx % BUFFER_SIZE;

      buffer_index = buffer_1 - buffer_2 + sIdx % BUFFER_SIZE;
      temp = kl + j;
      float buf1_val_one = buf1[buffer_index];
      float buf1_val_two = buf1[buffer_index + 1];

      float buf2_val_one = buf2[buffer_index];
      float buf2_val_two = buf2[buffer_index + 1];

      float buf3_val_one = buf3[buffer_index];
      float buf3_val_two = buf3[buffer_index + 1];

      if (buf_index == 0) {
        val = ((1 - wY) * ((1 - wX) * buf1_val_one + wX * buf1_val_two) +
               wY * ((1 - wX) * buf2_val_one + wX * buf2_val_two));
      } else if (buf_index == 1) {
        val = ((1 - wY) * ((1 - wX) * buf2_val_one + wX * buf2_val_one) +
               wY * ((1 - wX) * buf3_val_one + wX * buf3_val_two));
      } else {
        val = ((1 - wY) * ((1 - wX) * buf3_val_one + wX * buf3_val_two) +
               wY * ((1 - wX) * buf1_val_one + wX * buf1_val_two));
      }

      image[large_index] = val * kernel_mask;
    }

    switch (buf_index) {
      case 0:
        if (scanlinesPos < BUFFER_SIZE)
          buf3[scanlinesPos] = scanlines[kh * numScanlines + scanlinesPos];
        break;
      case 1:
        if (scanlinesPos < BUFFER_SIZE)
          buf1[scanlinesPos] = scanlines[kh * numScanlines + scanlinesPos];
        break;
      default:
        if (scanlinesPos < BUFFER_SIZE)
          buf2[scanlinesPos] = scanlines[kh * numScanlines + scanlinesPos];
        break;
    }
    kl = kl + STRIDE_SIZE;
  }
}

ScanConverter::ScanConverter(sycl::queue hq, float *input_addr, uint8_t *mask,
                             uint32_t *sampleIdx, float *weightX,
                             float *weightY, vec3s imageSize)
    : q(hq),
      input_dev(input_addr),
      m_mask(mask),
      m_sampleIdx(sampleIdx),
      m_weightX(weightX),
      m_weightY(weightY),
      m_imageSize(imageSize) {}

ScanConverter::ScanConverter(sycl::queue hq, uint8_t *mask,
                             uint32_t *sampleIdx, float *weightX,
                             float *weightY, vec3s imageSize)
    : q(hq),
      m_mask(mask),
      m_sampleIdx(sampleIdx),
      m_weightX(weightX),
      m_weightY(weightY),
      m_imageSize(imageSize) {}

void ScanConverter::getInput(float* input_addr){
  input_dev = input_addr;
}

uint8_t *ScanConverter::getMask() { return m_mask; }

void ScanConverter::SubmitKernel() { convert2D<float, float>(); }

template <typename InputType, typename OutputType>
void ScanConverter::convert2D() {
  float *inImage = input_dev;
  uint32_t numScanlines = 255;
  vec2s scanlineLayout = {255, 1};
  uint32_t numSamples = 2000;
  m_imageSize = {1667, 2000, 1};

  InputType *pScanlineData = inImage;

  output_dev = (OutputType *)sycl::malloc_device(
      m_imageSize.x * m_imageSize.y * m_imageSize.z * sizeof(OutputType), q);
  output = (OutputType *)malloc(m_imageSize.x * m_imageSize.y * m_imageSize.z *
                                sizeof(OutputType));

  auto pConv = output_dev;

  static long scan_call_count = 0;

  q.wait();

  sycl::event scan_event = q.submit([&](sycl::handler &cgh) {
    auto m_imageSize_x_ct2 = (uint32_t)m_imageSize.x;
    auto m_imageSize_y_ct3 = (uint32_t)m_imageSize.y;
    auto m_mask_get_ct4 = m_mask;
    auto m_sampleIdx_get_ct5 = m_sampleIdx;
    auto m_weightX_get_ct6 = m_weightX;
    auto m_weightY_get_ct7 = m_weightY;
    auto pScanlineData_get_ct8 = pScanlineData;
    auto pConv_get_ct9 = pConv;

    cgh.single_task<class ScanConverter>([=]()[[intel::kernel_args_restrict]] {
      scanConvert2D<float, float, float, uint32_t>(
          numScanlines, numSamples, m_imageSize_x_ct2, m_imageSize_y_ct3,
          m_mask_get_ct4, m_sampleIdx_get_ct5, m_weightX_get_ct6,
          m_weightY_get_ct7, pScanlineData_get_ct8, pConv_get_ct9);
    });
  });

  q.wait();
  scan_event.wait();
  Report_time(std::string("ScanConvertor kernel: "), scan_event);
  q.memcpy(output, output_dev,
           m_imageSize.x * m_imageSize.y * m_imageSize.z * sizeof(OutputType))
      .wait();
}

float *ScanConverter::getRes() { return output_dev; }
float *ScanConverter::getResHost() { return output; }

ScanConverter::~ScanConverter() {
  if (output_dev) free(output_dev, q);
  if (output) free(output);
}