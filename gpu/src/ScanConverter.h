// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#ifndef __SCANCONVERTER_H__
#define __SCANCONVERTER_H__

#include <cmath>
#include <iostream>
#include <memory>
#include "utility.hpp"

class ScanConverter {
 public:
  typedef uint32_t IndexType;
  typedef float WeightType;

  static constexpr int m_mappingMaxIterations = 1000;
  static constexpr double m_mappingDistanceThreshold = 1e-5;

  ScanConverter(sycl::queue hq, float* input_addr, uint8_t* mask,
                uint32_t* sampleIdx, float* weightX, float* weightY,
                vec3s imageSize, RawParam *p_Params);
  ScanConverter(sycl::queue hq, uint8_t* mask,
                uint32_t* sampleIdx, float* weightX, float* weightY,
                vec3s imageSize, RawParam *p_Params);
  ~ScanConverter();

  template <typename InputType, typename OutputType>
  void convert2D();
  uint8_t* getMask();
  void updateInternals();
  vec3s getImageSize() const { return m_imageSize; }
  void getInput(float *input);

  void SubmitKernel();

  float* getRes();
  float* getResHost();

  vec2i m_outputSize;
  std::vector<double> comsuming_time;

 private:
  sycl::queue q;

  size_t m_numScanlines;
  size_t m_numSamples;
  size_t m_filterLength;

  float* input_dev = NULL;
  float* output_dev = NULL;
  float* output = NULL;

  ScanlineRxParameters3D* scanlines = NULL;

  static constexpr double m_skewnessTestThreshold = 1e-6;

  bool m_is2D;

  uint8_t* m_mask;
  IndexType* m_sampleIdx;
  WeightType* m_weightX;
  WeightType* m_weightY;
  WeightType* m_weightZ;

  vec3s m_imageSize = {0, 0, 0};

  RawParam *params = NULL;
};

#endif  //!__SCANCONVERTER_H__