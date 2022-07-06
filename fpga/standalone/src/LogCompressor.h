// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#ifndef __LOGCOMPRESSOR_H__
#define __LOGCOMPRESSOR_H__

#include <CL/sycl.hpp>
#include <memory>
#include "utility.hpp"

using namespace sycl;

typedef vec3T<size_t> vec3s;

class LogCompressor {
 public:
  typedef float WorkType;

  void compress(vec3s size, double dynamicRange, double scale, double inMax);
  LogCompressor(float *input, sycl::queue in_q);
  LogCompressor(sycl::queue in_q);
  ~LogCompressor();
  void getInput(float *input);
  void SubmitKernel();

  float* getRes();
  float* getResHost();

 private:
  sycl::queue q;

  float* input_dev = NULL;
  float* output_dev = NULL;
  float* output = NULL;
};

#endif  //!__LOGCOMPRESSOR_H__
