// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#ifndef LOGCOMPRESSOR_H
#define LOGCOMPRESSOR_H

#include <sycl/sycl.hpp>
#include <memory>
#include "utility.hpp"

using namespace sycl;

typedef vec3T<size_t> vec3s;

class LogCompressor {
 public:
  typedef float WorkType;

  void compress(vec3s size, double dynamicRange, double scale, double inMax);
  LogCompressor(float* input, sycl::queue in_q);
  LogCompressor(sycl::queue in_q);
  void getInput(float *input);
  ~LogCompressor();
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
