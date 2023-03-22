// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include <CL/sycl.hpp>
#include "LogCompressor.h"
#include "utility.hpp"

#include <cmath>

using namespace std;
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

  Out operator()(const In& a) const {
    WorkType val = sycl::log10(std::abs(static_cast<WorkType>(a)) * _inScale +
                               (WorkType)1) *
                   _scaleOverDenominator;
    return clampCast<Out>(val);
  }
};

LogCompressor::LogCompressor(float* input, sycl::queue in_q)
    : q(in_q), input_dev(input) {}

LogCompressor::LogCompressor(sycl::queue in_q)
    : q(in_q) {}

void LogCompressor::getInput(float *input){
  input_dev = input;
}

#define PARA_NUM 8
void LogCompressor::compress(vec3s size, double dynamicRange, double scale,
                             double inMax) {
  const float* inImageData = input_dev;
  size_t width = size.x;
  size_t height = size.y;
  size_t depth = size.z;

  float outMax;
  if (std::is_integral<float>::value) {
    outMax = std::numeric_limits<float>::max();
  } else if (std::is_floating_point<float>::value) {
    outMax = static_cast<float>(255.0);
  }

  thrustLogcompress<float, float, WorkType> c(
      sycl::pow<double>(10, (dynamicRange / 20)), static_cast<float>(inMax),
      outMax, scale);

  output = (float*)malloc(width * height * depth * sizeof(float));
  output_dev =
      (float*)sycl::malloc_device(width * height * depth * sizeof(float), q);

  auto inImageData_t = inImageData;
  auto pComprGpu_t = output_dev;

  static long log_call_count = 0;
  static std::chrono::duration<double, std::milli> log_total_duration(0);

  q.wait();
  sycl::event log_event = q.submit([&](sycl::handler& h) {
    h.single_task<class LogCompress>([=]()[[intel::kernel_args_restrict]] {
      int reg = 2000 * 255;
      for (int i = 0; i < reg + PARA_NUM; i += PARA_NUM) {
#pragma unroll
        for (int j = 0; j < PARA_NUM; j++) {
          if (i + j < reg) {
            pComprGpu_t[i + j] = c(inImageData_t[i + j]);
          }
        }
      }
    });
  });
  q.wait();
  log_event.wait();
  Report_time(std::string("LogCompressor kernel: "), log_event);
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

float* LogCompressor::getRes() { return output_dev; }

float* LogCompressor::getResHost() {
  q.memcpy(output, output_dev, 255 * 2000 * sizeof(float)).wait();
  return output;
}

LogCompressor::~LogCompressor() {
  if (output_dev) free(output_dev, q);
  if (output) free(output);
}