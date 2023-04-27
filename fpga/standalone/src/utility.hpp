// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#ifndef _UTILITY_
#define _UTILITY_

#include <cmath>
#include <iomanip>
#include <limits>
#include <string>
#include <sycl/sycl.hpp>
#include "vec.h"

using namespace std;
using namespace cl::sycl;

#define __FLT_MAX__ 3.40282347e+38F
#define FLT_MAX __FLT_MAX__

static void Report_time(const std::string& msg, sycl::event& e) {
  auto time_start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();

  auto time_end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();

  double elapsed = (time_end - time_start) / 1e6;
  std::cout << msg << elapsed << " milliseconds\n";
}

template <typename T>
constexpr inline T squ(const T& x) {
  return x * x;
}

static void malloc_mem_log(std::string s) {
  std::cout << "Malloc memory in " << s << "fail.\n";
  exit(-1);
}

enum BeamformingType : uint32_t {
  DelayAndSum = 0,
  DelayAndStddev = 1,
  TestSignal = 2
};

enum WindowType : uint32_t {
  WindowRectangular = 0,
  WindowHann = 1,
  WindowHamming = 2,
  WindowGauss = 3,
  WindowINVALID = 4
};

template <typename T>
class LimitProxy {
 public:
  inline static T max();
  inline static T min();
};

template <>
class LimitProxy<float> {
 public:
  inline static float max() { return FLT_MAX; }
  inline static float min() { return -FLT_MAX; }
};

template <>
class LimitProxy<int16_t> {
 public:
  inline static int16_t max() { return 32767; }
  inline static int16_t min() { return -32767; }
};

template <>
class LimitProxy<uint8_t> {
 public:
  inline static uint8_t max() { return 255; }
  inline static uint8_t min() { return 0; }
};

template <typename ResultType, typename InputType>
ResultType clampCast(const InputType& x) {
  return static_cast<ResultType>(sycl::min(
      sycl::max(x, static_cast<InputType>(LimitProxy<ResultType>::min())),
      static_cast<InputType>(LimitProxy<ResultType>::max())));
}

template <typename ResultType, typename InputType>
struct clampCaster {
  ResultType operator()(const InputType& a) const {
    return clampCast<ResultType>(a);
  }
};

class ScanlineRxParameters3D {
 public:
  ScanlineRxParameters3D()
      : txParameters{{{0, 0}, {0, 0}, 0, 0}},
        position{0.0, 0.0, 0.0},
        direction{0.0, 0.0, 0.0},
        maxElementDistance{0.0, 0.0} {}

  struct TransmitParameters {
    vec2T<uint16_t> firstActiveElementIndex;  // index of the first active
                                              // transducer element
    vec2T<uint16_t>
        lastActiveElementIndex;  // index of the last active transducer element
    uint16_t txScanlineIdx;      // index of the corresponsing transmit scanline
    double
        initialDelay;  // the minmal delay in [s] that is to be used during rx
  };

  vec3 position;        // the position of the scanline
  vec3 direction;       // direction of the scanline
  double txWeights[4];  // Weights for interpolation between different transmits
  TransmitParameters txParameters[4];  // Parameters of the transmits to use
  vec2 maxElementDistance;

  vec3 getPoint(double depth) const { return position + depth * direction; }

  friend std::ostream& operator<<(std::ostream& os,
                                  const ScanlineRxParameters3D& params);
  friend std::istream& operator>>(std::istream& is,
                                  ScanlineRxParameters3D& params);
};

#endif