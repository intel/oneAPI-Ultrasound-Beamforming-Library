// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#ifndef _UTILITY_
#define _UTILITY_

#include <cmath>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>
#include "CL/sycl.hpp"
#include "vec.h"

using namespace std;
using namespace cl::sycl;

#define __FLT_MAX__ 3.40282347e+38F
#define FLT_MAX __FLT_MAX__

static double AvgVec(std::vector<double> &vec) {
  double res = 0;
  for (int i = 0; i < vec.size(); i++)
    res += vec[i];
  return res/vec.size();
} 

static double Report_time(const std::string& msg, sycl::event e) {
  cl::sycl::cl_ulong time_start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();

  cl::sycl::cl_ulong time_end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();

  double elapsed = (time_end - time_start) / 1e6;
  std::cout << msg << elapsed << " milliseconds\n";
  return elapsed;
}

template <typename T>
constexpr inline T squ(const T& x) {
  return x * x;
}

static void malloc_mem_log(std::string s) {
  std::cout << "Malloc memory in " << s << " fail.\n";
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
  return static_cast<ResultType>(std::min(
      std::max(x, static_cast<InputType>(LimitProxy<ResultType>::min())),
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

class RawParam {
 public:
  size_t numElements;
  size_t numReceivedChannels;
  size_t numSamples;
  size_t numTxScanlines;
  size_t rxNumDepths;
  vec2s scanlineLayout;
  vec2s elementLayout;
  double depth;
  double samplingFrequency;
  double speedOfSoundMMperS;

  RawParam(size_t p_numElements,
  size_t p_numReceivedChannels,
  size_t p_numSamples,
  size_t p_numTxScanlines,
  size_t p_rxNumDepths,
  vec2s p_scanlineLayout,
  vec2s p_elementLayout,
  double p_depth,
  double p_samplingFrequency,
  double p_speedOfSoundMMperS) {
    numElements = p_numElements;
    numReceivedChannels = p_numReceivedChannels;
    numSamples = p_numSamples;
    numTxScanlines = p_numTxScanlines;
    rxNumDepths = p_rxNumDepths;
    depth = p_depth;
    samplingFrequency = p_samplingFrequency;
    speedOfSoundMMperS = p_speedOfSoundMMperS;
    scanlineLayout.x = p_scanlineLayout.x;
    scanlineLayout.y = p_scanlineLayout.y;
    elementLayout.x = p_elementLayout.x;
    elementLayout.y = p_elementLayout.y;
  }
};

#endif