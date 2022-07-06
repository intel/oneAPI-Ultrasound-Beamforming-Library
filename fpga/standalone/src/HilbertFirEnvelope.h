// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#ifndef HILBERTFIRENVELOPE_H
#define HILBERTFIRENVELOPE_H
#pragma once

// #include "ultrasound.h"

#include <algorithm>
#include <cmath>
#include <dpct/dpct.hpp>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <CL/sycl.hpp>
#include "utility.hpp"

/// A factory for FIR filters
class FirFilterFactory {
 public:
  /// Enum for the different filter types
  enum FilterType {
    FilterTypeLowPass,
    FilterTypeHighPass,
    FilterTypeBandPass,
    FilterTypeHilbertTransformer
  };

  /// Enum for the different window types used in creating filters
  enum FilterWindow {
    FilterWindowRectangular,
    FilterWindowHann,
    FilterWindowHamming,
    FilterWindowKaiser
  };

  /// Returns a FIR filter constructed with the window-method
  template <typename ElementType>
  static ElementType *createFilter(const size_t &length, const FilterType &type,
                                   const FilterWindow &window,
                                   const double &samplingFrequency = 2.0,
                                   const double &frequency = 0.0,
                                   const double &bandwidth = 0.0) {
    ElementType *filter = createFilterNoWindow<ElementType>(
        length, type, samplingFrequency, frequency, bandwidth);
    applyWindowToFilter<ElementType>(length, filter, window);
    if (type == FilterTypeBandPass) {
      normalizeGain<ElementType>(length, filter, samplingFrequency, frequency);
    }

    return filter;
  }

 private:
  template <typename ElementType>
  static ElementType *createFilterNoWindow(const size_t &length,
                                           const FilterType &type,
                                           const double &samplingFrequency,
                                           const double &frequency,
                                           const double &bandwidth) {
    if (type == FilterTypeHighPass || type == FilterTypeBandPass ||
        type == FilterTypeLowPass) {
      assert(samplingFrequency != 0.0);
      assert(frequency != 0.0);
    }
    if (type == FilterTypeBandPass) {
      assert(bandwidth != 0.0);
    }

    ElementType omega =
        static_cast<ElementType>(2 * M_PI * frequency / samplingFrequency);
    ElementType omegaBandwidth =
        static_cast<ElementType>(2 * M_PI * bandwidth / samplingFrequency);
    int halfWidth = ((int)length - 1) / 2;

    auto filter = (ElementType *)malloc(length * sizeof(ElementType));
    // determine the filter function
    std::function<ElementType(int)> filterFunction =
        [&halfWidth](int n) -> ElementType {
      if (n == halfWidth) {
        return static_cast<ElementType>(1);
      } else {
        return static_cast<ElementType>(0);
      }
    };
    switch (type) {
      case FilterTypeHilbertTransformer:
        // Following formula 2 in
        // "Carrick, Matt, and Doug Jaeger. "Design and Application of a Hilbert
        // Transformer in a Digital Receiver." (2011)."
        filterFunction = [halfWidth](int n) -> ElementType {
          auto k = (n - halfWidth);
          if (k % 2 == 0) {
            return static_cast<ElementType>(0);
          } else {
            return static_cast<ElementType>(2.0 / (M_PI * k));
          }
        };
        break;
      case FilterTypeHighPass:
        filterFunction = [omega, halfWidth](int n) -> ElementType {
          if (n == halfWidth) {
            return static_cast<ElementType>(1 - omega / M_PI);
          } else {
            return static_cast<ElementType>(-omega / M_PI *
                                            sycl::sin(omega * (n - halfWidth)) /
                                            (omega * (n - halfWidth)));
          }
        };
        break;
      case FilterTypeBandPass:
        filterFunction = [omega, omegaBandwidth,
                          halfWidth](int n) -> ElementType {
          if (n == halfWidth) {
            return static_cast<ElementType>(2.0 * omegaBandwidth / M_PI);
          } else {
            return static_cast<ElementType>(
                2.0 * cos(omega * n - halfWidth) * omegaBandwidth / M_PI *
                sin(omegaBandwidth * (n - halfWidth)) /
                (omegaBandwidth * (n - halfWidth)));
          }
        };
        break;
      case FilterTypeLowPass:
      default:
        filterFunction = [omega, halfWidth](int n) -> ElementType {
          if (n == halfWidth) {
            return static_cast<ElementType>(omega / M_PI);
          } else {
            return static_cast<ElementType>(omega / M_PI *
                                            sin(omega * (n - halfWidth)) /
                                            (omega * (n - halfWidth)));
          }
        };
        break;
    }

    // create the filter
    for (size_t k = 0; k < length; k++) {
      filter[k] = filterFunction((int)k);
    }

    return filter;
  }

  template <typename ElementType>
  static void applyWindowToFilter(const size_t &length, ElementType *filter,
                                  FilterWindow window) {
    size_t filterLength = length;
    size_t maxN = filterLength - 1;
    ElementType beta = (ElementType)4.0;
    std::function<ElementType(int)> windowFunction =
        [filterLength](int n) -> ElementType {
      return static_cast<ElementType>(1);
    };
    switch (window) {
      case FilterWindowHann:
        windowFunction = [maxN](int n) -> ElementType {
          return static_cast<ElementType>(0.50 -
                                          0.50 * cos(2 * M_PI * n / maxN));
        };
        break;
      case FilterWindowHamming:
        windowFunction = [maxN](int n) -> ElementType {
          return static_cast<ElementType>(0.54 -
                                          0.46 * cos(2 * M_PI * n / maxN));
        };
        break;
      case FilterWindowKaiser:
        windowFunction = [maxN, beta](int n) -> ElementType {
          double argument =
              beta *
              sycl::sqrt(1.0 - (2 * ((ElementType)n - maxN / 2) / maxN) *
                                   (2 * ((ElementType)n - maxN / 2) / maxN));
          return static_cast<ElementType>(bessel0_1stKind(argument) /
                                          bessel0_1stKind(beta));
        };
        break;
      case FilterWindowRectangular:
      default:
        windowFunction = [](int n) -> ElementType {
          return static_cast<ElementType>(1);
        };
        break;
    }

    for (size_t k = 0; k < filterLength; k++) {
      filter[k] *= windowFunction((int)k);
    }
  }

  template <typename ElementType>
  static void normalizeGain(const size_t &length, ElementType *filter,
                            double samplingFrequency, double frequency) {
    ElementType omega =
        static_cast<ElementType>(2 * M_PI * frequency / samplingFrequency);
    ElementType gainR = 0;
    ElementType gainI = 0;

    for (int k = 0; k < length; k++) {
      gainR += filter[k] * cos(omega * (ElementType)k);
      gainI += filter[k] * sin(omega * (ElementType)k);
    }
    ElementType gain = sycl::sqrt(gainR * gainR + gainI * gainI);
    for (int k = 0; k < length; k++) {
      filter[k] /= gain;
    }
  }

  template <typename T>
  static T bessel0_1stKind(const T &x) {
    T sum = 0.0;
    // implemented look up factorial.
    static const int factorial[9] = {1,   2,    6,     24,    120,
                                     720, 5040, 40320, 362880};
    for (int k = 1; k < 10; k++) {
      T xPower = pow(x / (T)2.0, (T)k);
      // 1, 2, 6, 24, 120, 720, 5040, 40320, 362880
      sum += pow(xPower / (T)factorial[k - 1], (T)2.0);
    }
    return (T)1.0 + sum;
  }
};

class HilbertFirEnvelope {
 public:
  typedef float WorkType;
  HilbertFirEnvelope(sycl::queue hq, float *input_addr);
  HilbertFirEnvelope(sycl::queue hq);

  ~HilbertFirEnvelope();

  void getInput(float *input_addr);

  void prepareFilter();
  void SubmitKernel();

  float *getRes();
  float *getResHost();

 private:
  sycl::queue q;

  size_t m_numScanlines;
  size_t m_numSamples;
  size_t m_filterLength;

  WorkType *m_hilbertFilter = NULL;
  WorkType *m_hilbertFilter_dev = NULL;

  WorkType *input_dev = NULL;
  WorkType *output_dev = NULL;
  WorkType *output = NULL;
};

#endif