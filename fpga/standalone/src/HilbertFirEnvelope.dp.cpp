// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "HilbertFirEnvelope.h"

#define ROW 2000
#define COL 255
#define HALF_LENGTH 32
#define NUMSCANLINES 255
#define FILTER_LENGTH 65
const int H_VEC_SIZE = 4;

template <typename InputType, typename OutputType>
void vec_kernelFilterDemodulation(
    const InputType* __restrict__ signal,
    const HilbertFirEnvelope::WorkType* __restrict__ filter,
    OutputType* __restrict__ out, const int numSamples, const int numScanlines,
    const int filterLength) {
  [[intel::numbanks(FILTER_LENGTH -
                    1)]] float last_window[NUMSCANLINES][FILTER_LENGTH - 1];
  float last_window_register;
  float sample_list;
  float filterElement;
  float accumulator;
  float signalValue;
  float local_filter[FILTER_LENGTH];

  for (int i = 0; i < FILTER_LENGTH; i++) {
    local_filter[i] = filter[i];
  }

  //[[intelfpga::disable_loop_pipelining]]
  for (int j = 0; j < HALF_LENGTH; j++) {
    for (int i = 0; i < NUMSCANLINES; i++) {
      last_window[i][HALF_LENGTH + j] = signal[j * NUMSCANLINES + i];
    }
  }

  //[[intelfpga::disable_loop_pipelining]]
  for (int kh = 0; kh < ROW; kh++) {
    for (int kl = 0; kl < COL; kl++) {
      accumulator = 0;
      const int filterLength = FILTER_LENGTH;

#pragma unroll
      for (int j = 0; j < FILTER_LENGTH - 1; j++) {
        sample_list = last_window[kl][j];
        filterElement = local_filter[j];
        accumulator += sample_list * filterElement;
      }

#pragma unroll
      for (int j = 0; j < FILTER_LENGTH - 2; j++) {
        last_window[kl][j] = last_window[kl][j + 1];
      }

      if ((kh + FILTER_LENGTH / 2) < ROW) {
        last_window_register =
            signal[(FILTER_LENGTH / 2 + kh) * numScanlines + kl];
      } else {
        last_window_register = 0;
      }

      last_window[kl][FILTER_LENGTH - 2] = last_window_register;

      accumulator += last_window_register * local_filter[FILTER_LENGTH - 1];

      signalValue = last_window[kl][FILTER_LENGTH / 2];
      out[kh * numScanlines + kl] =
          sycl::sqrt(signalValue * signalValue + accumulator * accumulator);
    }
  }
}

HilbertFirEnvelope::HilbertFirEnvelope(sycl::queue hq, float* input_addr)
    : q(hq), input_dev(input_addr) {
  m_numScanlines = 255;
  m_numSamples = 2000;
  m_filterLength = 65;
  output = (float*)malloc(m_numScanlines * m_numSamples * sizeof(float));
  output_dev = (float*)sycl::malloc_device(
      m_numScanlines * m_numSamples * sizeof(float), q);
  prepareFilter();
}

HilbertFirEnvelope::HilbertFirEnvelope(sycl::queue hq)
    : q(hq) {
  m_numScanlines = 255;
  m_numSamples = 2000;
  m_filterLength = 65;
  output = (float*)malloc(m_numScanlines * m_numSamples * sizeof(float));
  output_dev = (float*)sycl::malloc_device(
      m_numScanlines * m_numSamples * sizeof(float), q);
  m_hilbertFilter_dev =
      (float*)sycl::malloc_device(m_filterLength * sizeof(float), q);
  prepareFilter();
}

void HilbertFirEnvelope::getInput(float* input_addr){
  input_dev = input_addr;
}

void HilbertFirEnvelope::prepareFilter() {
  m_hilbertFilter = FirFilterFactory::createFilter<float>(
      m_filterLength, FirFilterFactory::FilterTypeHilbertTransformer,
      FirFilterFactory::FilterWindowHamming);

  q.memcpy(m_hilbertFilter_dev, m_hilbertFilter, m_filterLength * sizeof(float))
      .wait();
}

void HilbertFirEnvelope::SubmitKernel() {
  q.wait();

  sycl::event e = q.submit([&](sycl::handler& cgh) {
    auto inImageData_get_ct0 = input_dev;
    auto m_hilbertFilter_get_ct1 = m_hilbertFilter_dev;
    auto pEnv_get_ct2 = output_dev;
    auto m_filterLength_ct5 = (int)m_filterLength;
    auto numSamples = m_numSamples;
    auto numScanlines = m_numScanlines;

    cgh.single_task<class HilbertEnvelope>([=
    ]()[[intel::kernel_args_restrict]] {
      vec_kernelFilterDemodulation<float, float>(
          inImageData_get_ct0, m_hilbertFilter_get_ct1, pEnv_get_ct2,
          numSamples, numScanlines, m_filterLength_ct5);
    });
  });

  q.wait();
  e.wait();

  Report_time(std::string("HilbertFirEnvelope kernel: "), e);
}

float* HilbertFirEnvelope::getRes() { return output_dev; }

float* HilbertFirEnvelope::getResHost() {
  q.memcpy(output, output_dev, m_numSamples * m_numScanlines * sizeof(float))
      .wait();
  return output;
}

HilbertFirEnvelope::~HilbertFirEnvelope() {
  if(m_hilbertFilter) free(m_hilbertFilter);
  if(m_hilbertFilter_dev) sycl::free(m_hilbertFilter_dev, q);
  if(output_dev) sycl::free(output_dev, q);
  if(output) free(output);
}