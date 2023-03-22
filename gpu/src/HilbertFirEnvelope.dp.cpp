// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "HilbertFirEnvelope.h"

const int H_VEC_SIZE = 4;
template <typename InputType, typename OutputType>
void kernelFilterDemodulation(
    const InputType* __restrict__ signal,
    const HilbertFirEnvelope::WorkType* __restrict__ filter,
    OutputType* __restrict__ out, const int numSamples, const int numScanlines,
    const int filterLength, sycl::nd_item<3> item_ct1) {
  int scanlineIdx = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
  int sampleIdx = item_ct1.get_local_range().get(1) * item_ct1.get_group(1) +
                  item_ct1.get_local_id(1);

  scanlineIdx *= H_VEC_SIZE;
  if (scanlineIdx < numScanlines && sampleIdx < numSamples) {
    sycl::vec<HilbertFirEnvelope::WorkType, H_VEC_SIZE> accumulator(0.0);

    int startPoint = sampleIdx - filterLength / 2;
    int endPoint = sampleIdx + filterLength / 2;
    int currentFilterElement = 0;

    for (int currentSample = startPoint; currentSample <= endPoint;
         currentSample++, currentFilterElement++) {
      if (currentSample >= 0 && currentSample < numSamples) {
        sycl::vec<HilbertFirEnvelope::WorkType, H_VEC_SIZE> vec_sample(0.0);
#pragma unroll
        for (int c = 0; c < H_VEC_SIZE; c++) {
          vec_sample[c] =
              static_cast<HilbertFirEnvelope::WorkType>(
                  signal[scanlineIdx + c + currentSample * numScanlines]) *
              filter[currentFilterElement];
        }
        accumulator += vec_sample;
      }
    }
#pragma unroll
    for (int c = 0; c < H_VEC_SIZE; c++) {
      HilbertFirEnvelope::WorkType signalValue =
          static_cast<HilbertFirEnvelope::WorkType>(
              signal[scanlineIdx + c + sampleIdx * numScanlines]);
      out[scanlineIdx + c + sampleIdx * numScanlines] =
          sycl::sqrt(squ(signalValue) + squ(accumulator[c]));
    }
  }
}

HilbertFirEnvelope::HilbertFirEnvelope(sycl::queue& in_q) {
  q = in_q;

  m_numScanlines = 255;
  m_numSamples = 2000;
  m_filterLength = 65;
  input_dev = (float*)sycl::malloc_device(
      m_numScanlines * m_numSamples * sizeof(float), q);
  output = (float*)sycl::malloc_host(
      m_numScanlines * m_numSamples * sizeof(float), q);
  output_dev = (float*)sycl::malloc_device(
      m_numScanlines * m_numSamples * sizeof(float), q);
  prepareFilter();
}

void HilbertFirEnvelope::prepareFilter() {
  m_hilbertFilter = FirFilterFactory::createFilter<float>(
      q, m_filterLength, FirFilterFactory::FilterTypeHilbertTransformer,
      FirFilterFactory::FilterWindowHamming);

  m_hilbertFilter_dev =
      (float*)sycl::malloc_device(m_filterLength * sizeof(float), q);

  q.memcpy(m_hilbertFilter_dev, m_hilbertFilter, m_filterLength * sizeof(float))
      .wait();
}

void HilbertFirEnvelope::copydata(float* host_addr, size_t len) {
  q.memcpy(input_dev, host_addr, len * sizeof(float)).wait();
}

void HilbertFirEnvelope::getInput(float* input_addr) { input_dev = input_addr; }

void HilbertFirEnvelope::SubmitKernel() {
  sycl::range<3> blockSizeFilter(1, 8, 16);
  sycl::range<3> gridSizeFilter(
      1,
      static_cast<unsigned int>((m_numSamples + blockSizeFilter[1] - 1) /
                                blockSizeFilter[1]),
      static_cast<unsigned int>((m_numScanlines + blockSizeFilter[2] - 1) /
                                blockSizeFilter[2]));

  sycl::event e = q.submit([&](sycl::handler& cgh) {
    auto inImageData_get_ct0 = input_dev;
    auto m_hilbertFilter_get_ct1 = m_hilbertFilter_dev;
    auto pEnv_get_ct2 = output_dev;
    auto numSamples_ct3 = m_numSamples;
    auto numScanlines_ct4 = m_numScanlines;
    auto m_filterLength_ct5 = (int)m_filterLength;

    cgh.parallel_for<class hilbert>(
        sycl::nd_range<3>(gridSizeFilter * blockSizeFilter, blockSizeFilter),
        [=](sycl::nd_item<3> item_ct1) {
          kernelFilterDemodulation<float, float>(
              inImageData_get_ct0, m_hilbertFilter_get_ct1, pEnv_get_ct2,
              numSamples_ct3, numScanlines_ct4, m_filterLength_ct5, item_ct1);
        });
  });

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
  if(m_hilbertFilter) sycl::free(m_hilbertFilter, q);
  if(m_hilbertFilter_dev) sycl::free(m_hilbertFilter_dev, q);
  if(output_dev) sycl::free(output_dev, q);
  if(output) sycl::free(output, q);
}