// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#ifndef ULTRASOUND_H
#define ULTRASOUND_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include "utility.hpp"

using namespace std;

class Beamforming2D {
 public:
  Beamforming2D(sycl::queue &in_q);

  ~Beamforming2D(void);

  int GetInputImage(const char *Paramfilename, const char *Inputfilename);

  int read_one_frame2dev(int16_t *raw_ptr, size_t len);

  int copy_data2dev();

  void SubmitKernel(int16_t *raw_ptr, size_t len);

  float *getRes();

  float *getResHost();

  sycl::queue q;
  ScanlineRxParameters3D *rxScanlines;
  float *rxDepths;
  float *rxElementXs;
  float *rxElementYs;
  float *window_data;
  int16_t *RFdata;

  float *s;

  float *rxDepths_dev;
  float *rxElementXs_dev;
  float *rxElementYs_dev;
  float *window_data_dev;
  float window_scale;
  int16_t *RFdata_dev;
  ScanlineRxParameters3D *rxScanlines_dev;

  float *s_dev;

  uint Width;
  uint Height;

  size_t numElements;
  size_t numReceivedChannels;
  size_t numSamples;
  size_t numTxScanlines;
  size_t numRxScanlines;
  size_t rxNumDepths;
  vec2s scanlineLayout;
  vec2s elementLayout;
  double depth;
  double samplingFrequency;
  double speedOfSoundMMperS;
  double dt;

  int additionalOffset;
  float fNumber;
  float speedOfSound;
  bool interpolateRFlines;
  bool interpolateBetweenTransmits;

  float windowParameter;
  size_t numEntriesPerFunction;
  BeamformingType mBeamformingType;

  uint8_t *m_mask;
  uint32_t *m_sampleIdx;
  float *m_weightX;
  float *m_weightY;

  vec3s m_imageSize;
};

#endif