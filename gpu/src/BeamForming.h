// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#ifndef ULTRASOUND_H
#define ULTRASOUND_H

#include <CL/sycl.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "utility.hpp"

using namespace std;

class Beamforming2D {
 public:
  Beamforming2D(sycl::queue &in_q);

  ~Beamforming2D(void);

  int GetInputImage(const char *Paramfilename, const char *Inputfilename, RawParam* &Params);

  int read_one_frame2dev(int16_t* raw_ptr, size_t len);

  int copy_data2dev();

  void SubmitKernel(int16_t* raw_ptr, size_t len);

  float *getRes();

  float *getResHost();

  sycl::queue q;
  ScanlineRxParameters3D *rxScanlines = NULL;
  float *rxDepths = NULL;
  float *rxElementXs = NULL;
  float *rxElementYs = NULL;
  float *window_data = NULL;
  int16_t *RFdata = NULL;

  float *s = NULL;

  float *rxDepths_dev = NULL;
  float *rxElementXs_dev = NULL;
  float *rxElementYs_dev = NULL;
  float *window_data_dev = NULL;
  float window_scale = 0;
  int16_t *RFdata_dev = NULL;
  ScanlineRxParameters3D *rxScanlines_dev = NULL;

  float *s_dev = NULL;

  uint Width = 0;
  uint Height = 0;

  size_t numElements = 0;
  size_t numReceivedChannels = 0;
  size_t numSamples = 0;
  size_t numTxScanlines = 0;
  size_t numRxScanlines = 0;
  size_t rxNumDepths = 0;
  vec2s scanlineLayout = {0, 0};
  vec2s elementLayout = {0, 0};
  double depth = 0;
  double samplingFrequency = 0;
  double speedOfSoundMMperS = 0;
  double dt = 0;

  int additionalOffset = 0;
  float fNumber = 0;
  float speedOfSound = 0;
  bool interpolateRFlines = 0;
  bool interpolateBetweenTransmits = 0;

  float windowParameter = 0;
  size_t numEntriesPerFunction = 0;
  BeamformingType mBeamformingType = BeamformingType::DelayAndSum;

  uint8_t *m_mask;
  uint32_t *m_sampleIdx;
  float *m_weightX;
  float *m_weightY;

  vec3s m_imageSize = {0, 0, 0};
  vec2i m_outputSize = {0, 0};

  std::vector<double> comsuming_time;
  std::vector<double> memcpy_time;
};

#endif