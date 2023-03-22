// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#ifndef ULTRASOUND_H
#define ULTRASOUND_H
#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <CL/sycl.hpp>

#include "utility.hpp"

// using namespace cl::sycl;
using namespace std;

class Beamforming2D {
 public:
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Beamforming2D();
  Beamforming2D(sycl::queue in_q) : q(in_q){};
  ~Beamforming2D();

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int GetInputImage(const char *Paramfilename, const char *Inputfilename,
                    BeamformingType type);

  /**
   * @brief read raw data of one frame from host to device (RFdata_dev)
   *
   * @param raw_ptr the host buffer to read.
   * @param len  the read data size (typically 128 * 64 * 2337)
   * @return int
   */
  int read_one_frame2dev(int16_t *raw_ptr, size_t len);

  /**
   * @brief This copy all parameters to device, this function should be called
   * only once
   *
   * @return int
   */
  int copy_data2dev();

  /**
   * @brief submit beamforming kernel to device(GPU)
   *
   * @return int
   */
  void SubmitKernel();

  /**
   * @brief Save output image in grayscale format.
   *
   * @param saveFilename the filename to save to.
   * @return int
   */
  void save_img(std::string saveFilename);

  sycl::queue getStream();
  float *getRes();
  float *getResHost();

  sycl::event e;

  // private:
  sycl::queue q;
  ScanlineRxParameters3D *rxScanlines;
  ScanlineRxParameters3D *origin_rxScanlines;
  float *rxDepths;
  float *rxElementXs;
  float *rxElementYs;
  float *window_data;
  int16_t *RFdata;
  int16_t *RFdata_shuffle;
  unsigned char *RFdata_shuffle_s1;
  unsigned char *RFdata_shuffle_s2;
  float *s;

  float *rxDepths_dev;
  float *rxElementXs_dev;
  float *rxElementYs_dev;
  float *window_data_dev;
  float window_scale;
  unsigned char *RFdata_dev;
  unsigned char *RFdata_dev1;
  ScanlineRxParameters3D *rxScanlines_dev;

  float *s_dev;
  float *s_tmp;

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

  vec3s m_imageSize = {0, 0, 0};
};

#endif