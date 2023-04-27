// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "BeamForming.h"
#include "HilbertFirEnvelope.h"
#include "LogCompressor.h"
#include "ScanConverter.h"
#include "shm.h"
#include "sycl_help.hpp"

using namespace std;
using namespace sycl;

const size_t raw_len = 128 * 64 * 2337;

#define SAVE_IMG 1

int main(int argc, char **argv) {
  const char *fileparam = argv[1];
  const char *filein = argv[2];

  string fileout("./res");

  if(argc == 4)
  {
    string file_out(argv[3]);
    fileout = file_out;
  }

  int mkdir = mkpath(fileout);

  auto property_list =
      sycl::property_list{sycl::property::queue::enable_profiling()};
  sycl::queue in_q = sycl::queue(gpu_selector_v, property_list);
  std::cout << std::endl
            << "Selected device: "
            << in_q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  Beamforming2D beamformer(in_q);

   int ret = beamformer.GetInputImage(fileparam, filein);
  if (ret) {
    std::cout << "Read file success.\n";
  }

  ret = beamformer.copy_data2dev();
  if (ret) {
    std::cout << "Copy data to device success.\n";
  }

  HilbertFirEnvelope hilbertenvelope(in_q);
  hilbertenvelope.prepareFilter();

  LogCompressor logcompressor(in_q);
  ScanConverter scanconvertor(in_q, beamformer.m_mask,
        beamformer.m_sampleIdx, beamformer.m_weightX, beamformer.m_weightY,
        beamformer.m_imageSize);

  size_t num_run = 0;

  for (size_t i = 0; i < 8; i++) {
    beamformer.read_one_frame2dev(beamformer.RFdata + raw_len * i, raw_len);
    beamformer.SubmitKernel(beamformer.RFdata + raw_len * i, raw_len);

#if SAVE_IMG
    std::string file_path1 = fileout + "frame_bf_" + std::to_string(num_run) + ".png";
    SaveImage(file_path1, beamformer.getResHost());
#endif

    hilbertenvelope.getInput(beamformer.getRes());
    hilbertenvelope.SubmitKernel();

#if SAVE_IMG
    std::string file_path2 = fileout + "frame_he_" + std::to_string(num_run) + ".png";
    SaveImage(file_path2, hilbertenvelope.getResHost());
#endif

    logcompressor.getInput(hilbertenvelope.getRes());
    logcompressor.SubmitKernel();

#if SAVE_IMG
    std::string file_path3 = fileout + "frame_lc_" + std::to_string(num_run) + ".png";
    SaveImage(file_path3, logcompressor.getResHost());
#endif

    scanconvertor.getInput(logcompressor.getRes());
    scanconvertor.SubmitKernel();

#if SAVE_IMG
    std::string file_path4 = fileout + "frame_sc_" + std::to_string(num_run) + ".png";
    SaveImage1(file_path4, scanconvertor.getResHost());
#endif

    num_run++;
  }

  return 0;
}