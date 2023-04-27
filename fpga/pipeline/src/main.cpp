// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "BeamForming.h"
#include "HilbertFirEnvelope.h"
#include "LogCompressor.h"
#include "ScanConverter.h"
#include "dpc_common.hpp"
#include "sycl_help.hpp"

using namespace std;

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

  mkpath(fileout);

#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  sycl::property_list properties{sycl::property::queue::enable_profiling()};

  sycl::queue q(selector, properties);

  std::cout << "Running on device: "
            << q.get_device().get_info<info::device::name>() << "\n";

  BeamformingType type = DelayAndSum;
  Beamforming2D beamformer(q);

  int ret = beamformer.GetInputImage(fileparam, filein, type);
  if (ret) {
    std::cout << "Read file success.\n";
  }

  ret = beamformer.copy_data2dev();
  if (ret) {
    std::cout << "Copy data to device success.\n";
  }

  HilbertFirEnvelope hilbertenvelope(q);

  hilbertenvelope.prepareFilter();

  LogCompressor logcompressor(q);

  ScanConverter scanconvertor(
      q, beamformer.m_mask, beamformer.m_sampleIdx, beamformer.m_weightX,
      beamformer.m_weightY, beamformer.m_imageSize);

  int num_run = 0;

  for (size_t i = 0;;) {
    beamformer.read_one_frame2dev(beamformer.RFdata + raw_len * i, raw_len);
    beamformer.SubmitKernel();

    hilbertenvelope.SubmitKernel();

    logcompressor.SubmitKernel();

    scanconvertor.SubmitKernel();

    beamformer.e.wait();
    Report_time(std::string("beamforming kernel: "), beamformer.e);

    hilbertenvelope.e.wait();
    Report_time(std::string("hilbertenvelope kernel: "), hilbertenvelope.e);

    logcompressor.e.wait();
    Report_time(std::string("logcompressor kernel: "), logcompressor.e);

    scanconvertor.e.wait();
    Report_time(std::string("scanconvertor kernel: "), scanconvertor.e);

#if SAVE_IMG
    std::string file_path1 = fileout + "frame_bf_" + std::to_string(num_run) + ".png";
    SaveImage(file_path1, beamformer.getResHost());

    std::string file_path2 = fileout + "frame_he_" + std::to_string(num_run) + ".png";
    SaveImage(file_path2, hilbertenvelope.getResHost());

    std::string file_path3 = fileout + "frame_lc_" + std::to_string(num_run) + ".png";
    SaveImage(file_path3, logcompressor.getResHost());

    std::string file_path4 = fileout + "frame_sc_" + std::to_string(num_run) + ".png";
    SaveImage1(file_path4, scanconvertor.getResHost());
#endif

    num_run++;

    if (num_run == 8) break;
  }

  return 0;
}