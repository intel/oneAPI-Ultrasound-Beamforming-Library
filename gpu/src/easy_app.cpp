// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "BeamForming.h"
#include "HilbertFirEnvelope.h"
#include "LogCompressor.h"
#include "ScanConverter.h"
#include "sycl_help.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <new>
#include <system_error>
#include <typeinfo>

using namespace std;
using namespace sycl;

const size_t raw_len = 128 * 64 * 2337;

#define SAVE_IMG 1

namespace {

int run_application(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <param_file> <raw_file> [output_dir]" << std::endl;
    return 1;
  }

  const char *fileparam = argv[1];
  const char *filein = argv[2];

  string fileout("./res");
  if (argc >= 4 && argv[3] && argv[3][0] != '\0') {
    fileout = string(argv[3]);
  }

  int run_steps = 8;

  std::error_code fs_ec;
  const auto output_path = std::filesystem::path(fileout);
  bool exists = std::filesystem::exists(output_path, fs_ec);
  if (fs_ec) {
    std::cerr << "Failed to inspect output directory '" << fileout
              << "': " << fs_ec.message() << std::endl;
    return 1;
  }
  if (!exists) {
    std::filesystem::create_directories(output_path, fs_ec);
    if (fs_ec) {
      std::cerr << "Failed to create output directory '" << fileout
                << "': " << fs_ec.message() << std::endl;
      return 1;
    }
  }

  auto property_list =
      sycl::property_list{sycl::property::queue::enable_profiling()};
  sycl::queue in_q = sycl::queue(gpu_selector{}, property_list);
  std::cout << std::endl
            << "Selected device: "
            << in_q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  Beamforming2D beamformer(in_q);

  RawParam *params = NULL;

  int ret = beamformer.GetInputImage(fileparam, filein, params);
  if (ret) {
    std::cout << "Read file success.\n";
  }

  if (params == nullptr) {
    std::cerr << "Parameter buffer is null after reading input." << std::endl;
    return 1;
  }

  ret = beamformer.copy_data2dev();
  if (ret) {
    std::cout << "Copy data to device success.\n";
  }

  HilbertFirEnvelope hilbertenvelope(in_q, params);
  hilbertenvelope.prepareFilter();

  LogCompressor logcompressor(in_q, params);
  ScanConverter scanconvertor(in_q, beamformer.m_mask,
        beamformer.m_sampleIdx, beamformer.m_weightX, beamformer.m_weightY,
        beamformer.m_imageSize, params);

  if (params == nullptr) {
    std::cerr << "Parameter buffer is null." << std::endl;
    return 1;
  }

  size_t num_run = 0;
  size_t raw_len = params->numReceivedChannels * params->numSamples * params->numTxScanlines;

  while(num_run < run_steps) {
    beamformer.read_one_frame2dev(beamformer.RFdata + raw_len * (num_run % 8), raw_len);
    beamformer.SubmitKernel(beamformer.RFdata + raw_len * (num_run % 8), raw_len);

#if SAVE_IMG
    std::string file_path1 = fileout + "/frame_" + std::to_string(num_run) + ".png";
    SaveImage(file_path1, beamformer.m_outputSize, beamformer.getResHost());
#endif

    hilbertenvelope.getInput(beamformer.getRes());
    hilbertenvelope.SubmitKernel();

#if SAVE_IMG
    std::string file_path2 = fileout + "/frame_he_" + std::to_string(num_run) + ".png";
    SaveImage(file_path2, hilbertenvelope.m_outputSize, hilbertenvelope.getResHost());
#endif

    logcompressor.getInput(hilbertenvelope.getRes());
    logcompressor.SubmitKernel();

#if SAVE_IMG
    std::string file_path3 = fileout + "/frame_lc_" + std::to_string(num_run) + ".png";
    SaveImage(file_path3, logcompressor.m_outputSize, logcompressor.getResHost());
#endif

    scanconvertor.getInput(logcompressor.getRes());
    scanconvertor.SubmitKernel();

#if SAVE_IMG
    std::string file_path4 = fileout + "/frame_sc_" + std::to_string(num_run) + ".png";
    SaveImage(file_path4, scanconvertor.m_outputSize, scanconvertor.getResHost());
#endif

    num_run++;
  }

  float total_time = 0;

  std::cout << std::endl << "====Summary====" << std::endl;
  std::cout << "Raw data copy avg time for 1 frame : " << AvgVec(beamformer.memcpy_time) << " ms." << std::endl;
  std::cout << "BeamForming avg time for 1 frame : " << AvgVec(beamformer.comsuming_time) << " ms." << std::endl;
  std::cout << "HilbertEnvelop avg time for 1 frame : " << AvgVec(hilbertenvelope.comsuming_time) << " ms." << std::endl;
  std::cout << "LogCompressor avg time for 1 frame : " << AvgVec(logcompressor.comsuming_time) << " ms." << std::endl;
  std::cout << "ScanConvertor avg time for 1 frame : " << AvgVec(scanconvertor.comsuming_time) << " ms." << std::endl;
  total_time += AvgVec(beamformer.comsuming_time) + AvgVec(hilbertenvelope.comsuming_time) + AvgVec(logcompressor.comsuming_time) + AvgVec(logcompressor.comsuming_time);
  if(total_time > 0)  {
    std::cout << "FPS without data copy : " << 1000 / total_time << std::endl;
  }
  total_time += AvgVec(beamformer.memcpy_time);
  if(total_time > 0)  {
    std::cout << "FPS with data copy : " << 1000 / total_time << std::endl;
  }

  delete params;

  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    return run_application(argc, argv);
  } catch (const std::bad_array_new_length &e) {
    std::cerr << "Failed to initialize beamformer inputs: " << e.what()
              << std::endl;
  } catch (const std::bad_alloc &e) {
    std::cerr << "Out of memory while initializing beamformer inputs: "
              << e.what() << std::endl;
  } catch (const std::bad_cast &e) {
    std::cerr << "Invalid cast detected: " << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Unexpected error: " << e.what() << std::endl;
  }
  return 1;
}