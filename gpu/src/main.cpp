// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "BeamForming.h"
#include "HilbertFirEnvelope.h"
#include "LogCompressor.h"
#include "ScanConverter.h"
#include "shm.h"
#include "sycl_help.hpp"

#include <numeric>
#include <cstdlib>

using namespace std;
using namespace sycl;

#define SAVE_IMG 1

int main(int argc, char **argv) {
  const char *fileparam = argv[1];
  const char *filein = argv[2];

  string fileout("./res");

  if(argc == 4)
    fileout = string(argv[3]);

  int mkdir = mkpath(fileout);

  auto property_list =
      cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
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

  int fd_1, fd_2;

  fd_1 = shm_open(shm_name[0], O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
  fd_2 = shm_open(shm_name[1], O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);

  if (-1 == fd_1 || -1 == fd_2) {
    std::cout << "Opening shared memory fails..\n";
    exit(-1);
  }

  struct stat shm_stat1, shm_stat2;
  int err_1 = fstat(fd_1, &shm_stat1);
  int err_2 = fstat(fd_2, &shm_stat2);

  if (-1 == err_1 || -1 == err_2) {
    std::cout << "Retrive shared memory segment info fail..\n";
    exit(-1);
  }

  std::cout << shm_name[0]
            << " shared memory segment size: " << shm_stat1.st_size
            << " bytes.\n";
  std::cout << shm_name[1]
            << " shared memory segment size: " << shm_stat2.st_size
            << " bytes.\n";

  int16_t *ptr_1 = (int16_t *)mmap(NULL, shm_stat1.st_size,
                                   PROT_READ | PROT_WRITE, MAP_SHARED, fd_1, 0);
  int16_t *ptr_2 = (int16_t *)mmap(NULL, shm_stat2.st_size,
                                   PROT_READ | PROT_WRITE, MAP_SHARED, fd_2, 0);

  std::cout << "shm_read mmap virutal memory address(should be page-aligned): "
            << ptr_1 << std::endl;
  std::cout << "shm_read mmap virutal memory address(should be page-aligned): "
            << ptr_2 << std::endl;

  vector<int16_t *> ptr = {ptr_1, ptr_2};

  /**
   * @brief add semaphore for data sync
   *
   */
  sem_t *nempty;
  sem_t *nstored;

  /**
   * @brief Just open the alreay existing semaphore
   *
   */
  nempty = sem_open(sem_name[0], 0);
  nstored = sem_open(sem_name[1], 0);

  if (nempty == SEM_FAILED || nstored == SEM_FAILED) {
    std::cout << "Open existing semaphore fail...\n";
    exit(-1);
  }

  uint num_run = 0;
  size_t raw_len = params->numReceivedChannels * params->numSamples * params->numTxScanlines;

  for (size_t i = 0; i < 8; i++) {
    sem_wait(nstored);
    beamformer.read_one_frame2dev(ptr[i % n_shm], raw_len);
    beamformer.SubmitKernel(ptr[i % n_shm], raw_len);

#if SAVE_IMG
    std::string file_path1 = fileout + "/frame_bf_" + std::to_string(num_run) + ".png";
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

    sem_post(nempty);
  }

  double total_time = 0;

  std::cout << std::endl << "====Summary====" << std::endl;
  std::cout << "Raw data copy avg time for 1 frame : " << AvgVec(beamformer.memcpy_time) << " ms." << std::endl;
  std::cout << "BeamForming avg time for 1 frame : " << AvgVec(beamformer.comsuming_time) << " ms." << std::endl;
  std::cout << "HilbertEnvelop avg time for 1 frame : " << AvgVec(hilbertenvelope.comsuming_time) << " ms." << std::endl;
  std::cout << "LogCompressor avg time for 1 frame : " << AvgVec(logcompressor.comsuming_time) << " ms." << std::endl;
  std::cout << "ScanConvertor avg time for 1 frame : " << AvgVec(scanconvertor.comsuming_time) << " ms." << std::endl;
  total_time += AvgVec(beamformer.comsuming_time) + AvgVec(hilbertenvelope.comsuming_time) + AvgVec(logcompressor.comsuming_time) + AvgVec(logcompressor.comsuming_time);
  std::cout << "FPS without data copy : " << 1000 / total_time << std::endl;
  total_time += AvgVec(beamformer.memcpy_time);
  std::cout << "FPS with data copy : " << 1000 / total_time << std::endl;

  if (params != nullptr) delete params;

  return 0;
}