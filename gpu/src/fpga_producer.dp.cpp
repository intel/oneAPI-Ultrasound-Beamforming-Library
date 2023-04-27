// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "dpc_common.hpp"
#include "shm.h"

using namespace std;
using namespace sycl;

const size_t raw_len = 128 * 64 * 2337;

/*  Provide shared memory object name.
 *  linux will place the shared memory at the /dev/shm/ file system
 */

int read_all_RFdata(const char *Inputfilename, int16_t *&rfdata,
                    uint32_t &num_frame, int64_t &file_len);

void clean_up_shm_sem();

static void Report_time(const std::string &msg, sycl::event e) {
  sycl::cl_ulong time_start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();

  sycl::cl_ulong time_end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();

  double elapsed = (time_end - time_start) / 1e6;
  std::cout << msg << elapsed << " milliseconds\n";
}

int main(int argc, char *argv[]) {
  const char *fileparam = argv[1];
  const char *filein = argv[2];

#ifdef FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#else
    auto selector = sycl::ext::intel::fpga_selector_v;
#endif

  auto props = sycl::property_list{sycl::property::queue::enable_profiling()};
  sycl::queue q(selector, props);
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  // make all shared memory object and semaphore are cleaned
  clean_up_shm_sem();

  int fd_1, fd_2, flags;
  int16_t *ptr_1, *ptr_2;
  struct stat stat_shmem1, stat_shmem2;

  flags = O_RDWR | O_CREAT;
  int mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;

  fd_1 = shm_open(shm_name[0], flags, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
  fd_2 = shm_open(shm_name[1], flags, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);

  if (fd_1 == -1 || fd_2 == -1) {
    std::cout << "Create shared memory object fail..\n";
    exit(-1);
  }

  int err_1 = ftruncate(fd_1, raw_len * sizeof(int16_t));
  int err_2 = ftruncate(fd_2, raw_len * sizeof(int16_t));
  if (err_1 == -1 || err_2 == -1) {
    std::cout << "Resize shared memory segment fail..\n";
    exit(-1);
  }

  err_1 = fstat(fd_1, &stat_shmem1);
  err_2 = fstat(fd_2, &stat_shmem2);
  if (err_1 == -1 || err_2 == -1) {
    std::cout << "Detect shared memory segment size fail..\n";
    exit(-1);
  }

  std::cout << "Shared memory segment 1 size: " << stat_shmem1.st_size
            << " bytes. " << stat_shmem1.st_size / (1 << 20) << " MB.\n";
  std::cout << "Shared memory segment 2 size: " << stat_shmem2.st_size
            << " bytes. " << stat_shmem2.st_size / (1 << 20) << " MB.\n";

  ptr_1 = (int16_t *)mmap(NULL, raw_len * sizeof(int16_t),
                          PROT_READ | PROT_WRITE, MAP_SHARED, fd_1, 0);
  ptr_2 = (int16_t *)mmap(NULL, raw_len * sizeof(int16_t),
                          PROT_READ | PROT_WRITE, MAP_SHARED, fd_2, 0);

  std::vector<int16_t *> ptr = {ptr_1, ptr_2};

  std::cout << "mmap in shm_create for shm 1 virtual memory address(should be "
               "page-aligned): "
            << ptr_1 << std::endl;
  std::cout << "mmap in shm_create for shm 2 virtual memory address(should be "
               "page-aligned): "
            << ptr_2 << std::endl;

  // add posix semaphore to implement "double buffering"
  sem_t *nempty;
  sem_t *nstored;

  nempty = sem_open(sem_name[0], CREAT_FLAG, ACC_MODE, n_sem);
  nstored = sem_open(sem_name[1], CREAT_FLAG, ACC_MODE, 0);

  if (nempty == SEM_FAILED || nstored == SEM_FAILED) {
    std::cout << "Create sem fail in shm_create...\n";
    exit(-1);
  }

  int16_t *raw_ptr;
  uint32_t num_frames;
  int64_t file_len;
  int ret = read_all_RFdata(filein, raw_ptr, num_frames, file_len);

  int16_t *fpga_ptr = (int16_t *)sycl::malloc_device(file_len, q);
  sycl::event e1 = q.memcpy(fpga_ptr, raw_ptr, file_len);
  e1.wait();
  Report_time(std::string("Copy all data to FPGA DDR: "), e1);

  int sub_time = 0;

  for (size_t i = 0; i < 8; i++) {
    err_1 = sem_wait(nempty);
    if (err_1 == -1) {
      std::cout << "sem wait fail...\n";
      exit(-1);
    }

    sycl::event e = q.memcpy(ptr[i % n_shm], fpga_ptr + sub_time * raw_len,
                             raw_len * sizeof(int16_t));
    e.wait();
    std::string str_out = "Copy " + std::to_string(sub_time) +
                          "st frame from FPGA DDR to system memory: ";
    Report_time(str_out, e);

    err_1 = sem_post(nstored);
    if (err_1 == -1) {
      std::cout << "sem_post fail...\n";
      exit(-1);
    }

    if (sub_time++ >= 7) {
      std::cout << "Read raw data finished.\n";
      break;
    }
  }

  clean_up_shm_sem();

  if(fpga_ptr) sycl::free(fpga_ptr, q);
  if(raw_ptr) free(raw_ptr);

  return 0;
}

int read_all_RFdata(const char *Inputfilename, int16_t *&rfdata,
                    uint32_t &num_frame, int64_t &file_len) {
  // read input data file
  ifstream rf(Inputfilename, ios::in | ios::binary);
  rf.seekg(0, ios::end);
  int64_t filelength = rf.tellg();
  rf.seekg(0);

  size_t valid_bytes = 128 * 64 * 2337 * sizeof(int16_t);
  uint32_t numFrames = filelength / valid_bytes;

  rfdata = (int16_t *)malloc(filelength);
  rf.read((char *)rfdata, filelength);
  rf.close();

  num_frame = numFrames;
  file_len = filelength;

  return 1;
}

void clean_up_shm_sem() {
  shm_unlink(shm_name[0]);
  shm_unlink(shm_name[1]);
  shm_unlink(real_sem_name[0]);
  shm_unlink(real_sem_name[1]);
}