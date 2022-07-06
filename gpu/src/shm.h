// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fcntl.h>
#include <unistd.h>

// System V  shared memory headers
//#include <sys/shm.h>

/**
 *  Posix semaphore headers
 */

#include <semaphore.h>

/**
 * @brief Some access mode for shared memory segment and semaphore
 */

#define ACC_MODE                                                         \
  (S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | \
   S_IWOTH | S_IXOTH)
#define CREAT_FLAG (O_CREAT | O_RDWR)

/**
 * @brief define common used posix shared memory object name and
 * semaphore name
 *
 */

#define n_shm 2
const char* shm_name[n_shm] = {"shm_one", "shm_two"};

#define n_sem 2
const char* sem_name[n_sem] = {"sem_nempty", "sem_nstored"};
const char* real_sem_name[n_sem] = {"sem.sem_nempty", "sem.sem_nstored"};

const int num_ite = 10;
