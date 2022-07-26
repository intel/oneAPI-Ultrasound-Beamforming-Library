//==================================================
// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
//
// @file sycl_help.hpp
//==================================================
#ifndef _SYCL_HELP_HPP_
#define _SYCL_HELP_HPP_

#include "CL/sycl.hpp"

// #include "dpc_common.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image/stb_image_resize.h"

#define SHOW_IMG_WIDTH 350
#define SHOW_IMG_HEIGHT 400
#define SHOW_IMG_CHANNELS 1

using namespace std;
using namespace cl::sycl;

int mkpath(std::string &s)
{
    size_t pre = 0;
    size_t end = 0;
    std::string sub;
    int ret = 0;

    if(s[s.size() - 1] != '/'){
        s += '/';
    }

    while((end = s.find_first_of('/',pre)) != std::string::npos){
        sub = s.substr(0, end);
        pre = ++end;
        if(sub.size()==0) continue;
        if((ret = ::mkdir(sub.c_str(), 0755)) && errno!=EEXIST){
            return ret;
        }
    }
    return ret;
}

void stb_write_img_u16_t(const std::string img_name, size_t h, size_t w,
                         uint16_t *&r_input, uint16_t *&g_input,
                         uint16_t *&b_input) {
  // we need to store r, g, b data into on array.
  uint8_t *data = new uint8_t[w * h * 3];
  size_t index = 0;
  for (size_t i = 0; i < h; i++) {
    for (size_t j = 0; j < w; j++) {
      data[index++] = r_input[i * w + j] >> 6;
      data[index++] = g_input[i * w + j] >> 6;
      data[index++] = b_input[i * w + j] >> 6;
    }
  }

  std::string img_jpg("jpg");

  std::size_t found = img_name.find(img_jpg);
  if (found != std::string::npos) {
    stbi_write_jpg(img_name.c_str(), w, h, 3, data, 1 << 16);
  } else {
    stbi_write_png(img_name.c_str(), w, h, 3, data, w * 3);
  }

  delete[] data;
}

void stb_write_img_u16_single_channel(const std::string img_name, size_t h,
                                      size_t w, uint16_t *&r_input) {
  // we need to store r, g, b data into on array.
  uint8_t *data = new uint8_t[w * h];
  size_t index = 0;
  for (size_t i = 0; i < h; i++) {
    for (size_t j = 0; j < w; j++) {
      data[index++] = r_input[i * w + j] >> 6;
    }
  }

  std::string img_jpg("jpg");

  std::size_t found = img_name.find(img_jpg);
  if (found != std::string::npos) {
    stbi_write_jpg(img_name.c_str(), w, h, 3, data, 1 << 16);
  } else {
    stbi_write_png(img_name.c_str(), w, h, 3, data, w * 3);
  }

  delete[] data;
}

void stb_write_img_u8_t(const std::string img_name, size_t h, size_t w,
                        uint8_t *&r_input, uint8_t *&g_input,
                        uint8_t *&b_input) {
  // we need to store r, g, b data into on array.
  uint8_t *data_u8_t = new uint8_t[w * h * 3];
  size_t index = 0;
  for (size_t i = 0; i < h; i++) {
    for (size_t j = 0; j < w; j++) {
      data_u8_t[index++] = r_input[i * w + j];
      data_u8_t[index++] = g_input[i * w + j];
      data_u8_t[index++] = b_input[i * w + j];
    }
  }

  std::string img_jpg("jpg");

  std::size_t found = img_name.find(img_jpg);
  if (found != std::string::npos) {
    stbi_write_jpg(img_name.c_str(), w, h, 3, data_u8_t, 1 << 16);
  } else {
    stbi_write_png(img_name.c_str(), w, h, 3, data_u8_t, w * 3);
  }

  delete[] data_u8_t;
}

void stb_write_img_u8_single_channel(const std::string img_name, size_t in_h,
                                     size_t in_w, float *&r_input) {
  // we need to store r, g, b data into on array.
  uint8_t *data_u8_t = new uint8_t[in_h * in_w];
  size_t index = 0;
  for (size_t i = 0; i < in_h; i++) {
    for (size_t j = 0; j < in_w; j++) {
      data_u8_t[index++] = static_cast<uint8_t>(
          std::min(static_cast<double>(abs(r_input[i * in_w + j])), 255.0));
    }
  }

  uint8_t *image_resize = new uint8_t[SHOW_IMG_WIDTH * SHOW_IMG_HEIGHT];

  stbir_resize_uint8(data_u8_t, in_w, in_h, 0, image_resize, SHOW_IMG_WIDTH,
                     SHOW_IMG_HEIGHT, 0, 1);

  std::string img_jpg("jpg");

  std::size_t found = img_name.find(img_jpg);
  if (found != std::string::npos) {
    stbi_write_jpg(img_name.c_str(), in_w, in_h, 3, data_u8_t, 1 << 16);
  } else {
    stbi_write_png(img_name.c_str(), SHOW_IMG_WIDTH, SHOW_IMG_HEIGHT,
                   SHOW_IMG_CHANNELS, image_resize, SHOW_IMG_WIDTH);
    ;
  }

  delete[] data_u8_t;
}

void SaveImage(std::string save_file_Name, float *res) {
  stb_write_img_u8_single_channel(save_file_Name, 2000, 255, res);
}

void SaveImage1(std::string save_file_Name, float *res) {
  stb_write_img_u8_single_channel(save_file_Name, 2000, 1667, res);
}

#endif  //_SYCL_HELP_HPP_
