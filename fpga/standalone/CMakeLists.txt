# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later

project(FPGACompile CXX)
cmake_minimum_required (VERSION 3.4)

if(UNIX)
    # Direct CMake to use dpcpp rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER dpcpp)
else() # Windows
    # Force CMake to use dpcpp rather than the default C++ compiler/linker 
    # (needed on Windows only)
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (dpcpp IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory (src)