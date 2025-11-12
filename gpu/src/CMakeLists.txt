# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later

option (USE_ZMC "Use Zero Memory Copy" ON)
message(STATUS "USE_ZMC is ${USE_ZMC}")

add_executable(easy_app
    easy_app.cpp
    BeamForming.dp.cpp
    HilbertFirEnvelope.dp.cpp
    LogCompressor.dp.cpp
    ScanConverter.dp.cpp
    BeamForming.h
    HilbertFirEnvelope.h
    LogCompressor.h
    ScanConverter.h
    sycl_help.h
    ultrasound_utility.h
    vec.h)

# Apply SYCL compile & link flags directly to the target (previous flags were never used).
if(MSVC)
    target_compile_options(easy_app PRIVATE -fsycl -Wall /EHsc)
else()
    target_compile_options(easy_app PRIVATE -fsycl -Wall)
endif()
target_link_options(easy_app PRIVATE -fsycl)

target_compile_definitions(easy_app PRIVATE $<$<BOOL:${USE_ZMC}>:USE_ZMC>)

if(CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
    target_compile_definitions(easy_app PRIVATE DEBUG_BUILD)
endif()
