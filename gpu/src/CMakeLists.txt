# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -std=c++17")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lrt -lpthread")

option (USE_ZMC "Use Zero Memory Copy" ON)
message("USE_ZMC is ${USE_ZMC}")
if(USE_ZMC)
    add_definitions(-DUSE_ZMC)
endif(USE_ZMC)

add_executable(ultrasound main.cpp BeamForming.dp.cpp BeamForming.h HilbertFirEnvelope.dp.cpp HilbertFirEnvelope.h LogCompressor.dp.cpp LogCompressor.h ScanConverter.dp.cpp ScanConverter.h shm.h sycl_help.hpp utility.hpp vec.h)
add_executable(easy_app easy_app.cpp BeamForming.dp.cpp BeamForming.h HilbertFirEnvelope.dp.cpp HilbertFirEnvelope.h LogCompressor.dp.cpp LogCompressor.h ScanConverter.dp.cpp ScanConverter.h shm.h sycl_help.hpp utility.hpp vec.h)

## Below are for the fpga_producer build.

option (COMPILE_FPGA "Compile FPGA Binary" OFF)
message("Compile FPGA binary is ${COMPILE_FPGA}")
if(COMPILE_FPGA)
set(SOURCE_FILE fpga_producer.dp.cpp shm.h)
set(TARGET_NAME fpga_producer)
set(EMULATOR_TARGET ${TARGET_NAME}.emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_BOARD)
    set(FPGA_BOARD "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_BOARD was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_BOARD} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_BOARD}")
endif()

set(EMULATOR_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(HARDWARE_COMPILE_FLAGS "${WIN_FLAG} -fintelfpga")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS}")

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_link_libraries(${EMULATOR_TARGET} rt pthread)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS} -DFPGA_EMULATOR")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET} ultrasound easy_app)

###############################################################################
### Generate Report
###############################################################################
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_link_libraries(${FPGA_EARLY_IMAGE} rt pthread)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE} ultrasound easy_app)
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_link_libraries(${FPGA_TARGET} rt pthread)
add_custom_target(fpga DEPENDS ${FPGA_TARGET} ultrasound easy_app)
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.

endif(COMPILE_FPGA)