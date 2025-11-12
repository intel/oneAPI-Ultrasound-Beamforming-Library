cmake_minimum_required(VERSION 4.20)

project(gpu_lib LANGUAGES CXX)

# Try to locate a oneAPI/DPC++ compiler if the current compiler is not IntelLLVM.
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    find_program(DPCPP_EXE NAMES dpcpp icpx icx)
    if(DPCPP_EXE)
        message(STATUS "Switching CMAKE_CXX_COMPILER to ${DPCPP_EXE}")
        set(CMAKE_CXX_COMPILER ${DPCPP_EXE} CACHE FILEPATH "DPC++ compiler" FORCE)
    else()
        message(WARNING "DPC++ compiler not found in PATH; ensure setvars.bat was run.")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Unified output directories.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
