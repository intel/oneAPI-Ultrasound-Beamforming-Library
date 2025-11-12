cmake_minimum_required(VERSION 3.24)

# Select Intel oneAPI compiler BEFORE project() if not already IntelLLVM.
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    if(WIN32)
        find_program(INTEL_ONEAPI_CL NAMES icx-cl)
        if(INTEL_ONEAPI_CL)
            message(STATUS "Using Intel oneAPI CL compiler: ${INTEL_ONEAPI_CL}")
            set(CMAKE_CXX_COMPILER ${INTEL_ONEAPI_CL} CACHE FILEPATH "Intel oneAPI CL compiler" FORCE)
        endif()
    else()
        find_program(INTEL_ONEAPI NAMES icx icpx)
        if(INTEL_ONEAPI)
            message(STATUS "Using Intel oneAPI compiler: ${INTEL_ONEAPI}")
            set(CMAKE_CXX_COMPILER ${INTEL_ONEAPI} CACHE FILEPATH "Intel oneAPI compiler" FORCE)
        endif()
    endif()
endif()

project(gpu_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Unified output directories.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
