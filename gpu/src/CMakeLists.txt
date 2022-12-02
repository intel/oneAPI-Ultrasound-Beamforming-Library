set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lrt -lpthread")

option (USE_ZMC "Use Zero Memory Copy" ON)
message("USE_ZMC is ${USE_ZMC}")
if(USE_ZMC)
    add_definitions(-DUSE_ZMC)
endif(USE_ZMC)

add_executable(easy_app easy_app.cpp BeamForming.dp.cpp BeamForming.h HilbertFirEnvelope.dp.cpp HilbertFirEnvelope.h LogCompressor.dp.cpp LogCompressor.h ScanConverter.dp.cpp ScanConverter.h sycl_help.h ultrasound_utility.h vec.h)