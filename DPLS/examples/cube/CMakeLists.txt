cmake_minimum_required(VERSION 3.12)

SET(CMAKE_C_COMPILER /usr/bin/gcc-8)
SET(CMAKE_CXX_COMPILER /usr/bin/g++-8)

#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#enable_language(CUDA)
#include(CheckLanguage)
#check_language(CUDA)

project(cubeDPLS LANGUAGES C CXX)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_SOURCE_DIR}/build EXCLUDE_FROM_ALL)

SET(CMAKE_CXX_FLAGS "-std=c++11 -g -fopenmp")
#SET(CMAKE_EXE_LINKER_FLAGS  "-lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/cublas -L/usr/local/cuda/lib64/libcusparse.so -L/usr/local/cuda/lib64/culibos -L/usr/local/cuda/lib64/cudart -ldl")

add_executable(cubeDPLS main.cpp)

target_link_libraries(cubeDPLS PRIVATE DPLS)
#target_include_directories(cubeDPLS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../libSimCommon/include)
#target_include_directories(cubeDPLS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../libSimCommon/includeCUDA)

#target_compile_options(cubeDPLS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#        -ccbin gcc-8
#        -arch=sm_60
#        >)


find_package(CUDA)
target_link_libraries(cubeDPLS ${CUDA_LIBRARY}) # unnecessary?
include_directories("${CUDA_INCLUDE_DIRS}")