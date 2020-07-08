cmake_minimum_required(VERSION 3.12)

SET(CMAKE_C_COMPILER /usr/bin/gcc-8)
SET(CMAKE_CXX_COMPILER /usr/bin/g++-8)

project(stairs LANGUAGES C CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_SOURCE_DIR}/build EXCLUDE_FROM_ALL)

SET(CMAKE_CXX_FLAGS "-std=c++11 -g -fopenmp")

add_executable(stairs main.cpp)
target_link_libraries(stairs simFluid)
