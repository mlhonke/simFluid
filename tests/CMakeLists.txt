cmake_minimum_required(VERSION 3.12)

SET(CMAKE_C_COMPILER /usr/bin/gcc-8)
SET(CMAKE_CXX_COMPILER /usr/bin/g++-8)

project(testSimFluid LANGUAGES C CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/build EXCLUDE_FROM_ALL)

SET(CMAKE_CXX_FLAGS "-std=c++11 -g")

file(GLOB includes ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*)

add_executable(testSimFluid ${includes} ${sources})
target_link_libraries(testSimFluid simFluid)
