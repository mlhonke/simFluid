cmake_minimum_required(VERSION 3.12)

SET(CMAKE_C_COMPILER /usr/bin/gcc-8)
SET(CMAKE_CXX_COMPILER /usr/bin/g++-8)

project(simFluid LANGUAGES C CXX)

# Points CMAKE to directory containing FindLIBIGL.cmake and other CMAKE find files.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/glad EXCLUDE_FROM_ALL)
find_package(glfw3 REQUIRED)

# Find and prepare to use CUDA
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

include_directories(/usr/local/cuda/samples/common/inc/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libigl/include)

option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_PNG               "Use PNG"           ON)

find_package(LIBIGL REQUIRED)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fopenmp")
SET(CMAKE_EXE_LINKER_FLAGS  "-lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/cublas -L/usr/local/cuda/lib64/libcusparse.so -L/usr/local/cuda/lib64/culibos -L/usr/local/cuda/lib64/cudart -ldl")
#set(CMAKE_RUNTIME_LIBRARY STATIC)

find_package(Armadillo REQUIRED QUIET)
include_directories(${ARMADILLO_INCLUDE_DIRS})

file(GLOB simFluidHeaders ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
file(GLOB simFluidCUDAHeaders ${CMAKE_CURRENT_SOURCE_DIR}/includeCUDA/*)
file(GLOB simFluidSources ${CMAKE_CURRENT_SOURCE_DIR}/src/*)

file(GLOB DPLSHeaders ${CMAKE_CURRENT_SOURCE_DIR}/DPLS/include/*)
file(GLOB DPLSCUDAHeaders ${CMAKE_CURRENT_SOURCE_DIR}/DPLS/includeCUDA/*)
file(GLOB DPLSSources ${CMAKE_CURRENT_SOURCE_DIR}/DPLS/src/*)

file(GLOB marchingTets ${CMAKE_CURRENT_SOURCE_DIR}/marching_tets/*)
add_library(simFluid STATIC
        ${simFluidCUDAHeaders}
        ${simFluidHeaders}
        ${simFluidSources}
        ${DPLSCUDAHeaders}
        ${DPLSHeaders}
        ${DPLSSources}        
        ${marchingTets})

target_include_directories(simFluid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/DPLS/include)
target_include_directories(simFluid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/DPLS/includeCUDA)
target_include_directories(simFluid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(simFluid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includeCUDA)
target_include_directories(simFluid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/marching_tets)

target_link_libraries(simFluid Eigen3::Eigen)
#target_link_libraries(simFluid glad)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_include_directories(simFluid PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(simFluid ${CUDART_LIBRARY})

find_package(CUDA)
set_target_properties(simFluid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(simFluid PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(simFluid PROPERTIES CUDA_ARCHITECTURES 60)
target_link_libraries(simFluid ${CUDA_LIBRARY}) # unnecessary?
target_link_libraries(simFluid ${CUDA_cusparse_LIBRARY})
target_link_libraries(simFluid ${CUDA_cublas_LIBRARY})

target_compile_options(simFluid PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -ccbin gcc-8
        >)

#find_package(OpenGL REQUIRED)
#target_link_libraries(simFluid OpenGL::GL)
#
#find_package(GLUT REQUIRED)
#target_link_libraries(simFluid ${GLUT_LIBRARIES})
#
#find_package(GLEW REQUIRED)
#target_link_libraries(simFluid GLEW::GLEW)
#
#find_package(glfw3 REQUIRED)
target_link_libraries(simFluid glfw)

target_link_libraries(simFluid ${ARMADILLO_LIBRARIES})

target_link_libraries(simFluid igl::png)