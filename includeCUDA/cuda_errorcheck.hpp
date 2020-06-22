//
// Created by graphics on 17/07/19.
//

#ifndef FERRO3D_CUDA_ERRORCHECK_HPP
#define FERRO3D_CUDA_ERRORCHECK_HPP

#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>

#define cuda_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif //FERRO3D_CUDA_ERRORCHECK_HPP