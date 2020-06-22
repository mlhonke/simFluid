//
// Created by graphics on 19/07/19.
//

#include "unit.cuh"
#include "advect_dev.cuh"
#include "cuvec.cuh"
#include <cuda_runtime.h>

__global__ void test_cuda_lerp(CUVEC::Vec3d pos, scalar_t* Q){
    scalar_t retval = cu_grid_trilerp(pos, Q, 10, 10, 10, 0.1);
    printf("Value lerp'd on CUDA is %f\n", retval);
}

__host__ void test_cuda_lerp_on_device(CUVEC::Vec3d pos, scalar_t* Q){
    test_cuda_lerp<<<1,1>>>(pos, Q);
}