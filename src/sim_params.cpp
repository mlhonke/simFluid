//
// Created by graphics on 2020-06-13.
//

#include "sim_params.hpp"
#include <cuda_runtime.h>
#include "cuda_errorcheck.hpp"

SimParams* copy_params_to_device(SimParams &params) {
    SimParams* DEV_params;
    cuda_check(cudaMalloc((void **) &DEV_params, sizeof(SimParams)));
    cuda_check(cudaMemcpy(DEV_params, &params, sizeof(SimParams), cudaMemcpyHostToDevice));
    return DEV_params;
}