//
// Created by graphics on 2020-06-13.
//

#include <iostream>
#include "sim_params.hpp"
#include <cuda_runtime.h>
#include "cuda_errorcheck.hpp"

SimParams* copy_params_to_device(SimParams &params) {
    SimParams* DEV_params;
    cuda_check(cudaMalloc((void **) &DEV_params, sizeof(SimParams)));
    cuda_check(cudaMemcpy(DEV_params, &params, sizeof(SimParams), cudaMemcpyHostToDevice));
    return DEV_params;
}

void create_sim_params_from_args(int argc, char **argv, SimParams *&retC, int &i){
    if (argc > 1) {
        retC = new SimParams{
                std::stoi(argv[i++]),
                std::stoi(argv[i++]),
                std::stoi(argv[i++]),
                std::stof(argv[i++]),
                std::stof(argv[i++]),
                std::stof(argv[i++]),
                std::stof(argv[i++])
        };
    } else {
        retC = new SimParams();
    }
}