//
// Created by graphics on 2020-06-09.
//

#include "sim_utils.hpp"
#include <cuda_runtime.h>
#include "cuda_errorcheck.hpp"

void init_mac_velocities(SimParams &params, macVel &V){
    V[0] = (CubeX(params.grid_w+1, params.grid_h, params.grid_d, arma::fill::zeros));
    V[1] = (CubeX(params.grid_w, params.grid_h+1, params.grid_d, arma::fill::zeros));
    V[2] = (CubeX(params.grid_w, params.grid_h, params.grid_d+1, arma::fill::zeros));
}

void init_cuda_mac_velocities(macVel &V, DEV_macVel &DEV_V){
    cuda_check(cudaMalloc(&DEV_V[0], V[0].n_elem*sizeof(scalar_t)))
    cuda_check(cudaMalloc(&DEV_V[1], V[1].n_elem*sizeof(scalar_t)))
    cuda_check(cudaMalloc(&DEV_V[2], V[2].n_elem*sizeof(scalar_t)))
}

void update_mac_velocities_on_host(macVel &V, DEV_macVel &DEV_V){
    cuda_check(cudaMemcpy(V[0].memptr(), DEV_V[0], V[0].n_elem*sizeof(scalar_t), cudaMemcpyDeviceToHost))
    cuda_check(cudaMemcpy(V[1].memptr(), DEV_V[1], V[1].n_elem*sizeof(scalar_t), cudaMemcpyDeviceToHost))
    cuda_check(cudaMemcpy(V[2].memptr(), DEV_V[2], V[2].n_elem*sizeof(scalar_t), cudaMemcpyDeviceToHost))
}

void update_mac_velocities_on_device(macVel &V, DEV_macVel &DEV_V){
    cuda_check(cudaMemcpy(DEV_V[0], V[0].memptr(), V[0].n_elem*sizeof(scalar_t), cudaMemcpyHostToDevice))
    cuda_check(cudaMemcpy(DEV_V[1], V[1].memptr(), V[1].n_elem*sizeof(scalar_t), cudaMemcpyHostToDevice))
    cuda_check(cudaMemcpy(DEV_V[2], V[2].memptr(), V[2].n_elem*sizeof(scalar_t), cudaMemcpyHostToDevice))
}

void free_mac_velocities_on_device(DEV_macVel &DEV_V){
    cuda_check(cudaFree(DEV_V[0]));
    cuda_check(cudaFree(DEV_V[1]));
    cuda_check(cudaFree(DEV_V[2]));
}

Vector3ui convert_index_to_coords(unsigned int d, unsigned int rows, unsigned int cols){
    unsigned int i = d % rows;
    unsigned int j = (d / rows) % cols;
    unsigned int k = d / (cols * rows);

    return {i, j, k};
}

bool is_coord_valid(Vector3i I, Vector3i dims){
    return (I(0) >= 0 && I(0) < dims(0) && I(1) >= 0 && I(1) < dims(1) && I(2) >= 0 && I(2) < dims(2));
}

Vector3 get_position(int i, int j, int k, scalar_t dx){
    return {i*dx, j*dx, k*dx};
}

Vector3 get_position(Vector3i pos, scalar_t dx){
    return {pos(0)*dx, pos(1)*dx, pos(2)*dx};
}

Vector3 get_position(Vector3ui pos, scalar_t dx){
    return {pos(0)*dx, pos(1)*dx, pos(2)*dx};
}