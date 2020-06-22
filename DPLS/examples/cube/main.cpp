/* cube
 * A minimum running example on how to use SimPLSCUDA (DPLS on GPU).
 */

#include <iostream>
#include <cuda_runtime.h>

#include "sim_params.hpp"
#include "sim_utils.hpp"
#include "sim_pls_cuda.hpp"

int main(int argc, char** argv){
    int* DEV_array;
    cudaError_t rc = cudaMalloc(&DEV_array, 10*sizeof(int));
    if (rc != cudaSuccess){
        std::cout << "Cuda error at cudaMalloc" << std::endl;
    }
    rc = cudaFree(DEV_array);
    if (rc != cudaSuccess){
        std::cout << "Cuda error at cudaFree" << std::endl;
    }

    auto params = SimParams(10, 10, 10, 1.0/10.0);
    SimParams* DEV_params;
    DEV_params = copy_params_to_device(params);

    std::array<scalar_t*, 3> DEV_V{};
    std::array<CubeX, 3> V{};
    init_mac_velocities(params, V);
    V[0].fill(1.0);
    init_cuda_mac_velocities(V, DEV_V);
    update_mac_velocities_on_device(V, DEV_V);

    auto DPLS = new SimLevelSet(params, DEV_params, DEV_V);
    DPLS->initialize_level_set_rectangle({3, 3, 3}, {(double) params.grid_w-4, (double) params.grid_h-4, (double) params.grid_d-4});
    std::cout << DPLS->LS << std::endl;

    for (int i = 0; i < 100; i++){
        std::cout << "======================================================" << std::endl;
        std::cout << "Current Step: " << i << std::endl;
        std::cout << DPLS->LS << std::endl;
        DPLS->advance(i, 0.001);
    }

    return 0;
}