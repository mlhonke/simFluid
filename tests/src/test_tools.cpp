//
// Created by graphics on 2020-07-03.
//

#include "test_tools.hpp"
#include "sim_utils.hpp"

void init_basic_cube_simulation(int cube_dim, SimParams *&C, macVel &V, scalar_t &dt) {
    C = new SimParams{cube_dim, cube_dim, cube_dim, 1.0/cube_dim};
    dt = 0.1;
    init_mac_velocities(*C, V);
}

void init_basic_cube_CUDA_simulation(int cube_dim, SimParams *&C, SimParams *&DEV_C, macVel &V, DEV_macVel &DEV_V,
                                     scalar_t &dt) {
    C = new SimParams{cube_dim, cube_dim, cube_dim, 1.0/cube_dim};
    DEV_C = copy_params_to_device(*C);
    init_mac_velocities(*C, V);
    init_cuda_mac_velocities(V, DEV_V);
    dt = 0.1;
}
