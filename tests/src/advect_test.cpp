//
// Created by graphics on 2020-07-01.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.hpp"
#include "advect.hpp"
#include "sim_utils.hpp"

void init_basic_cube_simulation(int cube_dim, SimParams* &C, macVel &V, scalar_t &dt){
    C = new SimParams{cube_dim, cube_dim, cube_dim, 1.0/cube_dim};
    dt = 0.1;
    init_mac_velocities(*C, V);
}

TEST_CASE("Advect RK3") {
    SimParams *C;
    macVel V;
    scalar_t dt;
    init_basic_cube_simulation(10, C, V, dt);

    CubeX q = CubeX(C->grid_w, C->grid_h, C->grid_d, arma::fill::zeros);


}