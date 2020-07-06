//
// Created by graphics on 2020-07-01.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.hpp"
#include "advect.hpp"
#include "sim_utils.hpp"
#include "test_tools.hpp"

TEST_CASE("Advect RK3") {
    SimParams *C;
    macVel V;
    scalar_t dt;
    int dim = 5;
    init_basic_cube_simulation(dim, C, V, dt);

    CubeX q = CubeX(C->grid_w, C->grid_h, C->grid_d);
    V[0].fill(1);
    V[1].fill(1);
    V[2].fill(1);
    dt = 1.0 / dim;
    scalar_t point_val = 1.0;

    auto c = coordIterator({0,0,0}, {dim-2,dim-2,dim-2});
    while(c.next()){
//        std::cout << c.get_string() << std::endl;
        q.fill(0);
        q(c.i, c.j, c.k) = point_val;
        advect_RK3(q, {0,0,0}, V, dt, *C, false, false);
        CHECK(almost_equal(q(c.i+1,c.j+1,c.k+1), point_val, 1e-14));
    }

    delete C;
}

TEST_CASE("Advect RK3 CUDA") {
    SimParams *C;
    SimParams *DEV_C;
    macVel V;
    DEV_macVel DEV_V;
    scalar_t dt;
    int dim = 5;
    init_basic_cube_CUDA_simulation(dim, C, DEV_C, V, DEV_V, dt);

    CubeX q = CubeX(C->grid_w, C->grid_h, C->grid_d);
    V[0].fill(1);
    V[1].fill(1);
    V[2].fill(1);
    update_mac_velocities_on_device(V, DEV_V);
    dt = 1.0 / dim;
    scalar_t point_val = 1.0;

    auto c = coordIterator({0,0,0}, {dim-2,dim-2,dim-2});
    while(c.next()){
        //        std::cout << c.get_string() << std::endl;
        q.fill(0);
        q(c.i, c.j, c.k) = point_val;
        advect_RK3_CUDA(q, {0,0,0}, DEV_V, dt, DEV_C, false, false);
        CHECK(almost_equal(q(c.i+1,c.j+1,c.k+1), point_val, 1e-14));
    }
    free_mac_velocities_on_device(DEV_V);
    delete C;
}