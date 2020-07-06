//
// Created by graphics on 2020-07-03.
//

#ifndef TESTSIMFLUID_TEST_TOOLS_H
#define TESTSIMFLUID_TEST_TOOLS_H

#include "sim_params.hpp"
#include "sim_external_types.hpp"

template <class T> bool almost_equal(T a, T b, T eps){
    return std::abs(a-b) < eps;
}

void init_basic_cube_simulation(int cube_dim, SimParams* &C, macVel &V, scalar_t &dt);
void init_basic_cube_CUDA_simulation(int cube_dim, SimParams *&C, SimParams* &DEV_C, macVel &V, DEV_macVel &DEV_V,
                                     scalar_t &dt);

#endif //TESTSIMFLUID_TEST_TOOLS_H
