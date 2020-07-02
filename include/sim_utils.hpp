//
// Created by graphics on 2020-06-09.
//

#ifndef SIM_UTILS_HPP
#define SIM_UTILS_HPP

#include "sim_params.hpp"
#include "sim_external_types.hpp"
#include <array>

void init_mac_velocities(SimParams &params, macVel &V);
void init_cuda_mac_velocities(macVel &V, DEV_macVel &DEV_V);
void update_mac_velocities_on_host(macVel &V, DEV_macVel &DEV_V);
void update_mac_velocities_on_device(macVel &V, DEV_macVel &DEV_V);
void free_mac_velocities_on_device(DEV_macVel &DEV_V);

Vector3ui convert_index_to_coords(unsigned int d, unsigned int rows, unsigned int cols);
bool is_coord_valid(Vector3i I, Vector3i dims);
Vector3 get_position(int i, int j, int k, scalar_t dx);
Vector3 get_position(Vector3ui pos, scalar_t dx);

template<typename T, typename R> R slice_y(const T &a){
    return a.subcube(0, a.n_cols/2, 0, a.n_rows-1, a.n_cols/2, a.n_slices-1);
}

template<typename T, typename R> R slice_x(const T &a){
    return a.subcube(a.n_rows/2, 0, 0, a.n_rows/2, a.n_cols-1, a.n_slices-1);
}

#endif //SIM_UTILS_HPP
