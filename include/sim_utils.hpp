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
// TODO: Enforcing unsigned int vector for pos not really necessary. Function can return a negative position. Let caller decide what to do with that.
Vector3 get_position(Vector3ui pos, scalar_t dx);
Vector3 get_position(Vector3i pos, scalar_t dx);

template<typename T, typename R> R slice_y(const T &a){
    return a.subcube(0, a.n_cols/2, 0, a.n_rows-1, a.n_cols/2, a.n_slices-1);
}

template<typename T, typename R> R slice_x(const T &a){
    return a.subcube(a.n_rows/2, 0, 0, a.n_rows/2, a.n_cols-1, a.n_slices-1);
}

/* coordIterator:
 * An experiment in avoiding triple nested loops.
 * TODO: Test performance of using this iterator versus using a triple nested loop.
 * Potentially useful, along with templates, to allow this simulation to easily switch between 2D and 3D modes.
 */
class coordIterator{
public:
    coordIterator(const Vector3i &min_in, const Vector3i &max_in):min(min_in){
        max = max_in - min_in + 1;
        d_max = max[0]*max[1]*max[2];
        i = min_in(0); j = min_in(1); k = min_in(2);
    }
    coordIterator& operator++() {
        i = d % max[0] + min[0];
        j = (d / max[0]) % max[1] + min[1];
        k = d / (max[0] * max[1]) + min[2];
        d++;
        return *this;
    }

    bool next(){
        this->operator++();
        return !is_done();
    }

    Vector3i operator*() {return {i,j,k};}

    bool is_done() const {
        return d > d_max;
    }

    std::string get_string() const{
        std::string out;
        out = "{" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + "}";
        return out;
    }

    Vector3i current;
    int i, j, k;
private:
    Vector3i max, min;
    int d = 0;
    int d_max = 0;
};

#endif //SIM_UTILS_HPP
