//
// Created by graLScs on 12/01/19.
//

#include "sim_levelset.hpp"
#include <queue>

#ifndef USECUDA
#ifndef FERRO_SIM_PLS_HPP
#define FERRO_SIM_PLS_HPP

class SimPLS : public SimLevelSet {
public:
    SimPLS(SimParams &C, std::array<CubeX, 3> &V);
    void advance(int cur_step, scalar_t dt);
    scalar_t get_curvature(Vector3 &pos);
//    void advect_particles(scalar_t dt, std::vector<Vector3> &points);
    void advect_particles(scalar_t dt, std::vector<Vector3> &points, int start, int end);

    // Three kinds of particles
    std::vector<Vector3> surface_points;
    std::vector<std::vector<Vector3>> grid_particles;
    std::vector<Vector3> pos_points;
    std::vector<Vector3> neg_points;

protected:
    void reset_LS(const CubeX& LS_new);

    Vector3 get_normal(const Vector3& pos);
    bool get_surface_point(const Vector3& pos, Vector3& result);

    void redistance_neighbour(const Vector3i &face, CubeX &unsigned_dist, CubeXi &closest_point, std::queue<Vector3i> &cell_queue, int cp);
    void redistance_grid_from_particles();
    void generate_sign_field(const std::vector<Vector3> &points, CubeX &field);
    void correct_grid_point_signs();
    void check_particle_bounds();
    void print_L1_norm_min_max(const std::vector<Vector3> &points);

    void reseed_particles();

    virtual scalar_t get_height_normal_pls(Vector3 base_in, Vector3 n_in, scalar_t h_ref, scalar_t dx, int n_search = 7);
    virtual scalar_t get_height_normal(Vector3 &base_in, Vector3 &n_in, scalar_t h_ref, scalar_t dx);

    void sort_particles();

    scalar_t bandwidth; //in number of cells
    int reseed_interval;
    int iteration;
    int surface_particles_per_cell;
    int sign_particles_per_cell;
    int n_threads = 3;

    int ni, nj, nk;
};

#endif //FERRO_SIM_PLS_HPP
#endif