//
// Created by graphics on 2020-06-15.
//

#ifndef DPLS_SIM_PLS_CUDA_HPP
#define DPLS_SIM_PLS_CUDA_HPP

#include <queue>

#include "sim_levelset.hpp"
#include "cuvec.cuh"

class SimPLSCUDA : public SimLevelSet {
public:
    SimPLSCUDA(SimParams &C, SimParams* DEV_C, std::array<scalar_t*, 3> &DEV_V);
    void advance(int cur_step, scalar_t dt);
    void save_data();
    void load_data();

    void allocate_space_for_particles_on_device();
    void update_particles_on_device();
    void update_particles_on_device_from_host();
    void free_particles_on_device();

    bool get_height_normal_pls_search(const CUVEC::Vec3i &coord, const CUVEC::Vec3d &base, const CUVEC::Vec3d &n_unit, scalar_t h_ref, scalar_t p_tol, scalar_t &best_error, scalar_t &best_height);
    scalar_t get_height_normal_pls(const Vector3 &base_in, const Vector3 &n_in, scalar_t h_ref, scalar_t dx_column, int n_search = 7);
    scalar_t get_height_normal(const Vector3 &base_in, const Vector3 &n_in, scalar_t h_ref, scalar_t dx_column) override;

    int get_number_of_blocks(int n_particles);
//    void test_height_function();
    scalar_t get_curvature(Vector3 &pos) override;
    void print_item(int* item);

    // Levelset
    scalar_t* DEV_LS, *DEV_grad_LS_x, *DEV_grad_LS_y, *DEV_grad_LS_z;
    scalar_t* DEV_LS_pos, *DEV_LS_neg;
    int* DEV_cp;
    CubeXi cp;
    CubeX LS_pos, LS_neg, LS_unsigned;

    // Three kinds of particles
    std::vector<CUVEC::Vec3d> sp;
    std::vector<CUVEC::Vec3d> pp;
    std::vector<CUVEC::Vec3d> np;
    CUVEC::Vec3d *DEV_sp = nullptr;
    CUVEC::Vec3d *DEV_pp = nullptr;
    CUVEC::Vec3d *DEV_np = nullptr;

    // CPU version of particles
    std::vector<Vector3> surface_points;
    std::vector<Vector3> pos_points;
    std::vector<Vector3> neg_points;

    std::vector<int> sp_index, np_index, pp_index;
    std::vector<int> sp_count, np_count, pp_count;
    int* DEV_sp_index, *DEV_np_index, *DEV_pp_index;
    int* DEV_sp_count, *DEV_np_count, *DEV_pp_count;

    int threads_in_block;

    int reseed_interval;
    int surface_particles_per_cell;
    int sign_particles_per_cell;

private:
    void generate_indicies_for_particles(int* DEV_p_keys, int* &DEV_p_index, int* &DEV_p_count, std::vector<int> &p_index, std::vector<int> &p_count, int n_p);
    void redistance_neighbour(const Vector3i &face, CubeX &unsigned_dist, const CubeXi &closest_point, std::queue<Vector3i> &cell_queue, int cp_id, scalar_t prop_bw);
    void propagate_interface_distances(scalar_t prop_bw);
    void correct_grid_point_signs();
    void copy_particles_to_host();
    void send_particles_to_host_pls();
    void reinit_pls();
    void print_all_levelsets();

    int iteration;
    scalar_t bandwidth; //in number of cells
    int n_threads = 3;
    int n_sp, n_pp, n_np;
    int ni, nj, nk;
};

void sort_particles_by_key(CUVEC::Vec3d *p, int *cell_ids, int n_p);

#endif //DPLS_SIM_PLS_CUDA_HPP
