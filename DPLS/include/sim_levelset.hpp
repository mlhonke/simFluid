//
// Created by graphics on 11/01/19.
//

#ifndef FERRO_SIM_LEVELSET_HPP
#define FERRO_SIM_LEVELSET_HPP

#include "sim_params.hpp"
#include "interpolate.hpp"

class SimLevelSet {
public:
    SimLevelSet():grid_w(0), grid_h(0), grid_d(0), dx(0), sim_w(0), sim_h(0), sim_d(0), n_cells(0){}
#ifdef USECUDA
    SimLevelSet(SimParams &C, SimParams* DEV_C, std::array<scalar_t*, 3> &DEV_V);
#else
    SimLevelSet(SimParams &C, std::array<CubeX, 3> &V);
#endif
    void shared_init();

    // overriddable functions
    virtual scalar_t get_curvature(Vector3 &pos);
    virtual void advance(int cur_step, scalar_t dt);

    void redistance();
    void redistance_interface();
    void initialize_level_set_rectangle(Vector3 I_min, Vector3 I_max);
    void initialize_level_set_circle(const Vector3 &center, scalar_t radius);
    int get_index(int i, int j, int k);
    int get_index(Vector3ui coord);
    void save_data();
    void load_data();

    CubeX LS;
    CubeX LS_K;
    CubeX LS_grad_x;
    CubeX LS_grad_y;

protected:
#ifdef USECUDA
    std::array<scalar_t*, 3> DEV_V;
    SimParams* DEV_C;
#else
    macVel V;
    SimParams &C;
#endif

//    scalar_t get_curvature_height(const Vector3 &pos);
    scalar_t get_LS(const Vector3& pos);
    scalar_t get_LS_cerp(const Vector3& pos);
    scalar_t get_curvature_laplace(const Vector3 &pos);
    scalar_t get_curvature_fedkiw(const Vector3 &pos);
    void precalc_fedkiw_curvature();
    virtual scalar_t get_height_normal(Vector3 &base, Vector3 &n, scalar_t h_ref, scalar_t dx);
    scalar_t get_height_normal_ls(Vector3 &base, Vector3 &n, scalar_t h_ref, scalar_t dx);
    scalar_t get_curvature_height_normal(Vector3 &pos);

    int const grid_w;
    int const grid_h;
    int const grid_d;
    scalar_t const dx;
    scalar_t const sim_w;
    scalar_t const sim_h;
    scalar_t const sim_d;
    int const n_cells;

private:
    void redistance_point(unsigned int i, unsigned int j, unsigned int k);
    CubeXi LS_is_interface;
};

#endif //FERRO_SIM_LEVELSET_HPP
