#ifndef SIM_PARAMS_HPP
#define SIM_PARAMS_HPP

#include "sim_types.hpp"

typedef struct SimParams {
    SimParams(){}

    SimParams(int grid_w, int grid_h, int grid_d, scalar_t dx, scalar_t frac_cfl, scalar_t max_dt, scalar_t render_dt) :
    grid_w(grid_w), grid_h(grid_h), grid_d(grid_d), dx(dx), frac_cfl(frac_cfl), max_dt(max_dt), render_dt(render_dt)
    {}

    int const grid_w = 60;
    int const grid_h = 60;
    int const grid_d = 60;
    scalar_t const dx = 1.0/60.0;
    scalar_t const frac_cfl = 0.1;
    scalar_t const max_dt = 0.1;
    scalar_t const render_dt = 0.1;

    // Calculated
    scalar_t const sim_w = dx*(grid_w-1);
    scalar_t const sim_h = dx*(grid_h-1);
    scalar_t const sim_d = dx*(grid_d-1);
    int const n_cells = grid_w*grid_h*grid_d;
} SimParams;

SimParams* copy_params_to_device(SimParams &params);

#endif
