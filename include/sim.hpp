#ifndef SIM_HPP
#define SIM_HPP

#include <armadillo>
#include <set>
#include <fstream>

#include "sim_params.hpp"
#include "sim_external_types.hpp"


class DisplayWindow;
class SimLabel;

class Sim {
public:
    Sim(SimParams C);
    bool is_coord_valid(Vector3i I);
    Vector3ui convert_index_to_coords(unsigned int d, unsigned int rows, unsigned int cols);
    void update_velocities_on_host();
    void update_velocities_on_device();
    void free_velocities_on_device();
    virtual void save_data();
    virtual void load_data();

    // Test cases
    bool test_cuda_interpolation();

    scalar_t max_dt = 0.1;
    scalar_t t = 0;
    scalar_t dt = 0.0;
    scalar_t render_dt = 0.02;
    scalar_t render_time = render_dt;
    bool do_render = false;
    macVel V;
    macVel V_solid;
    // We keep track of velocities as individual arrays since MAC grid
    std::array<scalar_t*, 3> DEV_V;
    SimParams *DEV_C;
    int cur_step = 0;
    SimLabel *fluid_label;

    std::ofstream logfile;

    // SIMULATION PARAMETERS (with default values set)

    //Simulation constants
    scalar_t const scale_w = 1.0/320.0;
    scalar_t const scale_h = scale_w;
    scalar_t const scale_d = scale_w;
    int const grid_w = 80;
    int const grid_h = 80;
    int const grid_d = 40;
    scalar_t const sim_w = scale_w*(grid_w-1);
    scalar_t const sim_h = scale_h*(grid_h-1);
    scalar_t const sim_d = scale_d*(grid_d-1);
    int const n_cells = grid_w*grid_h*grid_d;
    int const n_vel_x = (grid_w+1)*grid_h*grid_d;
    int const n_vel_y = grid_w*(grid_h+1)*grid_d;
    int const n_vel_z = grid_w*grid_h*(grid_d+1);

protected:
    scalar_t get_max_vel(const macVel &V_in);
    scalar_t get_new_timestep(const macVel &V_in);

    DisplayWindow* plot;
    bool run_unit_tests();
    scalar_t elapsed_time = 0;
};

#endif
