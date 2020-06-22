#include <cmath>
#include "sim.hpp"
#include "interpolate.hpp"
#include "eigen_types.hpp"
#include "unit.cuh"
#include "sim_label.hpp"
#include "sim_utils.hpp"

Sim::Sim(SimParams C)
    :
    scale_w(C.dx),
    scale_h(C.dx),
    scale_d(C.dx),
    grid_d(C.grid_d),
    grid_h(C.grid_h),
    grid_w(C.grid_w),
    sim_w(C.sim_w),
    sim_h(C.sim_h),
    sim_d(C.sim_d),
    n_cells(C.n_cells),
    frac_cfl(C.frac_cfl),
    max_dt(C.max_dt),
    render_dt(C.render_dt)
    {
//    plot = new DisplayWindow(*this);
    omp_set_num_threads(8);
    std::cout << "Number of threads available for Eigen to use: " << Eigen::nbThreads() << std::endl;
    std::cout << "Running a simulation in a " << sim_w << " X " << sim_h << " X " << sim_d << " volume." << std::endl;

    init_mac_velocities(C, V);
    init_mac_velocities(C, V_solid);
    init_cuda_mac_velocities(V, DEV_V);
    DEV_C = copy_params_to_device(C);
    fluid_label = new SimLabel(*this, false);
    logfile.open("logfile.txt");
}

void Sim::save_data(){
    std::cout << "Saving simulation data." << std::endl;

    V[0].save("V0.bin");
    V[1].save("V1.bin");
    V[2].save("V2.bin");
    fluid_label->save_data("label.bin");
}

void Sim::load_data(){
    V[0].load("V0.bin");
    V[1].load("V1.bin");
    V[2].load("V2.bin");
    update_velocities_on_device();
    fluid_label->load_data("label.bin");
}

bool Sim::is_coord_valid(Vector3i I){
    return (I(0) >= 0 && I(0) < grid_w && I(1) >= 0 && I(1) < grid_h && I(2) >= 0 && I(2) < grid_d);
}

scalar_t Sim::get_max_vel(const macVel &V_in){
    scalar_t max_vel = 0;
    for (unsigned int k = 0; k < grid_d; k++) {
        for (unsigned int j = 0; j < grid_h; j++) {
            for (unsigned int i = 0; i < grid_w+1; i++) {
                max_vel = std::max(max_vel, std::abs(V_in[0](i,j,k)));
            }
        }
    }

    for (unsigned int k = 0; k < grid_d; k++) {
        for (unsigned int j = 0; j < grid_h+1; j++) {
            for (unsigned int i = 0; i < grid_w; i++) {
                max_vel = std::max(max_vel, std::abs(V_in[1](i,j,k)));
            }
        }
    }

    for (unsigned int k = 0; k < grid_d+1; k++) {
        for (unsigned int j = 0; j < grid_h; j++) {
            for (unsigned int i = 0; i < grid_w; i++) {
                max_vel = std::max(max_vel, std::abs(V_in[2](i,j,k)));
            }
        }
    }

    return max_vel;
}

scalar_t Sim::get_new_timestep(const macVel &V_in){
    scalar_t max_vel = get_max_vel(V_in);
    scalar_t new_dt = frac_cfl*std::abs(scale_w / max_vel);

    if (new_dt > max_dt || cur_step == 0) {
        return max_dt;
    } else {
        return new_dt;
    }
}

Vector3ui Sim::convert_index_to_coords(unsigned int d, unsigned int rows, unsigned int cols){
    unsigned int i = d % rows;
    unsigned int j = (d / rows) % cols;
    unsigned int k = d / (cols * rows);

    return {i, j, k};
}

void Sim::update_velocities_on_host() {
    update_mac_velocities_on_host(V, DEV_V);
}

void Sim::update_velocities_on_device() {
    update_mac_velocities_on_device(V, DEV_V);
}

void Sim::free_velocities_on_device() {
    free_mac_velocities_on_device(DEV_V);
}
