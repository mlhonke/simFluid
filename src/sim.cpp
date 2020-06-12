#include <cmath>
#include "sim.hpp"
#include "interpolate.hpp"
#include "eigen_types.hpp"
#include "unit.cuh"
#include <cuda_runtime.h>
#include "cuda_errorcheck.hpp"
#include "sim_label.hpp"

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
    n_cells(C.n_cells)
    {
//    plot = new DisplayWindow(*this);
    omp_set_num_threads(8);
    std::cout << "Number of threads available for Eigen to use: " << Eigen::nbThreads() << std::endl;

    std::cout << "Running a simulation in a " << sim_w << " X " << sim_h << " X " << sim_d << " volume." << std::endl;

    // Setup the 3 cubes of velocity values (u, v, w). Rely on copy of each object.
    V[0] = (CubeX(grid_w+1, grid_h, grid_d, arma::fill::zeros));
    V[1] = (CubeX(grid_w, grid_h+1, grid_d, arma::fill::zeros));
    V[2] = (CubeX(grid_w, grid_h, grid_d+1, arma::fill::zeros));

    V_solid[0] = (CubeX(grid_w+1, grid_h, grid_d, arma::fill::zeros));
    V_solid[1] = (CubeX(grid_w, grid_h+1, grid_d, arma::fill::zeros));
    V_solid[2] = (CubeX(grid_w, grid_h, grid_d+1, arma::fill::zeros));

    // Setup device parameter structure.
    cuda_check(cudaMalloc(&DEV_C, sizeof(SimParams)))
    cuda_check(cudaMemcpy(DEV_C, &C, sizeof(SimParams), cudaMemcpyHostToDevice))

    // Allocating space for velocities on device.
    cuda_check(cudaMalloc(&DEV_V[0], n_vel_x*sizeof(scalar_t)))
    cuda_check(cudaMalloc(&DEV_V[1], n_vel_y*sizeof(scalar_t)))
    cuda_check(cudaMalloc(&DEV_V[2], n_vel_z*sizeof(scalar_t)))

    fluid_label = new SimLabel(*this, false);
//    test_cuda_interpolation();
//    test_cuda_RK3_advection(*this);

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

void Sim::update_velocities_on_host(){
    cuda_check(cudaMemcpy(V[0].memptr(), DEV_V[0], n_vel_x*sizeof(scalar_t), cudaMemcpyDeviceToHost))
    cuda_check(cudaMemcpy(V[1].memptr(), DEV_V[1], n_vel_y*sizeof(scalar_t), cudaMemcpyDeviceToHost))
    cuda_check(cudaMemcpy(V[2].memptr(), DEV_V[2], n_vel_z*sizeof(scalar_t), cudaMemcpyDeviceToHost))
}

void Sim::update_velocities_on_device(){
    cuda_check(cudaMemcpy(DEV_V[0], V[0].memptr(), n_vel_x*sizeof(scalar_t), cudaMemcpyHostToDevice))
    cuda_check(cudaMemcpy(DEV_V[1], V[1].memptr(), n_vel_y*sizeof(scalar_t), cudaMemcpyHostToDevice))
    cuda_check(cudaMemcpy(DEV_V[2], V[2].memptr(), n_vel_z*sizeof(scalar_t), cudaMemcpyHostToDevice))
}

void Sim::free_velocities_on_device(){
    cuda_check(cudaFree(DEV_V[0]));
    cuda_check(cudaFree(DEV_V[1]));
    cuda_check(cudaFree(DEV_V[2]));
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
//    max_vel += std::sqrt(scale_w*std::abs(g));

    if (cur_step == 0){
        return 0.1;
    }

    scalar_t new_dt;
    if ((std::abs(scale_w / max_vel) < max_dt)) {
        new_dt = std::abs(scale_w/max_vel);
    } else {
        new_dt = max_dt;
    }

    return new_dt;
}

Vector3ui Sim::convert_index_to_coords(unsigned int d, unsigned int rows, unsigned int cols){
    unsigned int i = d % rows;
    unsigned int j = (d / rows) % cols;
    unsigned int k = d / (cols * rows);

    return {i, j, k};
}

bool Sim::run_unit_tests() {
    return true;
}
