//
// Created by graphics on 19/07/19.
//

#include "unit.cuh"
#include <armadillo>
#include "sim_params.hpp"
#include "advect_dev.cuh"
#include "interpolate.hpp"
#include <cuda_runtime.h>
#include "cuda_errorcheck.hpp"

//bool test_cuda_interpolation(){
//    CubeX lerp_this(10, 10, 10, arma::fill::zeros);
//    lerp_this(4, 4, 4) = 1;
//    lerp_this(4, 5, 4) = 2;
//    lerp_this(5, 4, 4) = 2;
//    lerp_this(5, 5, 4) = 3;
//    lerp_this(4, 4, 5) = 2;
//    lerp_this(4, 5, 5) = 3;
//    lerp_this(5, 4, 5) = 3;
//    lerp_this(5, 5, 5) = 4;
//    // mid point between values should be 2.5;
//    scalar_t dx = 0.1;
//    CUVEC::Vec3d pos = {4.5*dx, 4.5*dx, 4.5*dx};
//    scalar_t cpu_lerp = grid_trilerp({4.5*dx, 4.5*dx, 4.5*dx}, lerp_this, dx);
//    std::cout << "CPU lerp'd value was " << cpu_lerp << std::endl;
//
//    scalar_t *DEV_lerp_this;
//    cudaMalloc(&DEV_lerp_this, lerp_this.n_elem*sizeof(scalar_t));
//    cudaMemcpy(DEV_lerp_this, lerp_this.memptr(), lerp_this.n_elem*sizeof(scalar_t), cudaMemcpyHostToDevice);
//    test_cuda_lerp_on_device(pos, DEV_lerp_this);
//    cudaFree(DEV_lerp_this);
//
//    return true;
//}

//bool test_cuda_RK3_advection(Sim &sim){
//    Vector3 offset = {0.5, 0, 0};
//    CubeX q(sim.grid_w, sim.grid_h, sim.grid_d, arma::fill::zeros);
//    q(5,5,5) = 1;
//    CubeX q2 = q;
//    std::cout << q.slice(5) << std::endl;
//    sim.dt = 0.05;
//    sim.V[0].fill(1);
//    sim.V[0](5, 5, 5) = 2;
//    sim.V[1].fill(0);
//    sim.V[2].fill(0);
//    sim.fluid_label->label.fill(0);
//    sim.fluid_label->update_label_on_device();
//    sim.update_velocities_on_device();
//    CubeX qcopy = q;
//    advect_RK3(qcopy, {0,0.5,0}, sim, false, false, true);
//    std::cout << qcopy.slice(5) << std::endl;
//    advect_RK3_CUDA(q, {0, 0.5, 0}, sim, false, false, true);
//    std::cout << q.slice(5) << std::endl;
//    CubeX Vcopy = sim.V[0];
//    advect_RK3(Vcopy, offset, sim, true, true, true);
//    std::cout << Vcopy.slice(5) << std::endl;
//    advect_RK3_CUDA(sim.V[0], offset, sim, true, true, true);
//    std::cout << sim.V[0].slice(5) << std::endl;
//    return true;
//}