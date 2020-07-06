//
// Created by graphics on 2020-07-03.
//

#include <cstdlib>
#include <cuda_runtime.h>

#include "doctest.hpp"
#include "test_tools.hpp"
#include "sim_utils.hpp"
#include "sim_pls_cuda_dev.cuh"

CUVEC::Vec3d rand_particle_position(const Vector3 &min, const Vector3 &max){
    Vector3 randVec(arma::fill::randu);
    randVec = min + randVec % (max-min); // % is element wise multiplication for armadillo

    return {randVec[0], randVec[1], randVec[2]};
}

TEST_CASE("Advecting particles on device."){
    // Particle test parameters.
    const int threads_in_block = 256;
    const int n_particles = 1024;
    const int n_blocks = n_particles / threads_in_block;

    // Setup a basic simulation.
    SimParams *C;
    SimParams *DEV_C;
    macVel V;
    DEV_macVel DEV_V;
    scalar_t dt;
    int dim = 5;
    init_basic_cube_CUDA_simulation(dim, C, DEV_C, V, DEV_V, dt);
    V[0].fill(1);
    V[1].fill(1);
    V[2].fill(1);
    update_mac_velocities_on_device(V, DEV_V);
    dt = 1.0 / dim;

    // Generate test particles.
    std::array<CUVEC::Vec3d, n_particles> p;
    for (int i = 0; i < n_particles; i++){
        p[i] = rand_particle_position({0, 0, 0}, {C->sim_w, C->sim_h, C->sim_d});
    }
    CUVEC::Vec3d *DEV_p;
    cudaMalloc(&DEV_p, p.size()*sizeof(CUVEC::Vec3d));
    cudaMemcpy(DEV_p, &p[0], p.size()*sizeof(CUVEC::Vec3d), cudaMemcpyHostToDevice);
    advect_particles_on_device(n_blocks, threads_in_block, n_particles, dt, DEV_p, DEV_V, DEV_C);
    std::array<CUVEC::Vec3d, n_particles> post_p;
    cudaMemcpy(&post_p[0], DEV_p, p.size()*sizeof(CUVEC::Vec3d), cudaMemcpyDeviceToHost);

    // Check advection of particles.
    for (int i = 0; i < n_particles; i++){
        scalar_t diff = CUVEC::mag(p[i] - post_p[i]);
        CHECK(almost_equal(diff, std::sqrt(dt*dt + dt*dt + dt*dt), 1e-14));
    }
}