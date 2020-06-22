#include <iostream>
#include <cuda_runtime.h>

#include "advect.hpp"
#include "interpolate.hpp"
#include "cuda_errorcheck.hpp"
#include "cuvec.cuh"
#include "advect_dev.cuh"

void advect_RK2(CubeX &q, const Vector3 &offset, std::array<CubeX, 3> &V, scalar_t dt, SimParams &C, bool do_clamp, bool do_clamp_q){
    int ni = (int) q.n_rows;
    int nj = (int) q.n_cols;
    int nk = (int) q.n_slices;
    Vector3 X_mid, X_s, X_p;
    CubeX q_temp = q;

    for (int k = 0; k < nk; k++) {
        for (int j = 0; j < nj; j++) {
            for (int i = 0; i < ni; i++) {
                Vector3i I = {i, j, k};
                X_s = C.dx * (I - offset);

                Vector3 vel_start = cerp_velocity(X_s, V, C.dx, do_clamp);
                X_mid = X_s - 0.5 * dt * vel_start;

                Vector3 vel_mid = cerp_velocity(X_mid, V, C.dx, do_clamp);
                X_p = X_s - dt * vel_mid;

                q_temp(i, j, k) = grid_tricerp(X_p + offset * C.dx, q, C.dx, do_clamp_q);
            }
        }
    }

    q = q_temp;
}

void advect_RK3(CubeX &q, const Vector3 &offset, std::array<CubeX, 3> &V, scalar_t dt, SimParams &C, bool do_clamp, bool do_clamp_q){
    Vector3 k1, k2, k3, X_s, X_p;
    CubeX q_temp = q;

    for (unsigned int k = 0; k < q_temp.n_slices; k++) {
        for (unsigned int j = 0; j < q_temp.n_cols; j++) {
            for (unsigned int i = 0; i < q_temp.n_rows; i++) {

                // Initial value at time step n.
                Vector3ui I = {i, j, k};
                X_s = C.dx * (I - offset);

                k1 = cerp_velocity(X_s, V, C.dx, do_clamp);
                k2 = cerp_velocity(X_s - 0.5 * dt * k1, V, C.dx, do_clamp);
                k3 = cerp_velocity(X_s - 0.75 * dt * k2, V, C.dx, do_clamp);

                // Value at time step n+1.
                X_p = X_s - (2.0 / 9.0) * dt * k1 - (3.0 / 9.0) * dt * k2 - (4.0 / 9.0) * dt * k3;

                q_temp(i, j, k) = grid_tricerp(X_p + offset * C.dx, q, C.dx, do_clamp_q);
            }
        }
    }

    q = q_temp;
}

void advect_RK3_CUDA(CubeX &q, const Vector3 &offset, std::array<scalar_t*, 3> DEV_V, scalar_t dt, SimParams* DEV_C, bool do_clamp, bool do_clamp_q){
    scalar_t* DEV_q, *DEV_q_prime;
    cuda_check(cudaMalloc(&DEV_q, q.n_elem*sizeof(scalar_t)));
    cuda_check(cudaMalloc(&DEV_q_prime, q.n_elem*sizeof(scalar_t)));
    cuda_check(cudaMemcpy(DEV_q, q.memptr(), q.n_elem*sizeof(scalar_t), cudaMemcpyHostToDevice));
    int threads_in_block = 256;
    int n_blocks = std::ceil(q.n_elem / (scalar_t) threads_in_block);
    CUVEC::Vec3d cu_offset(offset[0], offset[1], offset[2]);
//    CUVEC::Vec3d *DEV_cu_offset;
//    cuda_check(cudaMalloc(&DEV_cu_offset, sizeof(CUVEC::Vec3d)));
//    cuda_check(cudaMemcpy(DEV_cu_offset, &cu_offset, sizeof(CUVEC::Vec3d), cudaMemcpyHostToDevice));
    advect_RK3_on_device(DEV_q, DEV_q_prime, DEV_V[0], DEV_V[1], DEV_V[2], cu_offset, dt,
            do_clamp_q, q.n_rows, q.n_cols, q.n_slices, n_blocks, threads_in_block, DEV_C);
    cuda_check(cudaPeekAtLastError());
    cuda_check(cudaMemcpy(q.memptr(), DEV_q_prime, q.n_elem*sizeof(scalar_t), cudaMemcpyDeviceToHost));
    cudaFree(DEV_q);
    cudaFree(DEV_q_prime);
//    cudaFree(DEV_cu_offset);
}

Vector3 advect_particle_RK3(const Vector3 &X, std::array<CubeX, 3> &V, scalar_t dt, SimParams &C){
    Vector3 k1 = dt * cerp_velocity( X, V, C.dx);
    Vector3 k2 = dt * cerp_velocity( X + 0.5f * k1, V, C.dx);
    Vector3 k3 = dt * cerp_velocity( X - k1 + 3.0f * k2, V, C.dx);
    Vector3 retval = X + 1.0f/6.0f * ( k1 + 4.0f * k2 + k3 );

    return retval;
}

// Helper functions that encode the grid offsets for each velocity component when interpolating.
Vector3 cerp_velocity(const Vector3 &X, const macVel &V, const scalar_t dx, bool set_bounds){
    Vector3 R = {0.0, 0.0, 0.0};
    Vector3 offset = {0.5, 0.0, 0.0};

    R(0) = grid_tricerp(X+offset*dx, V[0], dx);
    offset = {0.0, 0.5, 0.0};
    R(1) = grid_tricerp(X+offset*dx, V[1], dx);
    offset = {0.0, 0.0, 0.5};
    R(2) = grid_tricerp(X+offset*dx, V[2], dx);

    return R;
}

// Helper functions that encode the grid offsets for each velocity component.
Vector3 lerp_velocity(const Vector3 &X, const macVel &V, const scalar_t dx, bool set_bounds){
    Vector3 R = {0.0, 0.0, 0.0};
    Vector3 offset = {0.5, 0.0, 0.0};
    R(0) = grid_trilerp(X+offset*dx, V[0], dx);
    offset = {0.0, 0.5, 0.0};
    R(1) = grid_trilerp(X+offset*dx, V[1], dx);
    offset = {0.0, 0.0, 0.5};
    R(2) = grid_trilerp(X+offset*dx, V[2], dx);

    return R;
}
