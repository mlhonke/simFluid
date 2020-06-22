//
// Created by graphics on 30/06/19.
//

#ifndef FERRO3D_SIM_ADVECT_CUH
#define FERRO3D_SIM_ADVECT_CUH

#include "sim_params.hpp"
#include "cuvec.cuh"

__host__ __device__ int ccti(int i, int j, int k, int n_rows, int n_cols);
__device__ scalar_t culerp(scalar_t x, scalar_t x1, scalar_t x2, scalar_t Q1, scalar_t Q2);
__device__ scalar_t cubilerp(scalar_t x, scalar_t y, scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2, scalar_t* vals);
__device__ scalar_t cutrilerp(CUVEC::Vec3d x, CUVEC::Vec3d x1, CUVEC::Vec3d x2, scalar_t* vals);
__device__ scalar_t cu_grid_trilerp(const CUVEC::Vec3d &p, const scalar_t *v, int n_rows, int n_cols, int n_slices, scalar_t dx, CUVEC::Vec3d offset = CUVEC::Vec3d(0,0,0));
__device__ scalar_t cu_grid_trilerp(const CUVEC::Vec3d &p, const scalar_t *v, SimParams& C, CUVEC::Vec3d offset = CUVEC::Vec3d(0,0,0));
__device__ scalar_t cu_grid_tricerp(const CUVEC::Vec3d &X, const scalar_t *q, bool clamp, int n_rows, int n_cols, int n_slices, scalar_t dx);
__device__ scalar_t cu_grid_tricerp(const CUVEC::Vec3d &X, const scalar_t *q, bool clamp, SimParams &C);
__device__ CUVEC::Vec3d cu_vel_trilerp(CUVEC::Vec3d p, scalar_t* vx, scalar_t* vy, scalar_t* vz, SimParams &C);
__host__ void advect_RK3_on_device(scalar_t *q,
                                   scalar_t *q_prime,
                                   const scalar_t *u,
                                   const scalar_t *v,
                                   const scalar_t *w,
                                   const CUVEC::Vec3d offset,
                                   scalar_t dt,
                                   bool do_clamp_q,
                                   int n_rows,
                                   int n_cols,
                                   int n_slices,
                                   int n_blocks,
                                   int threads_per_block,
                                   SimParams *C);


#endif //FERRO3D_SIM_ADVECT_CUH
