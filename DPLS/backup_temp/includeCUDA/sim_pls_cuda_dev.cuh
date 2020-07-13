#ifndef SIM_PLS_CUDA_DEV_CUH
#define SIM_PLS_CUDA_DEV_CUH

#include <array>
#include "sim_params.hpp"
#include "cuvec.cuh"

void sort_particles_by_key_on_device(CUVEC::Vec3d *p, int *cell_ids, int n_p);
void calculate_particle_cells_on_device(int n_blocks, int threads_in_block, int n_p, CUVEC::Vec3d *p, int *cell_ids, SimParams *C);
void reseed_surface_particles_on_device(int n_cells, CUVEC::Vec3i *coords, CUVEC::Vec3d *sp, scalar_t* LS, scalar_t *LS_grad_x, scalar_t *LS_grad_y, scalar_t *LS_grad_z, int sp_per_cell, scalar_t bandwidth, SimParams *C, int threads_in_block);
void reseed_sign_particles_on_device(int n_cells, CUVEC::Vec3d *sp, CUVEC::Vec3d *pp, CUVEC::Vec3d *np, scalar_t *LS, scalar_t *LS_grad_x, scalar_t *LS_grad_y, scalar_t *LS_grad_z, int sp_per_cell, int sign_per_cell, scalar_t bandwidth, SimParams *C, int threads_in_block);
void advect_particles_on_device(int n_blocks, int threads_in_block, int n_particles, scalar_t dt, CUVEC::Vec3d* DEV_p, std::array<scalar_t*, 3> DEV_V, SimParams *C);
void update_levelset_distances_on_device(int n_cells, int n_blocks, int threads_per_block, scalar_t* LS, CUVEC::Vec3d *p, int* index, int* count, int* cp, SimParams* C);

#endif //SIM_PLS_CUDA_DEV_CUH