//
// Created by graphics on 12/01/19.
//

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "advect_dev.cuh"
#include "sim_pls_cuda_dev.cuh"
#include "cuda_errorcheck.hpp"

__global__ void
advect_particles_3D(int n_particles, scalar_t dt, CUVEC::Vec3d *p, scalar_t *vx, scalar_t *vy, scalar_t *vz, SimParams *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles) { // Will cause some thread divergence in the last warp.
        CUVEC::Vec3d vel_start = cu_vel_trilerp(p[i], vx, vy, vz, *C);
        CUVEC::Vec3d pos_mid = p[i] + 0.5 * dt * vel_start;
        CUVEC::Vec3d vel_mid = cu_vel_trilerp(pos_mid, vx, vy, vz, *C);
        p[i] += dt * vel_mid;
//        CUVEC::Vec3d min_coord(0, 0, 0);
//        CUVEC::Vec3d max_coord(C->sim_w, C->sim_h, C->sim_d);
//        p[i] = CUVEC::clamp(p[i], min_coord, max_coord);
    }
}

__host__ void
advect_particles_on_device(int n_blocks, int threads_in_block, int n_particles, scalar_t dt, CUVEC::Vec3d *DEV_p,
                           std::array<scalar_t*, 3> DEV_V, SimParams *C) {
    advect_particles_3D <<< n_blocks, threads_in_block >>> (n_particles, dt, DEV_p, DEV_V[0], DEV_V[1], DEV_V[2], C);
    cuda_check(cudaPeekAtLastError());
}

__device__ void get_neighbours_cerped_pos(const CUVEC::Vec3d &pos, scalar_t *Q, scalar_t *sides, SimParams &C) {
    bool clamp = true;
    sides[0] = cu_grid_tricerp({pos[0] + C.dx, pos[1], pos[2]}, Q, clamp, C);
    sides[1] = cu_grid_tricerp({pos[0] - C.dx, pos[1], pos[2]}, Q, clamp, C);
    sides[2] = cu_grid_tricerp({pos[0], pos[1] + C.dx, pos[2]}, Q, clamp, C);
    sides[3] = cu_grid_tricerp({pos[0], pos[1] - C.dx, pos[2]}, Q, clamp, C);
    sides[4] = cu_grid_tricerp({pos[0], pos[1], pos[2] + C.dx}, Q, clamp, C);
    sides[5] = cu_grid_tricerp({pos[0], pos[1], pos[2] - C.dx}, Q, clamp, C);
}

__device__ void get_neighbours_lerped_pos(const CUVEC::Vec3d &pos, scalar_t *Q, scalar_t *sides, SimParams &C) {
    sides[0] = cu_grid_trilerp({pos[0] + C.dx, pos[1], pos[2]}, Q, C, {0, 0, 0});
    sides[1] = cu_grid_trilerp({pos[0] - C.dx, pos[1], pos[2]}, Q, C, {0, 0, 0});
    sides[2] = cu_grid_trilerp({pos[0], pos[1] + C.dx, pos[2]}, Q, C, {0, 0, 0});
    sides[3] = cu_grid_trilerp({pos[0], pos[1] - C.dx, pos[2]}, Q, C, {0, 0, 0});
    sides[4] = cu_grid_trilerp({pos[0], pos[1], pos[2] + C.dx}, Q, C, {0, 0, 0});
    sides[5] = cu_grid_trilerp({pos[0], pos[1], pos[2] - C.dx}, Q, C, {0, 0, 0});
}

__device__ CUVEC::Vec3d
cu_get_grad_lerped_pos(const CUVEC::Vec3d &pos, scalar_t *LS, SimParams &C) {
    CUVEC::Vec3d grad;
    scalar_t sides[6];
    get_neighbours_lerped_pos(pos, LS, sides, C);

    grad[0] = (sides[0] - sides[1]) / (2.0 * C.dx);
    grad[1] = (sides[2] - sides[3]) / (2.0 * C.dx);
    grad[2] = (sides[4] - sides[5]) / (2.0 * C.dx);

    return grad;
}

__device__ CUVEC::Vec3d
cu_get_grad_lerped(const CUVEC::Vec3d &pos, scalar_t *LS_grad_x, scalar_t *LS_grad_y, scalar_t *LS_grad_z,
                   SimParams &C) {
    CUVEC::Vec3d grad;
    grad[0] = cu_grid_trilerp(pos, LS_grad_x, C);
    grad[1] = cu_grid_trilerp(pos, LS_grad_y, C);
    grad[2] = cu_grid_trilerp(pos, LS_grad_z, C);

    return grad;
}

__device__ CUVEC::Vec3d
get_normal(const CUVEC::Vec3d &pos, scalar_t *LS, scalar_t *LS_grad_x, scalar_t *LS_grad_y, scalar_t *LS_grad_z,
           SimParams &C) {
    CUVEC::Vec3d grad;

//    grad = cu_get_grad_lerped(pos, LS_grad_x, LS_grad_y, LS_grad_z, scale_w);
    grad = cu_get_grad_lerped_pos(pos, LS, C);
    scalar_t magnitude = CUVEC::mag(grad);
    if (magnitude > 1e-10) {
        CUVEC::normalize(grad);
        return grad;
    } else {
        return {0, 0, 0};
    }
}

__device__ bool
get_surface_point(const CUVEC::Vec3d &pos, CUVEC::Vec3d &result, scalar_t *LS, scalar_t *LS_grad_x, scalar_t *LS_grad_y,
                  scalar_t *LS_grad_z, SimParams &C) {
    CUVEC::Vec3d search_pt = pos;
    bool clamp = false;
//    scalar_t dist = cu_grid_tricerp(search_pt, LS, clamp, C);
    scalar_t dist = cu_grid_trilerp(search_pt, LS, C);
    scalar_t tol = 1E-12;
    int iters = 0;
    while (fabs(dist) > tol * C.dx && iters < 300) {
        CUVEC::Vec3d normal = get_normal(search_pt, LS, LS_grad_x, LS_grad_y, LS_grad_z, C);
        search_pt -= dist * normal;
        dist = cu_grid_tricerp(search_pt, LS, clamp, C);
//        dist = cu_grid_trilerp(search_pt, LS, C);
        ++iters;
    }
    result = search_pt;
    bool valid = fabs(dist) <= tol * C.dx;
    return valid;
}

__device__ inline unsigned int randhash(unsigned int seed) {
    unsigned int i = (seed ^ 0xA3C59AC3u) * 2654435769u;
    i ^= (i >> 16);
    i *= 2654435769u;
    i ^= (i >> 16);
    i *= 2654435769u;
    return i;
}

__device__ inline float randhashf(unsigned int seed, float a, float b) {
    return ((b - a) * randhash(seed) / (float) UINT_MAX + a);
}

__constant__ scalar_t dir[2] = {-1, 1}; // avoid thread divergence
__global__ void reseed_particles_surface(int n_cells,
                                         CUVEC::Vec3i *coords,
                                         CUVEC::Vec3d *sp,
                                         CUVEC::Vec3d *pp,
                                         CUVEC::Vec3d *np,
                                         scalar_t *LS,
                                         scalar_t *LS_grad_x,
                                         scalar_t *LS_grad_y,
                                         scalar_t *LS_grad_z,
                                         int sp_per_cell,
                                         int sign_per_cell,
                                         scalar_t bandwidth,
                                         SimParams *C
                                        ) {
    (void) sign_per_cell; // Not used because we assign a signed particle with a surface particle each time.
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cell_id = id / sp_per_cell;

    if (cell_id < n_cells) {
        unsigned int i = coords[cell_id][0];
        unsigned int j = coords[cell_id][1];
        unsigned int k = coords[cell_id][2];

        unsigned int seed = 3 * id;
        CUVEC::Vec3d start(i * C->dx, j * C->dx, k * C->dx);
        scalar_t x_off = C->dx * randhashf(seed++, 0, 1) - 0.5 * C->dx;
        scalar_t y_off = C->dx * randhashf(seed++, 0, 1) - 0.5 * C->dx;
        scalar_t z_off = C->dx * randhashf(seed++, 0, 1) - 0.5 * C->dx;
        CUVEC::Vec3d offset(x_off, y_off, z_off);
        CUVEC::Vec3d newpt = start + offset;
        CUVEC::Vec3d min_coord(0, 0, 0);
        CUVEC::Vec3d max_coord(C->sim_w, C->sim_h, C->sim_d);
        newpt = CUVEC::clamp(newpt, min_coord, max_coord);
//    scalar_t LS_val = cu_grid_trilerp(newpt, LS, *C);
        unsigned int p_id = id;
//    if (fabs(LS_val) < 2.0*bandwidth * C->dx) {
//        if (LS_val > 0) {
//            pp[p_id] = newpt;
//            np[p_id] = CUVEC::Vec3d(-1000, -1000, -1000);
//        } else {
//            np[p_id] = newpt;
//            pp[p_id] = CUVEC::Vec3d(-1000, -1000, -1000);
//        }
        CUVEC::Vec3d surf;
        bool success = get_surface_point(newpt, surf, LS, LS_grad_x, LS_grad_y, LS_grad_z, *C);
//        if (success && (surf[0] < 0 || surf[1] < 0 || surf[2] < 0)){
//            printf("Arrived at a negative value, oops\n");
//        }
        if (success) {
//            surf[0] = fmax(fmin(surf[0], C->sim_w), 0);
//            surf[1] = fmax(fmin(surf[1], C->sim_h), 0);
//            surf[2] = fmax(fmin(surf[2], C->sim_d), 0);
            sp[p_id] = surf;
            // half of signed particles in each direction
            CUVEC::Vec3d p0 =
                    surf + dir[p_id % 2] * 0.25 * C->dx * get_normal(surf, LS, LS_grad_x, LS_grad_y, LS_grad_z, *C);
//            CUVEC::Vec3d p1 = surf - 0.25*C->dx*get_normal(surf, LS, LS_grad_x, LS_grad_y, LS_grad_z, *C);
            np[p_id] = CUVEC::Vec3d(-1000, -1000, -1000);
            pp[p_id] = CUVEC::Vec3d(-1000, -1000, -1000);
            if (cu_grid_trilerp(p0, LS, *C) > 0) {
                pp[p_id] = p0;
            } else {
                np[p_id] = p0;
            }

//            if ( cu_grid_trilerp(p1, LS, *C) > 0 ){
//                pp[p_id] = p1;
//            } else {
//                np[p_id] = p1;
//            }
        } else {
//            printf("Failed to seed a valid start position!\n");
            sp[p_id] = CUVEC::Vec3d(-1000, -1000, -1000);
            np[p_id] = CUVEC::Vec3d(-1000, -1000, -1000);
            pp[p_id] = CUVEC::Vec3d(-1000, -1000, -1000);
        }
//    } else {
//        sp[p_id] = CUVEC::Vec3d(-1000, -1000, -1000);
//        pp[p_id] = CUVEC::Vec3d(-1000, -1000, -1000);
//        np[p_id] = CUVEC::Vec3d(-1000, -1000, -1000);
//    }
    }
}

__host__ void reseed_surface_particles_on_device(int n_cells,
                                                 CUVEC::Vec3i *coords,
                                                 CUVEC::Vec3d *sp,
                                                 CUVEC::Vec3d *pp,
                                                 CUVEC::Vec3d *np,
                                                 scalar_t *LS,
                                                 scalar_t *LS_grad_x,
                                                 scalar_t *LS_grad_y,
                                                 scalar_t *LS_grad_z,
                                                 int sp_per_cell,
                                                 int sign_per_cell,
                                                 scalar_t bandwidth,
                                                 SimParams *C,
                                                 int threads_in_block) {
    int n_blocks = (n_cells*sp_per_cell)/threads_in_block + 1;
    reseed_particles_surface <<< n_blocks, threads_in_block >>>
                                           (n_cells, coords, sp, pp, np, LS, LS_grad_x, LS_grad_y, LS_grad_z,
                                                   sp_per_cell, sign_per_cell, bandwidth, C);
}

__global__ void calculate_particle_cells(int n_p, CUVEC::Vec3d *p, int *cell_ids, SimParams *C) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n_p) {
        int i = lrint(p[id][0] / C->dx);
        int j = lrint(p[id][1] / C->dx);
        int k = lrint(p[id][2] / C->dx);

//        if (i < 0 || j < 0) {
//            printf("%d, %d, %d, %f, %f, %f\n", i, j, k, p[id][0], p[id][1], p[id][2]);
//        }
        i = max(min(i, C->grid_w-2), 1);
        j = max(min(j, C->grid_h-2), 1);
        k = max(min(k, C->grid_d-2), 1);
        int cell_id = ccti(i, j, k, C->grid_w, C->grid_h);

//        printf("%d \n", cell_id);
        cell_ids[id] = cell_id;
    }
}

__host__ void
calculate_particle_cells_on_device(int n_blocks, int threads_in_block, int n_p, CUVEC::Vec3d *p, int *cell_ids,
                                   SimParams *C) {
    calculate_particle_cells <<< n_blocks, threads_in_block >>> (n_p, p, cell_ids, C);
    cuda_check(cudaPeekAtLastError());
}

__host__ void sort_particles_by_key_on_device(CUVEC::Vec3d *p, int *cell_ids, int n_p) {
    // GPU implementation
    thrust::device_ptr<CUVEC::Vec3d> vals(p);
    thrust::device_ptr<int> keys(cell_ids);
    thrust::sort_by_key(keys, keys + n_p, vals);
}

__device__ scalar_t
find_closest_particle(CUVEC::Vec3d *p, int n_p, CUVEC::Vec3d grid_pos, int *cp){
    scalar_t best_dist = 2000;
    for (int i = 0; i<n_p; i++){
        CUVEC::Vec3d sep = p[i] - grid_pos;
        scalar_t dist = CUVEC::mag(sep);
        if (dist<best_dist){
            best_dist = dist;
            *cp = i;
        }
    }

    return best_dist;
}

__device__ void do_grid_redistance_now(scalar_t * LS, CUVEC::Vec3d * p, int * count, CUVEC::Vec3d grid_pos, int p_id, int id, int *cp){
    int n_p = count[id];
    int candidate_cp;
    scalar_t dist = find_closest_particle(&p[p_id], n_p, grid_pos, &candidate_cp);
    if (dist < *LS){
        *LS = dist;
        *cp = candidate_cp + p_id;
    }
}

__device__ int cusgn(scalar_t val) {
    return (val > 0) - (val < 0);
}

// TODO: Improve performance by using the levelset to determine if its worth even looking for a particle nearby.
__global__ void
assign_grid_particle_dist(int n_cells, scalar_t *LS, CUVEC::Vec3d *p, int *index, int *count, int *cp, SimParams *C) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; // grid index
    int i = id % C->grid_w;
    int j = (id / C->grid_w) % C->grid_h;
    int k = id / (C->grid_h * C->grid_w);
    CUVEC::Vec3d grid_pos(i * C->dx, j * C->dx, k * C->dx);
//    printf("Processing cell %d, %d, %d\n", i, j, k);
    if (id < n_cells) {
        LS[id] = 1000;
        int p_id = index[id];
        if (p_id == -1) {
            for (int kn = k - 1; kn <= k + 1; kn++) {
                for (int jn = j - 1; jn <= j + 1; jn++) {
                    for (int in = i - 1; in <= i + 1; in++) {
                        if (kn >= 0 && kn < C->grid_d && jn >= 0 && jn < C->grid_h && in >= 0 && in < C->grid_w) {
                            int id_adj = ccti(in, jn, kn, C->grid_w, C->grid_h);
                            p_id = index[id_adj];
                            if (p_id != -1) {
//                                printf("Redistancing with other cell's particles.\n");
                                do_grid_redistance_now(&LS[id], p, count, grid_pos, p_id, id_adj, &cp[id]);
                            }
                        }
                    }
                }
            }
        } else {
//            printf("Redistancing with own cell's particles.\n");
            do_grid_redistance_now(&LS[id], p, count, grid_pos, p_id, id, &cp[id]);
        }
    }
}

__host__ void update_levelset_distances_on_device(int n_cells, int n_blocks, int threads_per_block,
                                                  scalar_t *LS, CUVEC::Vec3d *p, int *index, int *count, int *cp, SimParams* C) {
    assign_grid_particle_dist <<< n_blocks, threads_per_block >>> (n_cells, LS, p, index, count, cp, C);
    cuda_check(cudaPeekAtLastError());
}

__global__ void update_LS_signs(int n_cells, scalar_t *LS, scalar_t *LS_p, scalar_t *LS_n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n_cells) {
        if (LS_p[id] > LS_n[id]) {
            LS[id] = fabs(LS[id]);
        } else {
            LS[id] = -fabs(LS[id]);
        }
    }
}

//__global__ void generate_curvature(int n_cells, CUVEC::Vec3d *p, int* index, int* count, scalar_t* curv){
//    int id = blockIC.dx.x*blockDim.x + threadIC.dx.x;
//    int i = id % C.grid_w;
//    int j = (id / C.grid_w) % C.grid_h;
//    int k = id / (C.grid_w * C.C.grid_h);
//    CUVEC::Vec3d pos(i*scale_w, j*scale_h, k*scale_d); // cell position
//
//    if (id < n_cells){
//        for(int i = 0; i < count[id]; i++){
//            int p_id = index[id];
//            if (p_id != -1){
//
//            }
//        }
//    }
//}
//
//__host__ void generate_curvature_on_device(int n_cells, int n_blocks, int threads_per_block, CUVEC::Vec3d *p, int* index, int* count, scalar_t* curv){
//
//}