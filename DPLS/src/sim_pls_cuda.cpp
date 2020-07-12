////
//// Created by graphics on 02/07/19.
////

#include <cmath>
#include <map>
#include <algorithm>
#include <cuda_runtime.h>

#include "advect.hpp"
#include "sim_pls_cuda.hpp"
#include "sim_pls_cuda_dev.cuh"
#include "interpolate.hpp"
#include "advect_dev.cuh"
#include "cuda_errorcheck.hpp"
#include "sim_utils.hpp"
#include "fd_math.hpp"
#include "execTimer.hpp"

#ifdef USECUDA
#define VERBOSE_PLS // Enable outputs explaining PLS updates.

SimPLSCUDA::SimPLSCUDA(SimParams &C, SimParams* DEV_C, std::array<scalar_t*, 3> &DEV_V) : SimLevelSet(C, DEV_C, DEV_V) {
    bandwidth = 1.0;
    reseed_interval = 5;
    iteration = 1;
    surface_particles_per_cell = 100;
    sign_particles_per_cell = 50;
    ni = grid_w;
    nj = grid_h;
    nk = grid_d;
    threads_in_block = 256;

    LS_pos = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
    LS_neg = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
    LS_unsigned = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
    cp = CubeXi(grid_w, grid_h, grid_d);
    cp.fill(-1);

    cuda_check(cudaMalloc(&DEV_cp, n_cells*sizeof(int)));
    cuda_check(cudaMalloc(&DEV_LS, n_cells*sizeof(scalar_t)));
    cuda_check(cudaMalloc(&DEV_LS_pos, n_cells*sizeof(scalar_t)));
    cuda_check(cudaMalloc(&DEV_LS_neg, n_cells*sizeof(scalar_t)));
    cuda_check(cudaMalloc(&DEV_grad_LS_x, n_cells*sizeof(scalar_t)));
    cuda_check(cudaMalloc(&DEV_grad_LS_y, n_cells*sizeof(scalar_t)));
    cuda_check(cudaMalloc(&DEV_grad_LS_z, n_cells*sizeof(scalar_t)));
}

void SimPLSCUDA::advance(int cur_step, scalar_t dt) {
    ExecTimerSteps timer("SIMPLSCUDA", false);
    sp_index.clear(); np_index.clear(); pp_index.clear();
    sp_count.clear(); np_count.clear(); pp_count.clear();

    cuda_check(cudaMemcpy(DEV_LS, LS.memptr(), n_cells*sizeof(scalar_t), cudaMemcpyHostToDevice));
#ifdef VERBOSE_PLS
    std::cout << "Processing PLS" << std::endl;
#endif
    if (iteration == 1){
        reinit_pls();
    }
    timer.next("Copy LS, and initial init");

    // advect particles
#ifdef VERBOSE_PLS
    std::cout << "Advecting " << n_sp << "(surface) particles on device." << std::endl;
#endif

    advect_particles_on_device(get_number_of_blocks(n_sp), threads_in_block, n_sp, dt, DEV_sp, DEV_V, DEV_C);
    advect_particles_on_device(get_number_of_blocks(n_np), threads_in_block, n_np, dt, DEV_np, DEV_V, DEV_C);
    advect_particles_on_device(get_number_of_blocks(n_pp), threads_in_block, n_pp, dt, DEV_pp, DEV_V, DEV_C);
    timer.next("Advect particles");

    advect_RK3_CUDA(LS, {0, 0, 0}, DEV_V, dt, DEV_C, true, false);
    timer.next("Advect level set");
//    advect_RK3(LS, {0, 0, 0}, sim, true, false, true);
    cuda_check(cudaMemcpy(DEV_LS, LS.memptr(), n_cells*sizeof(scalar_t), cudaMemcpyHostToDevice));
//     Sort the particles after advection to prepare for redistancing
#ifdef VERBOSE_PLS
    std::cout << "Assign particles to cells." << std::endl;
#endif
    int* DEV_sp_keys, *DEV_np_keys, *DEV_pp_keys;
    cuda_check(cudaMalloc(&DEV_sp_keys, n_sp*sizeof(int)));
    cuda_check(cudaMalloc(&DEV_np_keys, n_np*sizeof(int)));
    cuda_check(cudaMalloc(&DEV_pp_keys, n_pp*sizeof(int)));
    calculate_particle_cells_on_device(get_number_of_blocks(n_sp), threads_in_block, n_sp, DEV_sp, DEV_sp_keys, DEV_C);
    calculate_particle_cells_on_device(get_number_of_blocks(n_np), threads_in_block, n_np, DEV_np, DEV_np_keys, DEV_C);
    calculate_particle_cells_on_device(get_number_of_blocks(n_pp), threads_in_block, n_pp, DEV_pp, DEV_pp_keys, DEV_C);
    timer.next("Calculate particle cells");
#ifdef VERBOSE_PLS
    std::cout << "Sorting particles." << std::endl;
#endif
    sort_particles_by_key_on_device(DEV_sp, DEV_sp_keys, n_sp);
    sort_particles_by_key_on_device(DEV_np, DEV_np_keys, n_np);
    sort_particles_by_key_on_device(DEV_pp, DEV_pp_keys, n_pp);
    generate_indicies_for_particles(DEV_sp_keys, DEV_sp_index, DEV_sp_count, sp_index, sp_count, n_sp);
    generate_indicies_for_particles(DEV_np_keys, DEV_np_index, DEV_np_count, np_index, np_count, n_np);
    generate_indicies_for_particles(DEV_pp_keys, DEV_pp_index, DEV_pp_count, pp_index, pp_count, n_pp);
    timer.next("Sort and generate indices for particles");
//    std::cout << "sp_index" << std::endl;
//    for (const auto &i: sp_index){
//        std::cout << i << std::endl;
//    }
//    std::cout << "np_index" << std::endl;
//    for (const auto &i: np_index){
//        std::cout << i << std::endl;
//    }
//    std::cout << "pp_index" << std::endl;
//    for (const auto &i: pp_index){
//        std::cout << i << std::endl;
//    }
    cp.fill(-1);
#ifdef VERBOSE_PLS
    std::cout << "Update signed and unsigned levelset distances." << std::endl;
#endif
    cuda_check(cudaMemcpy(DEV_cp, cp.memptr(), n_cells*sizeof(int), cudaMemcpyHostToDevice));
    update_levelset_distances_on_device(n_cells, get_number_of_blocks(n_cells), threads_in_block, DEV_LS, DEV_sp, DEV_sp_index, DEV_sp_count, DEV_cp, DEV_C);
    cuda_check(cudaMemcpy(LS_unsigned.memptr(), DEV_LS, n_cells*sizeof(scalar_t), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(cp.memptr(), DEV_cp, n_cells*sizeof(int), cudaMemcpyDeviceToHost));
    update_levelset_distances_on_device(n_cells, get_number_of_blocks(n_cells), threads_in_block, DEV_LS_pos, DEV_pp, DEV_pp_index, DEV_pp_count, DEV_cp, DEV_C);
    update_levelset_distances_on_device(n_cells, get_number_of_blocks(n_cells), threads_in_block, DEV_LS_neg, DEV_np, DEV_np_index, DEV_np_count, DEV_cp, DEV_C);
    cuda_check(cudaFree(DEV_sp_index)); cuda_check(cudaFree(DEV_pp_index)); cuda_check(cudaFree(DEV_np_index));
    cuda_check(cudaFree(DEV_sp_count)); cuda_check(cudaFree(DEV_pp_count)); cuda_check(cudaFree(DEV_np_count));
    cuda_check(cudaMemcpy(LS_pos.memptr(), DEV_LS_pos, n_cells*sizeof(scalar_t), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(LS_neg.memptr(), DEV_LS_neg, n_cells*sizeof(scalar_t), cudaMemcpyDeviceToHost));
    timer.next("Update level set distances");

    cuda_check(cudaMemcpy(&sp[0], DEV_sp, n_sp*sizeof(CUVEC::Vec3d), cudaMemcpyDeviceToHost));
//    cuda_check(cudaMemcpy(&np[0], DEV_np, n_np*sizeof(CUVEC::Vec3d), cudaMemcpyDeviceToHost));
//    cuda_check(cudaMemcpy(&pp[0], DEV_pp, n_pp*sizeof(CUVEC::Vec3d), cudaMemcpyDeviceToHost));
#ifdef VERBOSE_PLS
    std::cout << "Propagating levelset values from interface." << std::endl;
#endif
    timer.next("Copy particles to device");
    propagate_interface_distances();
    timer.next("Propagate interface distances");

    correct_grid_point_signs();
    timer.next("Correct grid point signs");
//    send_particles_to_host_pls();
//    redistance_grid_from_particles();
//    SimPLS::correct_grid_point_signs();

#ifdef DEBUGLS
    std::cout << "Level set after sign correction" << std::endl;
    std::cout << LS << std::endl;
#endif

    if (iteration % reseed_interval == 0) {
        cuda_check(cudaMemcpy(DEV_LS, LS.memptr(), n_cells*sizeof(scalar_t), cudaMemcpyHostToDevice));
//        SimPLS::reseed_particles();
//        update_particles_on_device_from_host();
        reinit_pls();
    }
    timer.next("Reinit PLS");

    ++iteration;

    precalc_fedkiw_curvature();
}

void SimPLSCUDA::reinit_pls(){
    sp.clear();
    free_particles_on_device();
#ifdef VERBOSE_PLS
    std::cout << "Reinitializing the PLS." << std::endl;
#endif
    std::vector<CUVEC::Vec3i> surf_coords;
    for (int k = 0; k < grid_d; k++) {
        for (int j = 0; j < grid_h; j++) {
            for (int i = 0; i < grid_w; i++){
                if (std::abs(LS(i,j,k)) < 1.0*bandwidth*dx){
                    surf_coords.push_back(CUVEC::Vec3i(i,j,k));
                }
            }
        }
    }

    n_sp = surf_coords.size()*surface_particles_per_cell;
    n_np = surf_coords.size()*sign_particles_per_cell;
    n_pp = surf_coords.size()*sign_particles_per_cell;
    cudaMalloc(&DEV_sp, n_sp*sizeof(CUVEC::Vec3d));
    cudaMalloc(&DEV_np, n_np*sizeof(CUVEC::Vec3d));
    cudaMalloc(&DEV_pp, n_pp*sizeof(CUVEC::Vec3d));
    sp.resize(n_sp);
    CUVEC::Vec3i *DEV_surf_coords;
    cudaMalloc(&DEV_surf_coords, surf_coords.size()*sizeof(CUVEC::Vec3i));
    cudaMemcpy(DEV_surf_coords, &surf_coords[0], surf_coords.size()*sizeof(CUVEC::Vec3i), cudaMemcpyHostToDevice);
#ifdef VERBOSE_PLS
    std::cout << "Reseeding particles over " << surf_coords.size() << " cells." << std::endl;
#endif
    int max_particles = surf_coords.size()*surface_particles_per_cell;
#ifdef VERBOSE_PLS
    std::cout << "Maximum permitted particle counted " << max_particles << std::endl;
    std::cout << "Copying and calculating LS gradient to device." << std::endl;
#endif
//    CubeX grad_LS_x = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
//    CubeX grad_LS_y = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
//    CubeX grad_LS_z = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
//    calc_grad(LS, grad_LS_x, grad_LS_y, grad_LS_z, dx);
//    cudaMemcpy(DEV_grad_LS_x, grad_LS_x.memptr(), n_cells*sizeof(scalar_t), cudaMemcpyHostToDevice);
//    cudaMemcpy(DEV_grad_LS_y, grad_LS_y.memptr(), n_cells*sizeof(scalar_t), cudaMemcpyHostToDevice);
//    cudaMemcpy(DEV_grad_LS_z, grad_LS_z.memptr(), n_cells*sizeof(scalar_t), cudaMemcpyHostToDevice);
    reseed_surface_particles_on_device(surf_coords.size(), DEV_surf_coords, DEV_sp, DEV_LS, DEV_grad_LS_x, DEV_grad_LS_y, DEV_grad_LS_z, surface_particles_per_cell, 1.0, DEV_C, threads_in_block);
    reseed_sign_particles_on_device(surf_coords.size(), DEV_sp, DEV_pp, DEV_np, DEV_LS, DEV_grad_LS_x, DEV_grad_LS_y, DEV_grad_LS_z, surface_particles_per_cell, sign_particles_per_cell, 1.0, DEV_C, threads_in_block);

    cudaFree(DEV_surf_coords);
}

void SimPLSCUDA::generate_indicies_for_particles(int* DEV_p_keys, int* &DEV_p_index, int* &DEV_p_count, std::vector<int> &p_index, std::vector<int> &p_count, int n_p){
    std::vector<int> p_keys;
    p_keys.resize(n_p);
    cudaMemcpy(&p_keys[0], DEV_p_keys, n_p*sizeof(int), cudaMemcpyDeviceToHost);
    int last = -1;
    p_index.resize(n_cells);
    p_count.resize(n_cells);
    std::fill(p_count.begin(), p_count.end(), 0);
    std::fill(p_index.begin(), p_index.end(), -1);
    int i = 0;
    int count = 0;
    int counting_key = p_keys[0];
    int n_stray = 0;
    for (const auto &k : p_keys){
        if (k < 0){
            n_stray++;
            i++;
            continue;
        }
        if (k != last) {
            p_index[k] = i;
            p_count[counting_key] = count;
            count = 0;
            counting_key = k;
        }
        count++;
        i++;
        last = k;
    }
    std::cout << "Number of stray particles " << n_stray << std::endl;
    p_count[counting_key] = count;
    p_keys.clear();
    cudaFree(DEV_p_keys);
    cuda_check(cudaMalloc(&DEV_p_index, n_cells*sizeof(int)));
    cuda_check(cudaMemcpy(DEV_p_index, &p_index[0], n_cells*sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMalloc(&DEV_p_count, n_cells*sizeof(int)));
    cuda_check(cudaMemcpy(DEV_p_count, &p_count[0], n_cells*sizeof(int), cudaMemcpyHostToDevice));
}

void SimPLSCUDA::allocate_space_for_particles_on_device(){
    cudaMalloc(&DEV_sp, sp.size()*sizeof(CUVEC::Vec3d));
    cudaMalloc(&DEV_np, np.size()*sizeof(CUVEC::Vec3d));
    cudaMalloc(&DEV_pp, pp.size()*sizeof(CUVEC::Vec3d));
}

void SimPLSCUDA::update_particles_on_device(){
    cudaMemcpy(DEV_sp, &sp[0], sp.size()*sizeof(CUVEC::Vec3d), cudaMemcpyHostToDevice);
    cudaMemcpy(DEV_np, &np[0], np.size()*sizeof(CUVEC::Vec3d), cudaMemcpyHostToDevice);
    cudaMemcpy(DEV_pp, &pp[0], pp.size()*sizeof(CUVEC::Vec3d), cudaMemcpyHostToDevice);
}

void SimPLSCUDA::update_particles_on_device_from_host(){
    sp.clear();
    np.clear();
    pp.clear();
    free_particles_on_device();
    n_sp = surface_points.size();
    n_pp = pos_points.size();
    n_np = neg_points.size();
    cudaMalloc(&DEV_sp, n_sp*sizeof(CUVEC::Vec3d));
    cudaMalloc(&DEV_np, n_np*sizeof(CUVEC::Vec3d));
    cudaMalloc(&DEV_pp, n_pp*sizeof(CUVEC::Vec3d));
    for (const auto &p : surface_points){
        sp.push_back({p[0], p[1], p[2]});
    }
    for (const auto &p : neg_points){
        np.push_back({p[0], p[1], p[2]});
    }
    for (const auto &p : pos_points){
        pp.push_back({p[0], p[1], p[2]});
    }
    update_particles_on_device();
}

void SimPLSCUDA::copy_particles_to_host(){
    cudaMemcpy(&sp[0], DEV_sp, n_sp*sizeof(CUVEC::Vec3d), cudaMemcpyDeviceToHost);
    cudaMemcpy(&np[0], DEV_np, n_np*sizeof(CUVEC::Vec3d), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pp[0], DEV_pp, n_pp*sizeof(CUVEC::Vec3d), cudaMemcpyDeviceToHost);
}

/* send_particles_to_host_pls
 * Suspicious of something the CUDA SimPLS class is doing? Load up a CPU SimPLS replacement method and convert the CUDA
 * particles to host particles for testing.
 */
void SimPLSCUDA::send_particles_to_host_pls(){
    copy_particles_to_host();
    surface_points.clear();
    neg_points.clear();
    pos_points.clear();

    for (const auto &p : sp){
        surface_points.push_back({p[0], p[1], p[2]});
    }
    for (const auto &p : np){
        neg_points.push_back({p[0], p[1], p[2]});
    }
    for (const auto &p : pp){
        pos_points.push_back({p[0], p[1], p[2]});
    }
}

void SimPLSCUDA::free_particles_on_device(){
    if (DEV_sp != nullptr)
        cudaFree(DEV_sp);
    if (DEV_np != nullptr)
        cudaFree(DEV_np);
    if (DEV_pp != nullptr)
        cudaFree(DEV_pp);
}

int SimPLSCUDA::get_number_of_blocks(int n_particles){
   return std::ceil(n_particles / (scalar_t)threads_in_block);
}

void SimPLSCUDA::redistance_neighbour(const Vector3i &face, CubeX &unsigned_dist, const CubeXi &closest_point, std::queue<Vector3i> &cell_queue, int cp_id){
    if (is_coord_valid(face, {grid_w, grid_h, grid_d})) {
        CUVEC::Vec3d pt(face(0)*dx, face(1)*dx, face(2)*dx);
        CUVEC::Vec3d sep = pt - sp[cp_id];
        scalar_t distance = CUVEC::mag(sep);
        if (distance < unsigned_dist(face(0), face(1), face(2)) || cp(face(0), face(1), face(2)) == -1) {
            unsigned_dist(face(0), face(1), face(2)) = distance;
            cp(face(0), face(1), face(2)) = cp_id;
//            if (unsigned_dist(face(0), face(1), face(2)) < dx*5.0){
                cell_queue.push({face(0), face(1), face(2)});
//            }
        }
    }
}

void SimPLSCUDA::propagate_interface_distances() {
    //build initial queue of cells with known closest points
    std::queue<Vector3i> cell_queue;
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                if (cp(i, j, k) != -1) {
                    cell_queue.push(Vector3i({i, j, k}));
                }
//                else {
//                    LS_unsigned(i, j, k) = std::abs(LS(i, j, k));
//                }
            }
        }
    }

    //Repeat until everyone has pushed to all their neighbours
    //and no new changes have occurred.
    while (!cell_queue.empty()) {
        Vector3i cell = cell_queue.front();

        //Get info
        int i = cell(0);
        int j = cell(1);
        int k = cell(2);
        int cp_id = cp(i, j, k);

        //Check neighbours;
        redistance_neighbour({i - 1, j, k}, LS_unsigned, cp, cell_queue, cp_id);
        redistance_neighbour({i + 1, j, k}, LS_unsigned, cp, cell_queue, cp_id);
        redistance_neighbour({i, j - 1, k}, LS_unsigned, cp, cell_queue, cp_id);
        redistance_neighbour({i, j + 1, k}, LS_unsigned, cp, cell_queue, cp_id);
        redistance_neighbour({i, j, k - 1}, LS_unsigned, cp, cell_queue, cp_id);
        redistance_neighbour({i, j, k + 1}, LS_unsigned, cp, cell_queue, cp_id);

        cell_queue.pop();
    }

    //Take existing signs for starters.
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                LS(i, j, k) = sgn(LS(i, j, k)) * LS_unsigned(i, j, k);
            }
        }
    }
}

void SimPLSCUDA::correct_grid_point_signs() {
    //Take the sign of the closer sign particle
    for (int k = 0; k < LS.n_slices; k++) {
        for (int j = 0; j < LS.n_cols; j++) {
            for (int i = 0; i < LS.n_rows; i++) {
                if (std::abs(LS(i, j, k)) < 1.0*bandwidth*dx) {
                    if (LS_neg(i, j, k) < LS_pos(i, j, k)) {
                        LS(i, j, k) = -std::abs(LS(i, j, k));
                    } else {
                        LS(i, j, k) = std::abs(LS(i, j, k));
                    }
                }
            }
        }
    }
}

scalar_t SimPLSCUDA::get_height_normal(const Vector3 &base, const Vector3 &n, scalar_t h_ref, scalar_t dx_column){
    return get_height_normal_pls(base, n, h_ref, dx_column);
}

scalar_t SimPLSCUDA::get_height_normal_pls(const Vector3 &base_in, const Vector3 &n_in, scalar_t h_ref, scalar_t dx_column, int n_search){
    CUVEC::Vec3d base(base_in(0), base_in(1), base_in(2));
    CUVEC::Vec3d n(n_in(0), n_in(1), n_in(2));
    CUVEC::Vec3d n_unit = CUVEC::normalized(n);
    scalar_t best_error = 1000;
    scalar_t best_height = 0;
    scalar_t p_dist = 1.0/std::sqrt(surface_particles_per_cell); // rough expected interparticle distances
    scalar_t cell_tol = 1.0*p_dist; // distance away from border to add neighbouring cells to the search
    scalar_t p_tol = 1.0*p_dist*dx;

    CUVEC::Vec3d search = base;
    for (int i = 0; i < n_search; i++) {
        std::vector<CUVEC::Vec3i> search_coords;
        CUVEC::Vec3d search_coord_d(search[0]/dx, search[1]/dx, search[2]/dx);
        CUVEC::Vec3i search_coord((int)std::round(search_coord_d[0]), (int)std::round(search_coord_d[1]), (int)std::round(search_coord_d[2]));
        search_coords.push_back(search_coord);
        search_coord_d += CUVEC::Vec3d(cell_tol, cell_tol, cell_tol);
        CUVEC::Vec3i search_coord_2((int)std::round(search_coord_d[0]), (int)std::round(search_coord_d[1]), (int)std::round(search_coord_d[2]));
        if (search_coord_2 != search_coord)
            search_coords.push_back(search_coord_2);
        search_coord_d -= CUVEC::Vec3d(2.0*cell_tol, 2.0*cell_tol, 2.0*cell_tol);
        search_coord_2 = CUVEC::Vec3i((int)std::round(search_coord_d[0]), (int)std::round(search_coord_d[1]), (int)std::round(search_coord_d[2]));
        if (search_coord_2 != search_coord){
            search_coords.push_back(search_coord_2);
        }
        for (const auto& coord : search_coords) {
            if (get_height_normal_pls_search(coord, base, n_unit, h_ref, p_tol, best_error, best_height)){
                return best_height;
            }
        }
        search += n;
    }

    return best_height;
}

bool SimPLSCUDA::get_height_normal_pls_search(const CUVEC::Vec3i &coord, const CUVEC::Vec3d &base,
        const CUVEC::Vec3d &n_unit, scalar_t h_ref, scalar_t p_tol, scalar_t &best_error, scalar_t &best_height){
    if (is_coord_valid({coord[0], coord[1], coord[2]}, {grid_w, grid_h, grid_d})) {
        int cell_id_search = ccti(coord[0], coord[1], coord[2], grid_w, grid_h);
        int sp_start_search = sp_index[cell_id_search];
        int sp_count_search = sp_count[cell_id_search];
        if (sp_start_search != -1) {
            for (int j = sp_start_search; j < sp_start_search + sp_count_search; j++) {
                CUVEC::Vec3d d = sp[j] - base; // separate vector between base and particle.
                scalar_t h = CUVEC::dot(d, n_unit); // the actual height is the part along the normal vector.
                scalar_t error = std::sqrt(std::abs(CUVEC::mag2(d) - h * h)); // what component is perpendicular to the normal vector.
                error += 0.1*std::abs(h_ref - h); // how much to penalize massive height differences (multi surface)
                if (error < best_error){
                    best_height = h;
                    if (error < p_tol){
                        return true;
                    }
                    best_error = error;
                }
            }
        }
    }

    return false;
}

scalar_t SimPLSCUDA::get_curvature(Vector3 &pos){
    scalar_t retval = clamp(1.0*(get_curvature_height_normal(pos)), -1.0/dx, 1.0/dx);
//    std::cout << retval << " " << pos(0) << " " << pos(1) << " " << pos(2) << std::endl;
    return retval;
}

void SimPLSCUDA::print_all_levelsets(){
    std::cout << "LS" << std::endl;
    std::cout << LS << std::endl;
    std::cout << "+LS" << std::endl;
    std::cout << LS_pos << std::endl;
    std::cout << "-LS" << std::endl;
    std::cout << LS_neg << std::endl;
}

void SimPLSCUDA::print_item(int* item){
    for (int k = 0; k < grid_d; k++){
        for (int i = 0; i < grid_w; i++){
            for (int j = 0; j < grid_h; j++){
                int index = ccti(i, j, k, grid_w, grid_h);
                std::cout << item[index] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
}

void SimPLSCUDA::save_data(){
    LS.save("LS.bin");
}

void SimPLSCUDA::load_data(){
    LS.load("LS.bin");
    reinit_pls();
}

void sort_particles_by_key(CUVEC::Vec3d *p, int *cell_ids, int n_p){
    std::vector<CUVEC::Vec3d> HOST_p;
    HOST_p.resize(n_p);
    cuda_check(cudaMemcpy(&HOST_p[0], p, n_p*sizeof(CUVEC::Vec3d), cudaMemcpyDeviceToHost));
    std::vector<int> HOST_key;
    HOST_key.resize(n_p);
    cuda_check(cudaMemcpy(&HOST_key[0], cell_ids, n_p*sizeof(int), cudaMemcpyDeviceToHost));
    std::multimap<int, CUVEC::Vec3d> cell_and_p;
    for (int i = 0; i < n_p; i++){
        cell_and_p.insert(std::pair<int, CUVEC::Vec3d>(HOST_key[i], HOST_p[i]));
    }
    int i = 0;
    for (auto const& cp : cell_and_p){
        HOST_key[i] = cp.first;
        HOST_p[i] = cp.second;
        i++;
    }
    cuda_check(cudaMemcpy(p, &HOST_p[0], n_p*sizeof(CUVEC::Vec3d), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(cell_ids, &HOST_key[0], n_p*sizeof(int), cudaMemcpyHostToDevice));
}

//void SimPLSCUDA::test_height_function(){
//    int n_shots = 100;
//    for (int i = 0; i < n_shots; i++){
//        scalar_t theta = ((scalar_t) i/ (scalar_t) n_shots)*(2*PI);
//        Vector3 n = {std::cos(theta), 0, std::sin(theta)};
//        n = arma::normalise(n);
//        std::cout << "Angle: " << theta << " degrees." << std::endl;
//        scalar_t height = get_height_normal_pls({sim_w/2.0, sim_h/2.0, sim_d/2.0}, dx*n, 0, dx, 10);
//        std::cout << "Height: " << height/dx << std::endl;
//    }
//
//    //Specific examples
//    scalar_t height = get_height_normal_pls({0.411416,0.463896, 0.398627}, {-0.0124096, -0.00925429, -0.039957}, 0, dx, 7);
//    std::cout << "Specific Test Height: " << height/dx << std::endl;
//
//    height = get_height_normal_pls({0.493445, 0.500042, 0.36478}, {0.0124096, -0.00925429, -0.039957}, 0, dx, 7);
//    std::cout << "Specific Test Height: " << height/dx << std::endl;
//
//    height = get_height_normal_pls({0.49258, 0.489817, 0.352168}, {0.00247348, -0.00733262, -0.0344006}, 0, dx, 7);
//    std::cout << "Specific Test Height: " << height/dx << std::endl;
//}

#endif