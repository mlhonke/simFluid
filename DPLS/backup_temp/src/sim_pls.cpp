//
// Created by graphics on 12/01/19.
//

#include "sim_pls.hpp"
#include <algorithm>
#include "fd_math.hpp"
#include "interpolate.hpp"
#include "advect_dev.cuh"
#include <iomanip>
#include <math.h>
#include <cmath>
#include <thread>
#include "sim_utils.hpp"

#ifndef USECUDA

SimPLS::SimPLS(SimParams &C, std::array<CubeX, 3> &V) : SimLevelSet(C, V) {
    bandwidth = 1.0;
    reseed_interval = 10;
    iteration = 1;
    surface_particles_per_cell = 200;
    sign_particles_per_cell = 200;
    ni = grid_w;
    nj = grid_h;
    nk = grid_d;
    grid_particles.resize(grid_w*grid_h*grid_d);
}

void SimPLS::reset_LS(const CubeX& LS_new) {
    LS = LS_new;
    ni = LS.n_rows;
    nj = LS.n_cols;
    nk = LS.n_slices;
    reseed_particles();
    iteration = 1;
}

void SimPLS::advance(int cur_step, scalar_t dt) {
    std::cout << "Processing PLS" << std::endl;
    if (iteration == 1){
//        precalc_grad();
        reseed_particles();
    }

    std::cout << "Number of particles seeded " << surface_points.size() << std::endl;

    int n_pos_points = pos_points.size()/n_threads;
    int n_neg_points = neg_points.size()/n_threads;
    int n_surface_points = surface_points.size()/n_threads;
    std::thread aths[n_threads*3]; //advection threads

    std::cout << "Starting advection." << std::endl;
    for (int i = 0; i < n_threads; i++){
        if (i == n_threads - 1){ // if last iteration make sure we include particles that didn't divide nicely
            aths[3*i] = std::thread(&SimPLS::advect_particles, this, dt, std::ref(pos_points), i * n_pos_points, (int) pos_points.size());
            aths[3*i+1] = std::thread(&SimPLS::advect_particles, this, dt, std::ref(neg_points), i * n_neg_points, (int) neg_points.size());
            aths[3*i+2] = std::thread(&SimPLS::advect_particles, this, dt, std::ref(surface_points), i * n_surface_points, (int) surface_points.size());
        } else {
            aths[3*i] = std::thread(&SimPLS::advect_particles, this, dt, std::ref(pos_points), i * n_pos_points, (i + 1) * n_pos_points);
            aths[3*i+1] = std::thread(&SimPLS::advect_particles, this, dt, std::ref(neg_points), i * n_neg_points, (i + 1) * n_neg_points);
            aths[3*i+2] = std::thread(&SimPLS::advect_particles, this, dt, std::ref(surface_points), i * n_surface_points, (i + 1) * n_surface_points);
        }
    }

    for (int i = 0; i < 3*n_threads; i++){
        aths[i].join();
    }

    std::cout << "Stopping advection." << std::endl;

#ifdef DEBUGLS
    std::cout << "LS before advection\n " << std::setprecision(5) << LS << std::endl;
#endif

//    sim.advect_quantity(LS, sim.u, sim.v, sim.label, sim.dt, true, false);
    advect_RK3(LS, {0, 0, 0}, V, dt, C, true, false);

#ifdef DEBUGLS
    std::cout << "LS after advection\n " << std::setprecision(5) << LS << std::endl;
#endif

    redistance_grid_from_particles();

#ifdef DEBUGLS
    std::cout << "Level set after redistancing" << std::endl;
    std::cout << LS << std::endl;
#endif

    correct_grid_point_signs();

#ifdef DEBUGLS
    std::cout << "Level set after sign correction" << std::endl;
    std::cout << LS << std::endl;
#endif

    if (iteration % reseed_interval == 0) {
//        precalc_grad();
        reseed_particles();
    }

    sort_particles();

    ++iteration;

//    precalc_fedkiw_curvature();
    std::cout << "Done processing PLS" << std::endl;
}

//void SimPLS::advect_particles(scalar_t dt, std::vector<Vector3> &points){
//    advect_particles(dt, points, 0, points.size());
//}

void SimPLS::advect_particles(scalar_t dt, std::vector<Vector3> &points, int start, int end){
    for (unsigned int i = start; i < end; ++i) {
        Vector3 pos = points[i];
        Vector3 vel = lerp_velocity(pos, V, dx, true);
        pos += 0.5*dt*vel;
        vel = lerp_velocity(pos, V, dx, true);
        points[i] += dt*vel;
    }
}

Vector3 SimPLS::get_normal(const Vector3& pos) {
    Vector3 grad = get_grad_lerped(pos, LS, dx);
//    grad(0) = grid_bilerp(pos(0), pos(1), LS_grad_x);
//    grad(1) = grid_bilerp(pos(0), pos(1), LS_grad_y);
    scalar_t magnitude = arma::norm(grad);
    if (magnitude > 1e-10) {
        grad = arma::normalise(grad);
        return grad;
    }
    else {
        return {0, 0, 0};
    }
}

bool SimPLS::get_surface_point(const Vector3& pos, Vector3& result) {
    Vector3 search_pt = pos;
    scalar_t dist = get_LS(search_pt);
    scalar_t tol = 1E-12;
    int iters = 0;
    while (std::abs(dist) > tol*dx && iters < 300){
        Vector3 normal = get_normal(search_pt);
        search_pt -= dist*normal;
        dist = get_LS_cerp(search_pt);
        ++iters;
    }
    result = search_pt;
    bool valid = std::abs(dist) <= tol*dx;
//    if (!valid)
//        std::cout << "get_surface_point failed to find point on surface" << std::endl;
    return valid;
}

void SimPLS::reseed_particles() {
    pos_points.clear();
    neg_points.clear();
    surface_points.clear();
    scalar_t nw_sign = std::sqrt(sign_particles_per_cell); // Assume square cells so nh = nw.
    scalar_t scale_sign = dx/(nw_sign-1);
    scalar_t nw_surface = std::sqrt(surface_particles_per_cell); // Assume square cells so nh = nw.
    scalar_t scale_surface = dx/(nw_surface-1);
    //std::cout << LS << std::endl;

    static int seed = 0;
    for (int k = 0; k < LS.n_slices - 1; k++) {
        for (int j = 0; j < LS.n_cols - 1; ++j) {
            for (int i = 0; i < LS.n_rows - 1; ++i) {
                seed = abs(i-grid_w/2)*grid_h + j;
                //sign particles
                if (std::abs(LS(i,j,k)) < 2.0*bandwidth*dx) {
//                    for (int p = 0; p < sign_particles_per_cell; ++p) {
//                        int ip = p / (int) nw_sign;
//                        int jp = p % (int) nw_sign;
//                        Vector3 start = sim.get_position(i, j, k);
////                scalar_t x_off = scale_sign* ip;
//                        scalar_t x_off = dx * marchingtets::randhashf(++seed, 0, 1);
////                scalar_t y_off = scale_sign* jp;
//                        scalar_t y_off = dx * marchingtets::randhashf(++seed, 0, 1);
//                        scalar_t z_off = dx * marchingtets::randhashf(++seed, 0, 1);
//                        Vector3 offset = {x_off, y_off, z_off};
//                        Vector3 newpt = start + offset;
//                        scalar_t LS_val = get_LS(newpt);
//                        if (std::abs(LS_val) < bandwidth * dx) {
//                            if (LS_val > 0) {
//                                pos_points.push_back(newpt);
//                            } else {
//                                neg_points.push_back(newpt);
//                            }
//                        }
//                    }

                    //surface particles
                    scalar_t dir[2] = {-1.0, 1.0};
                    for (int p = 0; p < surface_particles_per_cell; ++p) {
                        int ip = p / (int) nw_surface;
                        int jp = p % (int) nw_surface;
                        Vector3 start = get_position(i, j, k, dx);
//                scalar_t x_off = scale_surface* ip;
                        scalar_t x_off = dx * marchingtets::randhashf(++seed, 0, 1);
//                scalar_t y_off = scale_surface* jp;
                        scalar_t y_off = dx * marchingtets::randhashf(++seed, 0, 1);
                        scalar_t z_off = dx * marchingtets::randhashf(++seed, 0, 1);
                        Vector3 offset = {x_off, y_off, z_off};
                        Vector3 newpt = start + offset;
                        scalar_t LS_val = get_LS(newpt);
                        if (std::abs(LS_val) < bandwidth * dx) {
                            Vector3 surf;
                            bool success = get_surface_point(newpt, surf);
                            if (success) {
                                Vector3 p0 = surf + dir[p%2]*0.25*dx*get_normal(surf);
//                                Vector3 p1 = surf - 0.25*dx*get_normal(surf);
                                if (get_LS(p0) > 0){
                                    pos_points.push_back(p0);
                                } else {
                                    neg_points.push_back(p0);
                                }
//                                if (get_LS(p1) <= 0){
//                                    neg_points.push_back(p1);
//                                } else {
//                                    pos_points.push_back(p1);
//                                }
//                        std::cout << "Surface particle position: " << surf(0) << " " << surf(1) << std::endl;
                                surface_points.push_back(surf);
                            }
                        }
                    }
                }
            }
        }
    }
}

void SimPLS::generate_sign_field(const std::vector<Vector3> &points, CubeX &field){
    for (unsigned int p = 0; p < points.size(); ++p) {
        Vector3 pos = points[p];
        Vector3 grid_coords = (pos) / dx;
        grid_coords(0) = clamp(grid_coords(0), 0.0, LS.n_rows - 2.0);
        grid_coords(1) = clamp(grid_coords(1), 0.0, LS.n_cols - 2.0);
        grid_coords(2) = clamp(grid_coords(2), 0.0, LS.n_slices - 2.0);
        for (int offk = (int)floor(grid_coords(2)); offk < grid_coords(2) + 1; ++offk) {
            for (int offj = (int) floor(grid_coords(1)); offj < grid_coords(1) + 1; ++offj) {
                for (int offi = (int) floor(grid_coords(0)); offi < grid_coords(0) + 1; ++offi) {
                    Vector3 gridpoint = get_position(offi, offj, offk, dx);
                    scalar_t distance = get_distance(gridpoint, pos);
                    field(offi, offj, offk) = std::min(field(offi, offj, offk), distance);
                }
            }
        }
    }
}

void SimPLS::correct_grid_point_signs() {

    //Node gets the sign of the closest sign particle.

    //Put distance of nearest positive particle (if any) into pos_field
    CubeX pos_field(ni, nj, nk);
    pos_field.fill(10*(ni+nj+nk)*dx);
    generate_sign_field(pos_points, pos_field);

    //Repeat for negative particles
    CubeX neg_field(ni, nj, nk);
    neg_field.fill(10*(ni+nj+nk)*dx);
    generate_sign_field(neg_points, neg_field);

//    std::cout << pos_field << std::endl;
//    std::cout << neg_field << std::endl;

    //Take the sign of the closer sign particle
    for (int k = 0; k < LS.n_slices; ++k) {
        for (int j = 0; j < LS.n_cols; ++j) {
            for (int i = 0; i < LS.n_rows; ++i) {
                if (std::abs(LS(i, j, k)) < dx) {
                    if (neg_field(i, j, k) < pos_field(i, j, k)) {
                        LS(i, j, k) = -std::abs(LS(i, j, k));
                    } else {
                        LS(i, j, k) = std::abs(LS(i, j, k));
                    }
                }
            }
        }
    }

}

void SimPLS::redistance_neighbour(const Vector3i &face, CubeX &unsigned_dist, CubeXi &closest_point, std::queue<Vector3i> &cell_queue, int cp){
    if (is_coord_valid(face, {grid_w, grid_h, grid_w})) {
        Vector3 pt = get_position(face(0), face(1), face(2), dx);
        scalar_t distance = get_distance(pt, surface_points[cp]);
        if (distance < unsigned_dist(face(0), face(1), face(2)) || closest_point(face(0), face(1), face(2)) == -1) {
            unsigned_dist(face(0), face(1), face(2)) = distance;
            closest_point(face(0), face(1), face(2)) = cp;
            cell_queue.push({face(0), face(1), face(2)});
        }
    }
}

void SimPLS::redistance_grid_from_particles() {
    CubeX unsigned_dist(ni, nj, nk);
    unsigned_dist.fill(10*(ni*nj*nk)*dx);
    CubeXi closest_point(ni, nj, nk);
    closest_point.fill(-1);

    //compute near surface distances
    for (unsigned int p = 0; p < surface_points.size(); ++p) {
        Vector3 pos = surface_points[p];

        //Find containing cell
        Vector3 grid_coords = (pos) / dx;
        grid_coords(0) = clamp(grid_coords(0), 0.0, LS.n_rows - 2.0);
        grid_coords(1) = clamp(grid_coords(1), 0.0, LS.n_cols - 2.0);
        grid_coords(2) = clamp(grid_coords(2), 0.0, LS.n_slices - 2.0);

        //Compute distance to surrounding numerous nodes
        for (int offk = (int)floor(grid_coords(2)) - 1; offk < grid_coords(2) + 2; ++offk) {
            for (int offj = (int) floor(grid_coords(1)) - 1; offj < grid_coords(1) + 2; ++offj) {
                for (int offi = (int) floor(grid_coords(0)) - 1; offi < grid_coords(0) + 2; ++offi) {

                    if (offi < 0 || offi >= ni || offj < 0 || offj >= nj || offk < 0 || offk >= nk) continue;

                    Vector3 gridpoint = get_position(offi, offj, offk, dx);
                    scalar_t abs_LS = get_distance(gridpoint, pos);
                    if (abs_LS < unsigned_dist(offi, offj, offk)) {
                        unsigned_dist(offi, offj, offk) = abs_LS;
                        closest_point(offi, offj, offk) = p;
                    }
                }
            }
        }
    }

    //build initial queue of cells with known closest points
    std::queue<Vector3i> cell_queue;
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                if (closest_point(i, j, k) != -1) {
                    cell_queue.push(Vector3i({i, j, k}));
                }
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
        int cp = closest_point(i, j, k);

        //Check neighbours;
        redistance_neighbour({i - 1, j, k}, unsigned_dist, closest_point, cell_queue, cp);
        redistance_neighbour({i + 1, j, k}, unsigned_dist, closest_point, cell_queue, cp);
        redistance_neighbour({i, j - 1, k}, unsigned_dist, closest_point, cell_queue, cp);
        redistance_neighbour({i, j + 1, k}, unsigned_dist, closest_point, cell_queue, cp);
        redistance_neighbour({i, j, k - 1}, unsigned_dist, closest_point, cell_queue, cp);
        redistance_neighbour({i, j, k + 1}, unsigned_dist, closest_point, cell_queue, cp);

        cell_queue.pop();
    }

    //Take existing signs for starters.
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                LS(i, j, k) = sgn(LS(i, j, k)) * unsigned_dist(i, j, k);
            }
        }
    }

}

void SimPLS::sort_particles(){
    for (int k = 0; k < grid_d; k++) {
        for (int j = 0; j < grid_h; j++) {
            for (int i = 0; i < grid_w; i++) {
                grid_particles[get_index(i, j, k)].clear();
            }
        }
    }

    scalar_t pad_amt = 0.2;
    for (auto p: surface_points){
        scalar_t x = p(0)/dx;
        int i = (int) expanded_round(x, pad_amt);
        scalar_t y = p(1)/dx;
        int j = (int) expanded_round(y, pad_amt);
        scalar_t z = p(2)/dx;
        int k = (int) expanded_round(z, pad_amt);
        int index = get_index(i,j,k);
        grid_particles[index].push_back(p);
        scalar_t intpart;
        scalar_t x_dec = x - i;
        scalar_t y_dec = y - j;
        scalar_t z_dec = z - k;
        if (x_dec < 0.5+pad_amt && x_dec > 0.5-pad_amt){
            index = get_index(i+1,j,k);
            grid_particles[index].push_back(p);
        }
        if (x_dec > -0.5 - pad_amt && x_dec < -0.5 + pad_amt){
            index = get_index(i-1,j,k);
            grid_particles[index].push_back(p);
        }
        if (y_dec > 0.5 -pad_amt && y_dec < 0.5 + pad_amt){
            index = get_index(i, j+1,k);
            grid_particles[index].push_back(p);
        }
        if (y_dec > -0.5 - pad_amt && y_dec < -0.5 + pad_amt){
            index = get_index(i, j-1,k);
            grid_particles[index].push_back(p);
        }
        if (z_dec > 0.5 -pad_amt && z_dec < 0.5 + pad_amt){
            index = get_index(i, j, k+1);
            grid_particles[index].push_back(p);
        }
        if (z_dec > -0.5 - pad_amt && z_dec < -0.5 + pad_amt){
            index = get_index(i, j, k-1);
            grid_particles[index].push_back(p);
        }

    }

//    for (int i = 0; i < grid_w; i++){
//        for (int j = 0; j < grid_h; j++){
//            std::cout << grid_particles[get_index(i,j)].size() << " ";
//        }
//        std::cout << std::endl;
//    }
}

scalar_t SimPLS::get_height_normal(Vector3 &base, Vector3 &n, scalar_t h_ref, scalar_t dx){
    return get_height_normal_pls(base, n, h_ref, dx);
}

scalar_t SimPLS::get_height_normal_pls(Vector3 base_in, Vector3 n_in, scalar_t h_ref, scalar_t dx, int n_search){
    Vector3 base{base_in(0), base_in(1), base_in(2)};
    Vector3 n{n_in(0), n_in(1), n_in(2)};
    Vector3 n_unit = arma::normalise(n);
    scalar_t best_error = 1000;
    scalar_t best_height = 0;
    std::vector<scalar_t> candidates;
    scalar_t tolerance = 2.0*(dx/std::sqrt(surface_particles_per_cell));

    Vector3 search = base;
    for (int i = 0; i < n_search; i++) {
//        std::cout << "LS val at search point " << grid_tricerp({search[0], search[1], search[2]}, LS, scale_w, false) << std::endl;
        Vector3 search_coord_d{search[0]/dx, search[1]/dx, search[2]/dx};
        Vector3i search_coord{(int)std::round(search_coord_d[0]),
                              (int)std::round(search_coord_d[1]),
                              (int)std::round(search_coord_d[2])};

        if (is_coord_valid(search_coord, {grid_w, grid_h, grid_d})) {
            int cell_id_search = ccti(search_coord[0], search_coord[1], search_coord[2], grid_w, grid_h);
            int sp_start_search = 0;
            int sp_count_search = grid_particles[cell_id_search].size();

            for (int j = sp_start_search; j < sp_start_search + sp_count_search; j++) {
                Vector3 p = grid_particles[cell_id_search][j];
                Vector3 d = p - base; // separate vector between base and particle.
                scalar_t h = arma::dot(d, n_unit); // the actual height is the part along the normal vector.
                scalar_t error = std::sqrt(std::abs(arma::norm(d,2) - h * h)); // what component is perpendicular to the normal vector.
                error += 0.1*std::abs(h_ref - h); // how much to penalize massive height differences (multi surface)
                if (error < best_error){
                    best_height = h;
                    best_error = error;
                }
                if (error < tolerance) {
                    candidates.push_back(h);
                }
            }
        }

        search += n;
    }

    if (candidates.empty()){ // didn't find anything closer than this, where is the surface?
//        std::cout << "Failed to find a height. " << std::endl;
//        print_item(&sp_count[0]);
//        print_item(&sp_index[0]);
//        std::cout << "base in " << base_in(0) << " " << base_in(1) << " " << base_in(2) << std::endl;
//        std::cout << "n_in " << n_in(0) << " " << n_in(1) << " " << n_in(2) << std::endl;
        return best_height;
    }

//    std::cout << "Number of candidates found " << candidates.size() << std::endl;

//    if (!candidates.empty()){
//        best_error = std::abs(h_ref - candidates[0]);
//        for (const auto &h : candidates) {
//            scalar_t error = std::abs(h_ref - h);
//            if (error < best_error) {
//                best_error = error;
//                best_height = h;
//            }
//        }
//    }

    return best_height;
}

scalar_t SimPLS::get_curvature(Vector3 &pos){
    scalar_t retval = clamp(get_curvature_height_normal(pos), -1.0/dx, 1.0/dx);
//    std::cout << retval << " at " << pos(0) << " " << pos(1) << std::endl;
    return retval;
}

void SimPLS::check_particle_bounds(){
    std::cout << "Surface particles." << std::endl;
    print_L1_norm_min_max(surface_points);
    std::cout << "Neg particles." << std::endl;


    print_L1_norm_min_max(neg_points);
    std::cout << "Pos particles." << std::endl;
    print_L1_norm_min_max(pos_points);
}

void SimPLS::print_L1_norm_min_max(const std::vector<Vector3> &points){
    Vector3 min, max;
    min = points[0];
    max = points[0];
    for (const auto &p : points){
        if (p[0] + p[1] + p[2] < min[0] + min[1] + min[2]){
            min = p;
        }
        if (p[0] + p[1] + p[2] > max[0] + max[1] + max[2]){
            max = p;
        }
    }
    std::cout << "The min particle found is " << min << std::endl;
    std::cout << "The max particle found is " << max << std::endl;
}

#endif