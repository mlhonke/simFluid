//
// Created by graphics on 11/01/19.
//

#include <iomanip>

#include "sim_levelset.hpp"
#include "advect.hpp"
#include "fd_math.hpp"
#include "sim_utils.hpp"

#ifdef USECUDA
SimLevelSet::SimLevelSet(SimParams &C, SimParams* DEV_C, std::array<scalar_t*, 3> &DEV_V) : grid_w(C.grid_w), grid_h(C.grid_h), grid_d(C.grid_d), dx(C.dx),
n_cells(C.n_cells), sim_w(C.sim_w), sim_h(C.sim_h), sim_d(C.sim_d), DEV_V(DEV_V), DEV_C(DEV_C){
    shared_init();
}
#else
SimLevelSet::SimLevelSet(SimParams &C, std::array<CubeX, 3> &V) : grid_w(C.grid_w), grid_h(C.grid_h), grid_d(C.grid_d), dx(C.dx),
                                        n_cells(C.n_cells), sim_w(C.sim_w), sim_h(C.sim_h), sim_d(C.sim_d), V(V), C(C){
    shared_init();
}
#endif

void SimLevelSet::shared_init(){
    LS = CubeX(grid_w, grid_h, grid_d).fill(100);
    LS_grad_x = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
    LS_grad_y = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
    LS_is_interface = CubeXi(grid_w, grid_h, grid_d, arma::fill::zeros);
    LS_K = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
}

bool sort_decreasing(scalar_t i, scalar_t j) {
    return (std::abs(i) < std::abs(j));
}

scalar_t SimLevelSet::get_LS(const Vector3 &pos) {
    return grid_trilerp(pos, LS, dx);
}

scalar_t SimLevelSet::get_LS_cerp(const Vector3 &pos) {
    return grid_tricerp(pos, LS, dx, false);
}

int SimLevelSet::get_index(int i, int j, int k) {
    i = clamp(i, 0, grid_w - 1);
    j = clamp(j, 0, grid_h - 1);
    k = clamp(k, 0, grid_d - 1);
    return k * grid_w * grid_h + j * grid_w + i;
}

int SimLevelSet::get_index(Vector3ui coord) {
    return get_index(coord(0), coord(1), coord(2));
}


void SimLevelSet::advance(int cur_step, scalar_t dt) {
#ifdef USECUDA
    advect_RK3_CUDA(LS, {0, 0, 0}, DEV_V, dt, DEV_C, true, false);
#else
    advect_RK3(LS, {0, 0, 0}, V, dt, C, true, false, true);
#endif

    if (cur_step % 1 == 0) {
        redistance_interface();
    }

    redistance();

//    precalc_fedkiw_curvature();
//    std::cout << "LS curvature\n" << LS_K << std::endl;
}

void SimLevelSet::save_data() {
    LS.save("LS.bin");
}

void SimLevelSet::load_data() {
    LS.load("LS.bin");
}

void SimLevelSet::redistance_interface() {
    LS_is_interface.fill(0);
    CubeX LS_new(grid_w, grid_h, grid_d);
    LS_new.fill(100);

    unsigned int d = 0;
    std::array<scalar_t, 6> neighbours = {0, 0, 0, 0, 0, 0};
    for (auto &home : LS) {
        Vector3ui I = convert_index_to_coords(d, grid_w, grid_h);
        int n_neighbours = get_neighbours_all<CubeX, scalar_t>(neighbours, LS, I(0), I(1), I(2));

        scalar_t min_LS = sgn(home) * 100.0;
        for (int i_neighbour = 0; i_neighbour < n_neighbours; i_neighbour++) {
            if (!is_sign_equal(neighbours[i_neighbour], home)) {
                LS_is_interface(I(0), I(1), I(2)) = 1;
                scalar_t temp_LS = (home / std::abs(home)) * dx * (home / (home - neighbours[i_neighbour]));
                if (std::abs(temp_LS) < std::abs(min_LS)) {
                    min_LS = temp_LS;
                }
            }
        }

        LS_new(I(0), I(1), I(2)) = min_LS;

        d++;
    }

    LS = LS_new;
}

void SimLevelSet::redistance_point(unsigned int i, unsigned int j, unsigned int k) {
    if (!LS_is_interface(i, j, k)) {
        std::array<scalar_t, 6> neighbours = {0, 0, 0, 0, 0, 0};
        int n_neighbours = get_neighbours_all<CubeX, scalar_t>(neighbours, LS, i, j, k);
        scalar_t phi0, phi1, phi2;
        scalar_t home = LS(i, j, k);
        scalar_t flip_sign = 1;

        std::sort(neighbours.begin(), neighbours.begin() + n_neighbours, sort_decreasing);

        phi0 = neighbours[0];
        phi1 = neighbours[1];
        phi2 = neighbours[2];

        if (phi0 < 0 || phi1 < 0 ||
            phi2 < 0) { //If is negative, all should be since on one side of interface only / not on interface.
            phi0 = std::abs(phi0);
            phi1 = std::abs(phi1);
            phi2 = std::abs(phi2);
            flip_sign = -1;
        }

        scalar_t d = phi0 + dx;
        if (d > phi1) {
            d = 0.5 * (phi0 + phi1 + sqrt(2 * dx * dx - (phi1 - phi0) * (phi1 - phi0)));
        }

        if (d > phi2) {
            d = 1.0 / 3.0 * (phi0 + phi1 + phi2 + std::sqrt(
                    std::max(0.0, std::pow(phi0 + phi1 + phi2, 2) -
                                  3.0 * (phi0 * phi0 + phi1 * phi1 + phi2 * phi2 - dx * dx)))
            );
        }

        if (d < std::abs(home)) {
            LS(i, j, k) = flip_sign * d;
        }
    }
}

void SimLevelSet::redistance() {
    // In 2D 4 possible loop orders, 3D 2*4 = 8 possible loop orders.
    for (int k = 0; k < grid_d; k++) {
        for (int j = 0; j < grid_h; j++) {
            for (int i = 0; i < grid_w; i++) {
                redistance_point(i, j, k);
            }
        }
    }
    for (int k = 0; k < grid_d; k++) {
        for (int j = 0; j < grid_h; j++) {
            for (int i = (grid_w - 1); i >= 0; i--) {
                redistance_point(i, j, k);
            }
        }
    }
    for (int k = 0; k < grid_d; k++) {
        for (int j = grid_h - 1; j >= 0; j--) {
            for (int i = grid_w - 1; i >= 0; i--) {
                redistance_point(i, j, k);
            }
        }
    }
    for (int k = 0; k < grid_d; k++) {
        for (int j = grid_h - 1; j >= 0; j--) {
            for (int i = 0; i < grid_w; i++) {
                redistance_point(i, j, k);
            }
        }
    }
    for (int k = grid_d - 1; k >= 0; k--) {
        for (int j = 0; j < grid_h; j++) {
            for (int i = 0; i < grid_w; i++) {
                redistance_point(i, j, k);
            }
        }
    }
    for (int k = grid_d - 1; k >= 0; k--) {
        for (int j = 0; j < grid_h; j++) {
            for (int i = grid_w - 1; i >= 0; i--) {
                redistance_point(i, j, k);
            }
        }
    }
    for (int k = grid_d - 1; k >= 0; k--) {
        for (int j = grid_h - 1; j >= 0; j--) {
            for (int i = grid_w - 1; i >= 0; i--) {
                redistance_point(i, j, k);
            }
        }
    }
    for (int k = grid_d - 1; k >= 0; k--) {
        for (int j = grid_h - 1; j >= 0; j--) {
            for (int i = 0; i < grid_w; i++) {
                redistance_point(i, j, k);
            }
        }
    }
}

void SimLevelSet::initialize_level_set_rectangle(Vector3 I_min, Vector3 I_max) {
    unsigned int d = 0;
    I_min = I_min - 0.5;
    I_max = I_max + 0.5;
    Vector3 X_min = I_min * dx;
    Vector3 X_max = I_max * dx;
    for (auto &valLS : LS) {
        Vector3ui I = convert_index_to_coords(d, grid_w, grid_h);
        Vector3 X = {I(0) * dx, I(1) * dx, I(2) * dx};
//        if (I(2) > I_min(2)+2.0)
//            X(2) += 0.0*scale_w*std::sin(X(0)/(1.25*scale_w))*std::sin(X(1)/(1.25*scale_w));
        scalar_t phi;

        // Inside the box
        if (arma::all(X >= X_min) && arma::all(X <= X_max)) {
            arma::Col<scalar_t>::fixed<6> vals;

            for (int i = 0; i < 3; i++) {
                vals(i) = X_min(i) - X(i);
                vals(i + 3) = X(i) - X_max(i);
            }

            phi = vals.max();
        } else { // Outside the box
            Vector3 P = {0, 0, 0};

            for (int i = 0; i < 3; i++) {
                if (X(i) < X_min(i)) {
                    P(i) = X_min(i);
                } else if (X(i) > X_max(i)) {
                    P(i) = X_max(i);
                } else {
                    P(i) = X(i);
                }
            }

            phi = arma::norm(X - P, 2);
        }

        // Can place multiple objects in the environment and run this function since we only update LS values if they become closer.
        if (std::abs(valLS) > std::abs(phi)) {
            valLS = phi;
        }
        d++;
    }
}

void SimLevelSet::initialize_level_set_circle(const Vector3 &center, scalar_t radius) {
    unsigned int l = 0;
    for (auto &valLS : LS) {
        Vector3ui p = convert_index_to_coords(l, grid_w, grid_h);
        Vector3i p2 = arma::conv_to<arma::Col<int>>::from(p);
        Vector3 d_xyz = p2 - center;
        scalar_t d = arma::norm(d_xyz, 2);
        scalar_t new_val = dx * (d - radius);
        if (new_val < valLS) {
            valLS = new_val;
        }
        l++;
    }
}

scalar_t SimLevelSet::get_height_normal(Vector3 &base, Vector3 &n, scalar_t h_ref, scalar_t dx) {
    return get_height_normal_ls(base, n, h_ref, dx);
}

scalar_t SimLevelSet::get_height_normal_ls(Vector3 &base, Vector3 &n, scalar_t h_ref, scalar_t dx) {
    std::vector<scalar_t> candidates;
    scalar_t prev_ls = grid_tricerp(base, LS, dx, false); // initial ls value
    Vector3 search = base;
    scalar_t h = 0;
    //TODO: Newton search instead of interpolation to reduce diffusion.
    for (int i = 0; i < 7; i++) {
        search += n;
        scalar_t new_ls = grid_tricerp(search, LS, dx, false);
        if (!is_sign_equal(prev_ls, new_ls)) {
            scalar_t theta = prev_ls / (prev_ls - new_ls);
            candidates.push_back(h + theta * dx);
        }
        prev_ls = new_ls;
        h += dx;
    }

    if (candidates.empty()) {
//        std::cout << "Failed to find a height, consider adjusting stencil size." << std::endl;
        return 0;
    }

    scalar_t h_min_diff = 1000;
    scalar_t h_best = 0;
    for (auto hc: candidates) {
        scalar_t h_diff = std::abs(hc - h_ref);
        if (h_diff < h_min_diff) {
            h_best = hc;
            h_min_diff = h_diff;
        }
    }

    return h_best;
}

scalar_t SimLevelSet::get_curvature_height_normal(Vector3 &pos) {
    Vector3 n = get_grad_lerped(pos, LS, dx); // normal
//    std::cout << "Pre Normal " << n(0) << " " << n(1) << std::endl;
    n = arma::normalise(n);
    scalar_t n_eps = 10E-10;

    scalar_t dx_column = dx;
    if (std::abs(std::abs(n(0)) - 1.0) < n_eps || std::abs(std::abs(n(1)) - 1.0) < n_eps ||
        std::abs(std::abs(n(2)) - 1.0) < n_eps) { // Completely along an axis.
    } else {
        scalar_t square_theta = std::abs(atan(n(2) / n(0))); // guaranteed to be valid
        if (square_theta > PI / 4.0) square_theta = PI / 2.0 - square_theta;
        scalar_t dx_xz = dx / cos(square_theta);
//        std::cout << "dx_xz " << dx_xz/dx << std::endl;
//        scalar_t square_phi = std::abs(atan(n(1)/n(0)));
//        if (square_phi > PI/4.0) square_phi = PI/2.0 - square_phi;
//        dx = dx_xz/cos(square_phi);
        scalar_t square_phi = std::abs(atan(n(1) / n(0)));
        if (square_phi > PI / 4.0) square_phi = PI / 2.0 - square_phi;
        scalar_t a = tan(square_phi) * dx;
        dx_column = std::sqrt(dx_xz * dx_xz + a * a);
    }
//    std::cout << "dx frac " << dx / dx << std::endl;
    n = dx_column * n;
//    std::cout << "Normal " << n(0) << " " << n(1) << std::endl;

    Vector3 Base[3][3];
    Vector3 base = pos - 3.0 * n;
    Base[1][1] = base;
    Vector3 v = {dx_column, 0, 0}; // use this vector find a perpendicular vector
    Vector3 np = arma::cross(n, v);
    if (arma::norm(np) <
        0.1 * dx_column * dx_column) { // if the vector is too small then v was fairly parallel, try a different vector for v
        v = {0, dx_column, 0};
        np = arma::cross(n, v);
    }
    np = dx_column * arma::normalise(np);
    Vector3 np2 = arma::cross(np, n); // to get the other perpendicular vector.
    np2 = dx_column * arma::normalise(np2);
    Base[1][2] = base + np;
    Base[1][0] = base - np;
    Base[2][1] = base + np2;
    Base[0][1] = base - np2;
    Base[2][2] = base + np + np2;
    Base[0][2] = base + np - np2;
    Base[2][0] = base - np + np2;
    Base[0][0] = base - np - np2;

    CubeX H(3, 3, 1); // Use cube for compatiblity with my finite difference functions.
    scalar_t h_ref = get_distance(base, pos); // Estimate center column height. Pos should be near/on surface.
//    std::cout << "H_ref " << h_ref << std::endl;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            H(i + 1, j + 1, 0) = get_height_normal(Base[i + 1][j + 1], n, h_ref, dx_column);
        }
    }

    scalar_t hxx = d_xx(1, 1, 0, H, dx_column);
    scalar_t hyy = d_yy(1, 1, 0, H, dx_column);
    scalar_t hx = d_x(1, 1, 0, H, dx_column);
    scalar_t hy = d_y(1, 1, 0, H, dx_column);
    scalar_t K_num = hxx + hyy + hyy * hx * hx + hxx * hy * hy - 2.0 * d_xy(1, 1, 0, H, dx_column) * hx * hy;
    scalar_t K_denom = std::pow(1.0 + hy * hy + hx * hx, 1.5);
    scalar_t K = -K_num / K_denom;

    return 0.5 * K; // seem to be a factor of 2 off compared with expected. I think this matches theoretical now.
}

scalar_t SimLevelSet::get_curvature(Vector3 &pos) {
    scalar_t retval = clamp(1.0 * get_curvature_height_normal(pos), -1.0 / dx, 1.0 / dx);
//    std::cout << retval << " " << pos(0) << " " << pos(1) << " " << pos(2) << std::endl;
//    std::cout << retval << std::endl;
    return retval;
}

scalar_t SimLevelSet::get_curvature_laplace(const Vector3 &pos) {
    return get_laplace_lerped(pos, LS, dx);
}

void SimLevelSet::precalc_fedkiw_curvature() {
    for (int k = 0; k < LS.n_slices; k++) {
        for (int j = 0; j < LS.n_cols; j++) {
            for (int i = 0; i < LS.n_rows; i++) {
                scalar_t phi_y = d_y(i, j, k, LS, dx);
                scalar_t phi_x = d_x(i, j, k, LS, dx);
                scalar_t phi_z = d_z(i, j, k, LS, dx);
                scalar_t phi_xx = d_xx(i, j, k, LS, dx);
                scalar_t phi_yy = d_yy(i, j, k, LS, dx);
                scalar_t phi_zz = d_zz(i, j, k, LS, dx);
                scalar_t T1 = phi_yy * phi_x * phi_x;
                scalar_t T2 = -2.0 * phi_x * phi_y * d_xy(i, j, k, LS, dx);
                scalar_t T3 = phi_xx * phi_y * phi_y;
                scalar_t T4 = phi_x * phi_x * phi_zz;
                scalar_t T5 = -2.0 * phi_x * phi_z * d_xz(i, j, k, LS, dx);
                scalar_t T6 = phi_z * phi_z * phi_xx;
                scalar_t T7 = phi_y * phi_y * phi_zz;
                scalar_t T8 = -2.0 * phi_y * phi_z * d_yz(i, j, k, LS, dx);
                scalar_t T9 = phi_z * phi_z * phi_yy;
                scalar_t num = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9;
                scalar_t denom = std::pow(std::sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z), 3);
//                std::cout << "num " << num << std::endl;
//                std::cout << "denom " << denom << std::endl;
                scalar_t K = num / denom;
                LS_K(i, j, k) = 0.5 * K;
            }
        }
    }
}

scalar_t SimLevelSet::get_curvature_fedkiw(const Vector3 &pos) {
    scalar_t i = pos(0) / dx;
    scalar_t j = pos(1) / dx;
    scalar_t k = pos(2) / dx;

//    std::cout << i << " " << j << " " << k << std::endl;
    return clamp(grid_trilerp(pos, LS_K, dx), -1.0 / dx, 1.0 / dx);
}
