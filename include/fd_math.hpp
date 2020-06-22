//
// Created by Michael Honke on 06/12/18.
//

#ifndef FERRO_FD_MATH_H
#define FERRO_FD_MATH_H

#include <set>
#include "sim_params.hpp"
#include "sim_external_types.hpp"

void calc_grad(const CubeX &P, CubeX& Gx, CubeX& Gy, CubeX& Gz, scalar_t dx);
Vector3 get_grad_lerped(const Vector3& pos, const CubeX& Q, scalar_t dx);
Vector3 get_grad_cerped(const Vector3& pos, const CubeX& Q, scalar_t dx);
scalar_t get_laplace_lerped(const Vector3& pos, const CubeX& Q, scalar_t dx);
scalar_t get_distance(const Vector3& a, const Vector3& b);
scalar_t get_Langevin(scalar_t alpha);
scalar_t make_non_zero(scalar_t val);
scalar_t d_xx(int i, int j, int k, const CubeX& Q, scalar_t dx);
scalar_t d_yy(int i, int j, int k, const CubeX& Q, scalar_t dx);
scalar_t d_zz(int i, int j, int k, const CubeX& Q, scalar_t dx);
scalar_t d_x(int i, int j, int k, const CubeX& Q, scalar_t dx);
scalar_t d_y(int i, int j, int k, const CubeX& Q, scalar_t dx);
scalar_t d_z(int i, int j, int k, const CubeX& Q, scalar_t dx);
scalar_t d_xy(int i, int j, int k, const CubeX& Q, scalar_t dx);
scalar_t d_xz(int i, int j, int k, const CubeX& Q, scalar_t dx);
scalar_t d_yz(int i, int j, int k, const CubeX& Q, scalar_t dx);
scalar_t calc_mesh_volume(const std::vector<Vector3> &x, const std::vector<Vector3ui> &tri);
std::vector<scalar_t> get_neighbours_lerped(const Vector3 &pos, const CubeX& Q, scalar_t dx);
scalar_t expanded_round(scalar_t val, scalar_t pad_amt);

template <typename T, typename R, class C = std::less<R>>
inline std::set<R, C> get_neighbours(T &A, unsigned int i, unsigned int j, unsigned int k){
    std::set<R, C> neighbours;

    if (i > 0) {
        neighbours.insert(A(i - 1, j, k));
    }
    if (i < A.n_rows - 1) {
        neighbours.insert(A(i + 1, j, k));
    }
    if (j > 0) {
        neighbours.insert(A(i, j - 1, k));
    }
    if (j < A.n_cols - 1) {
        neighbours.insert(A(i, j + 1, k));
    }
    if (k > 0) {
        neighbours.insert(A(i, j, k - 1));
    }
    if (k < A.n_slices - 1) {
        neighbours.insert(A(i, j, k + 1));
    }

    return neighbours;
}

template <typename T, typename R>
int inline get_neighbours_all(std::array<R, 6> &neighbours, const T &A, unsigned int i, unsigned int j, unsigned int k){
    int n_index = 0;
    if (i > 0) {
        neighbours[n_index++] = A(i - 1, j, k);
    }
    if (i < A.n_rows - 1) {
        neighbours[n_index++] = A(i + 1, j, k);
    }
    if (j > 0) {
        neighbours[n_index++] = A(i, j - 1, k);
    }
    if (j < A.n_cols - 1) {
        neighbours[n_index++] = A(i, j + 1, k);
    }
    if (k > 0) {
        neighbours[n_index++] = A(i, j, k - 1);
    }
    if (k < A.n_slices - 1) {
        neighbours[n_index++] = A(i, j, k + 1);
    }

    return n_index;
}

template <typename T>
void inline get_neighbours_coords(std::array<Vector3ui, 6> &neighbours, const T &A, unsigned int i, unsigned int j, unsigned int k){
    int n_index = 0;
    if (i > 0) {
        neighbours[n_index++] = {i - 1, j, k};
    }
    if (i < A.n_rows - 1) {
        neighbours[n_index++] = {i + 1, j, k};
    }
    if (j > 0) {
        neighbours[n_index++] = {i, j - 1, k};
    }
    if (j < A.n_cols - 1) {
        neighbours[n_index++] = {i, j + 1, k};
    }
    if (k > 0) {
        neighbours[n_index++] = {i, j, k - 1};
    }
    if (k < A.n_slices - 1) {
        neighbours[n_index] = {i, j, k + 1};
    }
}

template <typename T>
bool is_sign_equal(T a, T b){
    if (a >= 0 && b >= 0){
        return true;
    } else if (a < 0 && b < 0){
        return true;
    } else {
        return false;
    }
}

#endif //FERRO_FD_MATH_H
