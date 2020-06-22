//
// Created by graphics on 06/12/18.
//

#include "fd_math.hpp"
#include "interpolate.hpp"

void calc_grad(const CubeX& P, CubeX& Gx, CubeX& Gy, CubeX& Gz, scalar_t dx){
    for (unsigned int k = 0; k < Gx.n_slices; k++) {
        for (unsigned int j = 0; j < Gx.n_cols; j++) {
            for (unsigned int i = 0; i < Gx.n_rows; i++) {
                scalar_t PX1, PX2, PY1, PY2, PZ1, PZ2;
                scalar_t nx, ny, nz;
                nx = 2; // default case, we are two grid cells distance for central difference
                ny = 2;
                nz = 2;

                if (i != 0) {
                    PX1 = P(i - 1, j, k);
                } else {
                    PX1 = P(i, j, k); // Don't use central difference on boundary. Not the worse thing in the world.
                    nx = 1;
                }
                if (i < Gx.n_rows - 1) {
                    PX2 = P(i + 1, j,k);
                } else {
                    PX2 = P(i, j,k);
                    nx = 1;
                }
                if (j != 0) {
                    PY1 = P(i, j - 1,k);
                } else {
                    PY1 = P(i, j,k);
                    ny = 1;
                }
                if (j < Gx.n_cols - 1) {
                    PY2 = P(i, j + 1,k);
                } else {
                    PY2 = P(i, j, k);
                    ny = 1;
                }
                if (k != 0) {
                    PZ1 = P(i, j,k-1);
                } else {
                    PZ1 = P(i, j,k);
                    nz = 1;
                }
                if (k < Gx.n_slices - 1) {
                    PZ2 = P(i, j, k+1);
                } else {
                    PZ2 = P(i, j, k);
                    nz = 1;
                }

                Gx(i, j, k) = (PX2 - PX1) / (nx * dx);
                Gy(i, j, k) = (PY2 - PY1) / (ny * dx);
                Gz(i, j, k) = (PZ2 - PZ1) / (nz * dx);
            }
        }
    }
}

std::vector<scalar_t> get_neighbours_lerped(const Vector3 &pos, const CubeX& Q, scalar_t dx){
    std::vector<scalar_t> sides;

    sides.push_back(grid_trilerp({pos(0)+dx, pos(1), pos(2)}, Q, dx));
    sides.push_back(grid_trilerp({pos(0)-dx, pos(1), pos(2)}, Q, dx));
    sides.push_back(grid_trilerp({pos(0), pos(1)+dx, pos(2)}, Q, dx));
    sides.push_back(grid_trilerp({pos(0), pos(1)-dx, pos(2)}, Q, dx));
    sides.push_back(grid_trilerp({pos(0), pos(1), pos(2)+dx}, Q, dx));
    sides.push_back(grid_trilerp({pos(0), pos(1), pos(2)-dx}, Q, dx));

    return sides;
}

std::vector<scalar_t> get_neighbours_cerped(const Vector3 &pos, const CubeX& Q, scalar_t dx){
    std::vector<scalar_t> sides;

    sides.push_back(grid_tricerp({pos(0)+dx, pos(1), pos(2)}, Q, dx));
    sides.push_back(grid_tricerp({pos(0)-dx, pos(1), pos(2)}, Q, dx));
    sides.push_back(grid_tricerp({pos(0), pos(1)+dx, pos(2)}, Q, dx));
    sides.push_back(grid_tricerp({pos(0), pos(1)-dx, pos(2)}, Q, dx));
    sides.push_back(grid_tricerp({pos(0), pos(1), pos(2)+dx}, Q, dx));
    sides.push_back(grid_tricerp({pos(0), pos(1), pos(2)-dx}, Q, dx));

    return sides;
}

Vector3 get_grad_lerped(const Vector3& pos, const CubeX& Q, scalar_t dx){
    Vector3 grad;
    std::vector<scalar_t> sides = get_neighbours_lerped(pos, Q, dx);

    grad(0) = (sides[0]-sides[1])/(2.0*dx);
    grad(1) = (sides[2]-sides[3])/(2.0*dx);
    grad(2) = (sides[4]-sides[5])/(2.0*dx);

    return grad;
}

Vector3 get_grad_cerped(const Vector3& pos, const CubeX& Q, scalar_t dx){
    Vector3 grad;
    std::vector<scalar_t> sides = get_neighbours_cerped(pos, Q, dx);

    grad(0) = (sides[0]-sides[1])/(2.0*dx);
    grad(1) = (sides[2]-sides[3])/(2.0*dx);
    grad(2) = (sides[4]-sides[5])/(2.0*dx);

    return grad;
}

//TODO: Use appropriate interpolation technique
scalar_t get_laplace_lerped(const Vector3 &pos, const CubeX &Q, scalar_t dx){
    std::vector<scalar_t> sides = get_neighbours_lerped(pos, Q, dx);
    scalar_t center = grid_trilerp(pos, Q, dx); //should be zero theoretically for surface tension use.

    return (sides[0] + sides[1] + sides[2] + sides[3] + sides[4] + sides[5] - 6*center)/(dx*dx);
}

scalar_t get_distance(const Vector3& a, const Vector3& b){
    return std::sqrt((a(0)-b(0))*(a(0)-b(0)) + (a(1)-b(1))*(a(1)-b(1)) + (a(2)-b(2))*(a(2)-b(2)));
}

scalar_t get_Langevin(scalar_t alpha){
    return 1/tanh(alpha) - 1/alpha;
}

scalar_t make_non_zero(scalar_t val){
    scalar_t eps_val = 10E-10;
    if (std::abs(val) < eps_val){
        return bisgn(val)*eps_val;
    } else {
        return val;
    }
}

scalar_t d_error(int i, int j, int k){
    printf("The derivative requested at %d, %d, %d, could not be calculated.", i, j, k);

    return 0;
}

scalar_t d_xx(int i, int j, int k, const CubeX& Q, scalar_t dx){
    int nx = Q.n_rows;

    if (i > 0 && i < nx-1){
        return (Q(i+1, j, k) - 2.0*Q(i,j,k) + Q(i-1,j,k))/(dx*dx);
    } else if (i == 0){
        return (Q(i+2, j,k) - 2.0*Q(i+1,j,k) + Q(i,j,k))/(dx*dx);
    } else if (i == nx - 1){
        return (Q(i,j,k) - 2.0*Q(i-1,j,k)+Q(i-2,j,k))/(dx*dx);
    }

    return d_error(i, j, k);
}

scalar_t d_yy(int i, int j, int k, const CubeX& Q, scalar_t dx){
    int ny = Q.n_cols;

    if (j > 0 && j < ny-1){
        return (Q(i, j+1,k) - 2.0*Q(i,j,k) + Q(i,j-1,k))/(dx*dx);
    } else if (j == 0){
        return (Q(i, j+2,k) - 2.0*Q(i,j+1,k) + Q(i,j,k))/(dx*dx);
    } else if (j == ny - 1){
        return (Q(i,j,k) - 2.0*Q(i,j-1,k)+Q(i,j-2,k))/(dx*dx);
    }

    return d_error(i, j, k);
}

scalar_t d_zz(int i, int j, int k, const CubeX& Q, scalar_t dx){
    int nz = Q.n_slices;

    if (k > 0 && k < nz-1){
        return (Q(i, j,k+1) - 2.0*Q(i,j,k) + Q(i,j,k-1))/(dx*dx);
    } else if (k == 0){
        return (Q(i, j,k+2) - 2.0*Q(i,j,k+1) + Q(i,j,k))/(dx*dx);
    } else if (k == nz - 1){
        return (Q(i,j,k) - 2.0*Q(i,j,k-1)+Q(i,j,k-2))/(dx*dx);
    }

    return d_error(i, j, k);
}

scalar_t d_x(int i, int j, int k, const CubeX& Q, scalar_t dx){
    int nx = Q.n_rows;

    if (i > 0 && i < nx-1){
        return 0.5*(Q(i+1, j,k) - Q(i-1,j,k))/(dx);
    } else if (i == 0){
        return (Q(i+1,j,k) - Q(i, j,k))/(dx);
    } else if (i == nx - 1){
        return (Q(i,j,k) - Q(i-1,j,k))/(dx);
    }

    return d_error(i, j, k);
}

scalar_t d_y(int i, int j, int k, const CubeX& Q, scalar_t dx) {
    int ny = Q.n_cols;

    if (j > 0 && j < ny - 1) {
        return 0.5 * (Q(i, j + 1,k) - Q(i, j - 1,k)) / (dx);
    } else if (j == 0) {
        return (Q(i, j + 1,k) - Q(i, j,k)) / (dx);
    } else if (j == ny - 1) {
        return (Q(i, j,k) - Q(i, j - 1,k)) / (dx);
    }

    return d_error(i, j, k);
}

scalar_t d_z(int i, int j, int k, const CubeX& Q, scalar_t dx) {
    int nz = Q.n_slices;

    if (k > 0 && k < nz - 1) {
        return 0.5 * (Q(i, j,k+1) - Q(i, j,k-1)) / (dx);
    } else if (k == 0) {
        return (Q(i, j,k+1) - Q(i, j,k)) / (dx);
    } else if (k == nz - 1) {
        return (Q(i, j,k) - Q(i, j, k-1)) / (dx);
    }

    return d_error(i, j, k);
}

scalar_t d_xy(int i, int j, int k, const CubeX& Q, scalar_t dx){
    if (j > 0 && j < Q.n_cols - 1){
        scalar_t dp = d_x(i, j+1, k, Q, dx);
        scalar_t dn = d_x(i, j-1, k, Q, dx);
        return 0.5*((dp-dn)/dx);
    } else if (j == 0){
        scalar_t dp = d_x(i, j+1, k, Q, dx);
        scalar_t dn = d_x(i, j, k, Q, dx);
        return (dp-dn)/dx;
    } else if (j == Q.n_cols-1){
        scalar_t dp = d_x(i, j, k, Q, dx);
        scalar_t dn = d_x(i, j-1, k, Q, dx);
        return (dp-dn)/dx;
    }

    return d_error(i, j, k);
}

scalar_t d_xz(int i, int j, int k, const CubeX& Q, scalar_t dx){
    if (k > 0 && k < Q.n_slices - 1){
        scalar_t dp = d_x(i, j, k+1, Q, dx);
        scalar_t dn = d_x(i, j, k-1, Q, dx);
        return 0.5*((dp-dn)/dx);
    } else if (k == 0){
        scalar_t dp = d_x(i, j, k+1, Q, dx);
        scalar_t dn = d_x(i, j, k, Q, dx);
        return (dp-dn)/dx;
    } else if (k == Q.n_slices-1){
        scalar_t dp = d_x(i, j, k, Q, dx);
        scalar_t dn = d_x(i, j, k-1, Q, dx);
        return (dp-dn)/dx;
    }

    return d_error(i, j, k);
}

scalar_t d_yz(int i, int j, int k, const CubeX& Q, scalar_t dx){
    if (k > 0 && k < Q.n_slices - 1){
        scalar_t dp = d_y(i, j, k+1, Q, dx);
        scalar_t dn = d_y(i, j, k-1, Q, dx);
        return 0.5*((dp-dn)/dx);
    } else if (k == 0){
        scalar_t dp = d_y(i, j, k+1, Q, dx);
        scalar_t dn = d_y(i, j, k, Q, dx);
        return (dp-dn)/dx;
    } else if (k == Q.n_slices-1){
        scalar_t dp = d_y(i, j, k, Q, dx);
        scalar_t dn = d_y(i, j, k-1, Q, dx);
        return (dp-dn)/dx;
    }

    return d_error(i, j, k);
}

scalar_t calc_mesh_volume(const std::vector<Vector3> &x, const std::vector<Vector3ui> &tri){
    scalar_t V = 0;

    for (const auto &t : tri){
        V += (1.0/6.0)*arma::dot(x[t[0]], arma::cross(x[t[1]], x[t[2]]));
    }

    return V;
}

scalar_t expanded_round(scalar_t val, scalar_t pad_amt){
    scalar_t intpart;
    scalar_t dec = std::modf(val, &intpart);

    if (dec >= 0.5-pad_amt){
        intpart++;
    }

    return intpart;
}