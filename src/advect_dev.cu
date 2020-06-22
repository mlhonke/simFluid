#include "advect_dev.cuh"

// dummy function to make it easy to place break points since I don't have cuda-gdb in a UI.
__device__ void cu_break_here(){

}

__host__ __device__ int ccti(int i, int j, int k, int n_rows, int n_cols){
    return k*(n_cols*n_rows) + j*n_rows + i;
}

__device__ scalar_t culerp(scalar_t x, scalar_t x1, scalar_t x2, scalar_t Q1, scalar_t Q2){
    scalar_t t = (x - x1) / (x2 - x1);

    return Q1*(1.0-t) + t*Q2;
}

__device__ scalar_t cubilerp(scalar_t x, scalar_t y, scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2, scalar_t* vals){
    scalar_t L1 = culerp(x, x1, x2, vals[0], vals[1]);
    scalar_t L2 = culerp(x, x1, x2, vals[2], vals[3]);
    scalar_t r = culerp(y, y1, y2, L1, L2);

    return r;
}

__device__ scalar_t cutrilerp(CUVEC::Vec3d x, CUVEC::Vec3d x1, CUVEC::Vec3d x2, scalar_t* vals){
    scalar_t L1 = cubilerp(x[0], x[1], x1[0], x1[1], x2[0], x2[1], &vals[0]);
    scalar_t L2 = cubilerp(x[0], x[1], x1[0], x1[1], x2[0], x2[1], &vals[4]);
    return culerp(x[2], x1[2], x2[2], L1, L2);
}

__device__ scalar_t cu_grid_trilerp(const CUVEC::Vec3d &p, const scalar_t *v, SimParams& C, CUVEC::Vec3d offset){
    return cu_grid_trilerp(p, v, C.grid_w, C.grid_h, C.grid_d, C.dx, offset);
}

__device__ scalar_t cu_grid_trilerp(const CUVEC::Vec3d &p, const scalar_t *v, int n_rows, int n_cols, int n_slices, scalar_t const dx, CUVEC::Vec3d offset){
    CUVEC::Vec3d x = p + offset*dx; // offset position to match velocity grid.
    int i = (int) (x[0] / dx);
    int j = (int) (x[1] / dx);
    int k = (int) (x[2] / dx);

    if (i < 0){
        i = 0;
    } else if (i > n_rows-2){
        i = n_rows-2;
    }

    if (j < 0){
        j = 0;
    } else if (j > n_cols-2){
        j = n_cols-2;
    }

    if (k < 0){
        k = 0;
    } else if (k > n_slices-2){
        k = n_slices-2;
    }

    CUVEC::Vec3d x1(i*dx, j*dx, k*dx);
    CUVEC::Vec3d x2((i+1)*dx, (j+1)*dx, (k+1)*dx);
    scalar_t vals[8];
    vals[0] = v[ccti(i, j, k, n_rows, n_cols)];
    vals[1] = v[ccti(i+1, j, k, n_rows, n_cols)];
    vals[2] = v[ccti(i, j+1, k, n_rows, n_cols)];
    vals[3] = v[ccti(i+1, j+1, k, n_rows, n_cols)];
    vals[4] = v[ccti(i, j, k+1, n_rows, n_cols)];
    vals[5] = v[ccti(i+1, j, k+1, n_rows, n_cols)];
    vals[6] = v[ccti(i, j+1, k+1, n_rows, n_cols)];
    vals[7] = v[ccti(i+1, j+1, k+1, n_rows, n_cols)];
    scalar_t r = cutrilerp(x, x1, x2, vals);
    return r;
}

__device__ CUVEC::Vec3d cu_vel_trilerp(CUVEC::Vec3d p, scalar_t* vx, scalar_t* vy, scalar_t* vz, SimParams &C){
    CUVEC::Vec3d vel;
    vel[0] = cu_grid_trilerp(p, vx, C.grid_w+1, C.grid_h, C.grid_d, C.dx, CUVEC::Vec3d(0.5, 0, 0));
    vel[1] = cu_grid_trilerp(p, vy, C.grid_w, C.grid_h+1, C.grid_d, C.dx, CUVEC::Vec3d(0, 0.5, 0));
    vel[2] = cu_grid_trilerp(p, vz, C.grid_w, C.grid_h, C.grid_d+1, C.dx, CUVEC::Vec3d(0, 0, 0.5));
    return vel;
}

__device__ scalar_t cu_cerp(scalar_t x, const CUVEC::Vec4d &X, const CUVEC::Vec4d &Q, bool clamp){
    scalar_t val = 0;
    scalar_t s = (x - X[1])/(X[2] - X[1]);

    val += ((-1.0/3.0)*s + 0.5*s*s - (1.0/6.0)*s*s*s)*Q[0];
    val += (1.0 - s*s + 0.5*(s*s*s - s))*Q[1];
    val += (s + 0.5*(s*s - s*s*s))*Q[2];
    val += ((1.0/6.0)*(s*s*s - s))*Q[3];

    if(clamp){
        if (val > CUVEC::max(Q)){
            val = CUVEC::max(Q);
        } else if (val < CUVEC::min(Q)){
            val = CUVEC::min(Q);
        }
    }

    return val;
}

__device__ scalar_t cu_bicerp(scalar_t x, scalar_t y, CUVEC::Vec4d X, CUVEC::Vec4d Y, scalar_t **Q, bool clamp){
    CUVEC::Vec4d Qy;
    for (int i = 0; i < 4; i++){
        CUVEC::Vec4d Qx = {Q[0][i], Q[1][i], Q[2][i], Q[3][i]};
        Qy[i] = cu_cerp(x, X, Qx, clamp);
    }

    return cu_cerp(y, Y, Qy, clamp);
}

__device__ scalar_t cu_grid_bilerp(scalar_t x, scalar_t y, const scalar_t *q, int n_rows, int n_cols, int k, const scalar_t dx){
    int i = (int) (x * (1/dx));
    int j = (int) (y * (1/dx));
    int nx = (int) n_rows;
    int ny = (int) n_cols;

    if (i < 0){
        i = 0;
    } else if (i > nx-2){
        i = nx-2;
    }

    if (j < 0){
        j = 0;
    } else if (j > ny-2){
        j = ny-2;
    }

    scalar_t x1 = i * dx;
    scalar_t y1 = j * dx;
    scalar_t x2 = (i+1) * dx;
    scalar_t y2 = (j+1) * dx;
//    printf("The rows and cols are %d and %d\n", n_rows, n_cols);

    scalar_t Q[4];
    Q[0] = q[ccti(i, j, k, n_rows, n_cols)];
    Q[1] = q[ccti(i+1, j, k, n_rows, n_cols)];
    Q[2] = q[ccti(i, j+1, k, n_rows, n_cols)];
    Q[3] = q[ccti(i+1, j+1, k, n_rows, n_cols)];

    return cubilerp(x, y, x1, y1, x2, y2, Q);
}


__device__ scalar_t cu_grid_bicerp(scalar_t x, scalar_t y, const scalar_t *q, bool clamp, int n_rows, int n_cols, int k, const scalar_t dx){
    scalar_t retval;
    int i = (int) (x * (1/dx));
    int j = (int) (y * (1/dx));
    int nx = n_rows;
    int ny = n_cols;

    if (i > 0 && j > 0 && i < nx-2 && j < ny-2){
        CUVEC::Vec4d X;
        X[0] = (i-1) * dx;
        X[1] = i * dx;
        X[2] = (i+1) * dx;
        X[3] = (i+2) * dx;

        CUVEC::Vec4d Y;
        Y[0] = (j-1) * dx;
        Y[1] = j * dx;
        Y[2] = (j+1) * dx;
        Y[3] = (j+2) * dx;

        scalar_t* Q[4];
        scalar_t Q0[4], Q1[4], Q2[4], Q3[4];
        Q[0] = Q0; Q[1] = Q1; Q[2] = Q2; Q[3] = Q3;

        int si = 0;
        for (int s = j-1; s <= j+2; s++){
            int ri = 0;
            for (int r = i-1; r <= i+2; r++){
                Q[ri][si] = q[ccti(r, s, k, n_rows, n_cols)];
                ri++;
            }
            si++;
        }

        retval = cu_bicerp(x, y, X, Y, Q, clamp);
    } else { // around boundaries switch to linear interpolation (not enough points to use cubic interpolation)
        retval = cu_grid_bilerp(x, y, q, n_rows, n_cols, k, dx);
    }

    return retval;
}

__device__ scalar_t cu_grid_tricerp(const CUVEC::Vec3d &X, const scalar_t *q, bool clamp, SimParams &C) {
    return cu_grid_tricerp(X, q, clamp, C.grid_w, C.grid_h, C.grid_d, C.dx);
}

__device__ scalar_t cu_grid_tricerp(const CUVEC::Vec3d &X, const scalar_t *q, bool clamp, int n_rows, int n_cols, int n_slices, const scalar_t dx){
    int k = (int) (X[2] * (1.0/dx));
    int nz = n_slices;
    CUVEC::Vec4d L;
    CUVEC::Vec4d Z;

    if (k > 0 && k < nz-2){
        int i = 0;
        for (int l = k-1; l <= k+2; l++) {
            L[i] = cu_grid_bicerp(X[0], X[1], q, clamp, n_rows, n_cols, l, dx);
            Z[i] = l*dx;
            i++;
        }

        return cu_cerp(X[2], Z, L, clamp);
    } else {
        CUVEC::Vec3d offset(0,0,0);
        return cu_grid_trilerp(X, q, n_rows, n_cols, n_slices, dx, offset);
    }

}

__device__ CUVEC::Vec3d cu_vel_tricerp(const CUVEC::Vec3d &X, const scalar_t *u, const scalar_t *v, const scalar_t *w, SimParams &C){
    CUVEC::Vec3d R = {0.0, 0.0, 0.0};

    CUVEC::Vec3d offset = {0.5, 0.0, 0.0};
    R[0] = cu_grid_tricerp(X+offset*C.dx, u, true, C.grid_w+1, C.grid_h, C.grid_d, C.dx);
    offset = {0.0, 0.5, 0.0};
    R[1] = cu_grid_tricerp(X+offset*C.dx, v, true, C.grid_w, C.grid_h+1, C.grid_d, C.dx);
    offset = {0.0, 0.0, 0.5};
    R[2] = cu_grid_tricerp(X+offset*C.dx, w, true, C.grid_w, C.grid_h, C.grid_d+1, C.dx);

    return R;
}

__device__ int cu_get_label(CUVEC::Vec3d X, int* label, const scalar_t dx, CUVEC::Vec3i dim, bool tell_me_boundary = false) {
    scalar_t eps = 0.000001;
    scalar_t ipart = X[0] * (1/dx);
    scalar_t jpart = X[1] * (1/dx);
    scalar_t kpart = X[2] * (1/dx);
    int i = lround(ipart);
    int j = lround(jpart);
    int k = lround(kpart);
    int il = i+1;
    int jl = j+1;
    int kl = k+1;
    ipart -= i;
    jpart -= j;
    kpart -= k;

    // FIXME: Is grid_w, grid_h, grid_d and dim equivalent here?!
    if (il >= 0 && il < dim[0] && jl >= 0 && jl < dim[1] && kl >= 0 && kl < dim[2]) {
        int cell1 = label[ccti(il, jl, kl, dim[0], dim[1])];
        int cell2 = -1; // -1 for not on an edge.
        if (fabs(ipart - 0.5) < eps) {
            cell2 = label[ccti(il+1, jl, kl, dim[0], dim[1])];
        } else if (fabs(ipart + 0.5) < eps) {
            cell2 = label[ccti(il-1, jl, kl, dim[0], dim[1])];
        } else if (fabs(jpart - 0.5) < eps) {
            cell2 = label[ccti(il, jl+1, kl, dim[0], dim[1])];
        } else if (fabs(jpart + 0.5) < eps) {
            cell2 = label[ccti(il, jl-1, kl, dim[0], dim[1])];
        } else if (fabs(kpart - 0.5) < eps) {
            cell2 = label[ccti(il, jl, kl+1, dim[0], dim[1])];
        } else if (fabs(kpart + 0.5) < eps) {
            cell2 = label[ccti(il, jl, kl-1, dim[0], dim[1])];
        }

        if (tell_me_boundary){
            if ((cell1 == 0 && cell2 == 1) || (cell1 == 1 && cell2 == 0)){ //Air fluid boundary
                return 10;
            }
            if ((cell1 == 1 && cell2 == 2) || (cell1 == 2 && cell2 == 1)){ //Fluid solid boundary
                return 12;
            }
        }

        if ( cell1 == 2 || cell2 == 2 ){
            return 2;
        } else if ( cell1 == 1 || cell2 == 1){
            return 1;
        } else {
            return 0;
        }

    } else {
        return 2; //TODO: Figure out a better way of saying "invalid points" maybe with a bool return or something.
    }
}

__global__ void cu_advect_RK3(  scalar_t *q,
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
                                SimParams *C){
    CUVEC::Vec3d k1, k2, k3, X_s, X_p;
    // REMINDER generate a copy of the to-be-advected data.
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    int n_cells = n_rows*n_cols*n_slices;
    int i = id % n_rows;
    int j = (id / n_rows) % n_cols;
    int k = id / (n_cols * n_rows);

    if (id < n_cells) {
        // Initial value at time step n.
        CUVEC::Vec3d I = {(double) i, (double) j, (double) k};
        X_s = C->dx * (I - offset);

        k1 = cu_vel_tricerp(X_s, u, v, w, *C);
        k2 = cu_vel_tricerp(X_s - 0.5 * dt * k1, u, v, w, *C);
        k3 = cu_vel_tricerp(X_s - 0.75 * dt * k2, u, v, w, *C);

        // Value at time step n+1.
        X_p = X_s - (2.0 / 9.0) * dt * k1 - (3.0 / 9.0) * dt * k2 - (4.0 / 9.0) * dt * k3;
        CUVEC::Vec3d X_p_offset = X_p + offset * C->dx;
        q_prime[id] = cu_grid_tricerp(X_p_offset, q, do_clamp_q, n_rows, n_cols, n_slices, C->dx);
    }
    // REMINDER set newly advected values to be current simulation values after advection is complete.
}

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
                              SimParams *C){
    cu_advect_RK3<<<n_blocks, threads_per_block>>>(q, q_prime, u, v, w, offset, dt, do_clamp_q, n_rows, n_cols, n_slices, C);
}