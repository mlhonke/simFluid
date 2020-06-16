//
// Created by graphics on 02/05/19.
//

#ifndef FERRO3D_CUDACG_HPP
#define FERRO3D_CUDACG_HPP
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sim_params.hpp"
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>       // helper for CUDA error checking


class CudaCG{
public:
    CudaCG(int N_in, int n_val);
    void load_matrix(int* I_in, int* J_in, scalar_t* val_in, scalar_t* x_in, scalar_t* rhs_in, int N_in, int n_val);
    void load_diagonal(int* I_in, int *J_in, scalar_t* val_in);
    void solve();
    scalar_t get_error();
    bool project = false;
    void free_memory();
    void allocate_memory();
    void set_new_sizes(int N_in, int n_val);
    int k;

private:
    cublasHandle_t cublasHandle = 0;
    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t descr = 0;

    const int max_iter = 10000;
    int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL, N_alloc = 0, nz_alloc = 0;
    int *d_col, *d_row;
    int qatest = 0;
    const scalar_t tol = 1e-16;
    scalar_t *x, *rhs;
    scalar_t r0, r1, alpha, beta;
    scalar_t *d_val, *d_x;
    scalar_t *d_r, *d_p, *d_omega, *d_y, *d_ones, *d_z;
    scalar_t *d_valp;
    int *d_colp, *d_rowp;
    scalar_t *val = NULL;
    scalar_t *d_valsILU0;
    scalar_t *valsILU0;
    scalar_t rsum, diff, err = 0.0;
    scalar_t qaerr1, qaerr2 = 0.0;
    scalar_t dot, numerator, denominator, nalpha;
    const scalar_t floatone = 1.0;
    const scalar_t floatzero = 0.0;

    int nErrors = 0;
};


#endif //FERRO3D_CUDACG_HPP
