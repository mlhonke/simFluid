//
// Created by graphics on 02/05/19.
//

#include "CudaCG.hpp"
#include "cuda_errorcheck.hpp"


CudaCG::CudaCG(int N_in, int n_val){
    M = N = N_in;
    nz = n_val;

    /* Create CUBLAS context */
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Create CUSPARSE context */
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    /* Description of the A matrix*/
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    /* Define the properties of the matrix */
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    printf("conjugateGradientPrecond starting...\n");

    /* This will pick the best possible CUDA capable device */
    cudaDeviceProp deviceProp;
    int devID = 0;
    printf("GPU selected Device ID = %d \n", devID);

    if (devID < 0){
        printf("Invalid GPU device %d selected,  exiting...\n", devID);
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    /* Statistics about the GPU device */
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
    deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
}

// n_val: Number of entries in the matrix.
// N_in: Number of entries.
void CudaCG::set_new_sizes(int N_in, int n_val){
    M = N = N_in;
    nz = n_val;
}

void CudaCG::allocate_memory(){
    /* Allocate required memory */
    checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(scalar_t)));
    checkCudaErrors(cudaMalloc((void **)&d_colp, N*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_rowp, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_valp, N*sizeof(scalar_t)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(scalar_t)));
    checkCudaErrors(cudaMalloc((void **)&d_y, N*sizeof(scalar_t)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(scalar_t)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(scalar_t)));
    checkCudaErrors(cudaMalloc((void **)&d_omega, N*sizeof(scalar_t)));
    checkCudaErrors(cudaMalloc((void **)&d_z, N*sizeof(scalar_t)));

    // Hack for easy sum with dot product using just cublas since asum isn't what we want.
    checkCudaErrors(cudaMalloc((void **)&d_ones, N*sizeof(scalar_t)));
    scalar_t *ones = new scalar_t[nz];
    for (int i = 0; i < nz; i++)
        ones[i] = 1.0;
    checkCudaErrors(cudaMemcpy(d_ones, ones, N*sizeof(scalar_t), cudaMemcpyHostToDevice));
    delete[] ones;
}

void CudaCG::free_memory(){
    /* Free required memory */
    checkCudaErrors(cudaFree(d_col));
    checkCudaErrors(cudaFree(d_colp));
    checkCudaErrors(cudaFree(d_row));
    checkCudaErrors(cudaFree(d_rowp));
    checkCudaErrors(cudaFree(d_val));
    checkCudaErrors(cudaFree(d_valp));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_z));
    checkCudaErrors(cudaFree(d_r));
    checkCudaErrors(cudaFree(d_p));
    checkCudaErrors(cudaFree(d_omega));
    checkCudaErrors(cudaFree(d_ones));
}

void CudaCG::load_diagonal(int* I_in, int *J_in, scalar_t* val_in){
    checkCudaErrors(cudaMemcpy(d_colp, J_in, N*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rowp, I_in, (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_valp, val_in, N*sizeof(scalar_t), cudaMemcpyHostToDevice));
}

void CudaCG::load_matrix(int* I_in, int* J_in, scalar_t* val_in, scalar_t* x_in, scalar_t* rhs_in, int N_in, int n_val){
    // update solver dimensions
    N = N_in;
    nz = n_val;
    // Need more memory obviously get it, otherwise don't get less unless we really need much less to make it worthwhile.
    if (N_in > N_alloc || n_val > nz_alloc || N_in < 0.5*N_alloc || n_val < 0.5*nz_alloc){
//        std::cout << "Reallocating GPU memory due to matrix dimension change." << std::endl;
        if (N_alloc != 0 && nz_alloc != 0) // only free memory if some has been allocated
            free_memory();
        N_alloc = N_in;
        nz_alloc = n_val;
        allocate_memory();
    }

    I = I_in;
    J = J_in;
    val = val_in;
    rhs = rhs_in;
    x = x_in;

    checkCudaErrors(cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, val, nz*sizeof(scalar_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, x, N*sizeof(scalar_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_r, rhs, N*sizeof(scalar_t), cudaMemcpyHostToDevice));
}

void CudaCG::solve(){
//    printf("Convergence of conjugate gradient with diagonal preconditioning: \n");
    k = 0;
    r0 = 0;
    scalar_t rn2  = 1.0;

    cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatone, descr, d_val, d_row, d_col, d_x, &floatzero, d_omega);
    scalar_t negone = -1.0;
    cublasDaxpy(cublasHandle, N, &negone, d_omega, 1, d_r, 1);
    cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, N, &floatone, descr, d_valp, d_rowp, d_colp, d_r, &floatzero, d_p);
    cublasDdot(cublasHandle, N, d_r, 1, d_p, 1, &r1);
    r1 = std::abs(r1);
    cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &rn2);

    while (rn2 > tol*tol && k <= max_iter)
    {
        cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatone, descr, d_val, d_row, d_col, d_p, &floatzero, d_omega);
        cublasDdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot);
        alpha = r1/dot;
        cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
        nalpha = -alpha;
        cublasDaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);

        if (project) {
            scalar_t residual_mean;
            cublasDdot(cublasHandle, N, d_ones, 1, d_r, 1, &residual_mean);
            residual_mean = -residual_mean / (scalar_t) N;
            cublasDaxpy(cublasHandle, N, &residual_mean, d_ones, 1, d_r, 1);
        }

        cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &rn2);

        cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, N, &floatone, descr, d_valp, d_rowp, d_colp, d_r, &floatzero, d_z);
        r0 = r1;
        cublasDdot(cublasHandle, N, d_r, 1, d_z, 1, &r1);
        r1 = std::abs(r1);
        beta = r1/r0;
        cublasDscal(cublasHandle, N, &beta, d_p, 1);
        cublasDaxpy(cublasHandle, N, &floatone, d_z, 1, d_p, 1) ;

        if (project) {
            scalar_t p_mean;
            cublasDdot(cublasHandle, N, d_ones, 1, d_p, 1, &p_mean);
            p_mean = -p_mean / (scalar_t) N;
            cublasDaxpy(cublasHandle, N, &p_mean, d_ones, 1, d_p, 1);
        }

        k++;
    }

    printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

    cudaMemcpy(x, d_x, N*sizeof(scalar_t), cudaMemcpyDeviceToHost);
}

scalar_t CudaCG::get_error(){
/* check result */
    err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i+1]; j++)
        {
            rsum += val[j]*x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
    nErrors += (k > max_iter) ? 1 : 0;
    qaerr1 = err;
    std::cout << "Measured error: " << err << std::endl;

    return err;
}
