#include<cublas_v2.h>
extern "C" {
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>
    #include <cblas.h>
    #include <math.h>
    #include<omp.h>

    void matmult_nat(int m, int n, int k,double **A, double **B, double **C) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;  
                for (int l = 0; l < k; l++) {
                    C[i][j] += A[i][l] * B[l][j];
                }
            }
        }
    }

    void matmult_lib(int m, int n, int k, double **A, double **B, double **C) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, *A, k, *B, n, 0.0, *C, n);

    }

    void matmult_lib_offload(int m, int n, int k, double **A, double **B, double **C) {
            cublasHandle_t handle;
            cublasCreate(&handle);
            const double alpha = 1.0;
            const double beta = 0.0;

            // Allocate flattened matrices on the device
            double *data_A = (double*)omp_target_alloc(m * k * sizeof(double), omp_get_default_device());
            double *data_B = (double*)omp_target_alloc(k * n * sizeof(double), omp_get_default_device());
            double *data_C = (double*)omp_target_alloc(m * n * sizeof(double), omp_get_default_device());

            if (data_A == NULL || data_B == NULL || data_C == NULL) {
                fprintf(stderr, "Error: Unable to allocate device memory.\n");
                cublasDestroy(handle);
                return;
            }

            // Copy input matrices A and B to the device
            omp_target_memcpy(data_A, A[0], m * k * sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());
            omp_target_memcpy(data_B, B[0], k * n * sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());

            // Initialize C to zero on the device
            omp_target_memcpy(data_C, C[0], m * n * sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());

            // Perform matrix multiplication on the device
            cublasDgemm(handle, 
                        CUBLAS_OP_N, CUBLAS_OP_N, 
                        m, n, k, 
                        &alpha, 
                        data_A, m,  // A is m x k
                        data_B, k,  // B is k x n
                        &beta, 
                        data_C, m); // C is m x n

            // Copy the result matrix C back to the host
            omp_target_memcpy(C[0], data_C, m * n * sizeof(double), 0, 0, omp_get_initial_device(), omp_get_default_device());

            // Free device memory
            omp_target_free(data_A, omp_get_default_device());
            omp_target_free(data_B, omp_get_default_device());
            omp_target_free(data_C, omp_get_default_device());

            // Destroy the cuBLAS handle
            cublasDestroy(handle);
        }


    void matmult_asy_offload(int m, int n, int k, double **A, double **B, double **C) {
        #define SLABS 1
        int slab_size = m / SLABS; 

        #pragma omp target enter data map(to: B[0:k][0:n])

        #pragma omp parallel for
        for (int s = 0; s < SLABS; ++s) {
            int start_row = s * slab_size;
            int end_row = start_row + slab_size;

            #pragma omp target teams distribute parallel for collapse (2)\
            map(to:A[start_row:slab_size][0:k]) map(from:C[start_row:slab_size][0:n]) nowait
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < n; ++j) {
                    double sum=0;
                    for (int p = 0; p < k; ++p) {
                        sum +=A[i][p]*B[p][j];
                    }
                    C[i][j] =sum;
                }
            }
        }
        #pragma omp taskwait
        #pragma omp target exit data map(release:B[0:k][0:n])
    }


    void matmult_mnk_offload(int m, int n, int k,double **A, double **B, double **C) {
            
            #pragma omp target teams distribute parallel for collapse(2) map(to: B[0:k][0:n], A[0:m][0:k]) map(from:C[0:m][0:n])
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double sum =0;
                    for (int l = 0; l < k; l++) {
                        sum+= A[i][l] * B[l][j];
                    }
                    C[i][j] = sum;
                }
            }
    }



    void matmult_blk_offload(int m,int n,int k, double **A, double **B, double **C){
        #define bs 32
        #pragma omp target teams distribute parallel for collapse(2) map(from:C[0:m][0:n]) map(to: B[0:k][0:n], A[0:m][0:k])
        for (int i= 0; i< m; i+= bs) {
            for (int j = 0; j < n; ++j) {
                double sum[bs] = {0};
                if (i+ bs -1 < m) {
                    // for (int i0 = i; i0 < i+bs; i0++) {
                    for (int i0 = 0; i0 < bs; i0++) {
                        for (int l = 0; l < k; l++) {
                            sum[i0] += A[i0+i][l] * B[l][j];
                        }
                        C[i0+i][j] = sum[i0];
                    }
                } else {
                    for (int i0 = i; i0 < m; i0++) {
                        for (int l = 0; l < k; l++) {
                            sum[i0-i] += A[i0][l] * B[l][j];
                        }
                        C[i0][j] = sum[i0-i];
                    }
                }            
            }
        }
    }



    void matmult_mkn_omp(int m, int n, int k,double **A, double **B, double **C) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;  
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            for (int l = 0; l < k; l++) {
                for (int j = 0; j < n; j++) {
                    C[i][j] += A[i][l] * B[l][j];
                }
            }
        }
    }

    void matmult_mkn_offload(int m, int n, int k,double **A, double **B, double **C) {
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;  
            }
        }
        #pragma omp target enter data map(to:B[0:k][0:n], A[0:m][0:k])   
         
        #pragma omp target teams loop map(from:C[0:m][0:n]) is_device_ptr(A,B)  
        for (int i = 0; i < m; i++) {
            for (int l = 0; l < k; l++) {
                for (int j = 0; j < n; j++) {
                    C[i][j] += A[i][l] * B[l][j];
                }
            }
        
        }
        #pragma omp target exit data map(release:B[0:k][0:n], A[0:m][0:k], C[0:m][0:n])
    }
}