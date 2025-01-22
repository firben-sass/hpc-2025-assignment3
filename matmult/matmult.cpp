#include<cublas_v2.h>
extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <math.h>
#include<omp.h>

// Definining the min macro
#define min(a, b) ((a) < (b) ? (a) : (b))

// Compile with optimizer: gcc -g -O3 -ffast-math -funroll-loops -fopt-info -c matmult.c
// gcc matmult.c -o matmult.o -I/usr/include -L/usr/lib64/atlas -lsatlas
// Function to perform matrix multiplication
void matmult_nat(int m, int n, int k,double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
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

// void matmult_asy_offload(int m, int n, int k,double **A, double **B, double **C) {
//     // #pragma target data enter ..
//     #define SLABS 4
//     //...
//     #pragma omp parallel for
//     for (int s= 0; s< SLABS; ++s) {
//         // #pragma omp target update to(A... )

//         // Compute C for slap s
        
//         // #pragma omp target update from(C... )
    
//     }
//     // ...
// }


void matmult_asy_offload(int m, int n, int k, double **A, double **B, double **C) {
    #define SLABS 5 
    int slab_size = m / SLABS; 

        #pragma omp target data map(to: B[0:k][0:n], A[0:m][0:k]) map(alloc: C[0:m][0:n])
        for (int s = 0; s < SLABS; ++s) {
            int start_row = s * slab_size;
            int end_row = start_row + slab_size;

            #pragma omp target update map(to:C[start_row:end_row][0:n]) nowait
            #pragma omp parallel for 
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < n; ++j) {
                    double sum=0;
                    for (int p = 0; p < k; ++p) {
                        sum +=A[i][p]*B[p][j];
                    }
                        C[i][j] =sum;
                }
            }
            #pragma omp target update map(from:C[start_row:end_row][0:n])
        }
}


void matmult_mnk_offload(int m, int n, int k,double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
        }
    }
    #pragma omp target teams distribute parallel for collapse(2) 
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
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
        }
    }
    #define bs 5
    #pragma omp target teams distribute parallel for collapse(2) 
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
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
        }
    }
    int i,j,l;
    #pragma omp parallel shared(m,k,n) private(i,j,l)
    {
    for (i = 0; i < m; i++) {
        for (l = 0; l < k; l++) {
            for (j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
        }
    }
}

void matmult_mkn_offload(int m, int n, int k,double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
        }
    }

    #pragma omp target teams loop
    for (int i = 0; i < m; i++) 
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    
}
}