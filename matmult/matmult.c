#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <math.h>


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
    /*
    Dimensions: m,n,k
    alpha: 1.0
    beta: 0.0
    A: Matrix A, dimensions m*k
    B: Matrix B, dimensions k*n
    C: Matrix C, dimensions m*n

    For the input we give the second dimension of the matrices.
    */
}

/* Permutations of mnk
1. mnk
2. mkn
3. nmk
4. nkm
5. kmn
6. knm
*/

// 1. mnk permutation
void matmult_mnk(int m, int n, int k,double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

// 2. mkn permutation
void matmult_mkn(int m, int n, int k,double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
        }
    }

    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

// 3. nmk permutation
void matmult_nmk(int m, int n, int k,double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
        }
    }

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            for (int l = 0; l < k; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

// 4. nkm permutation
void matmult_nkm(int m, int n, int k,double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
        }
    }

    for (int j = 0; j < n; j++) {
        for (int l = 0; l < k; l++) {
            for (int i = 0; i < m; i++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

// 5. kmn permutation
void matmult_kmn(int m, int n, int k,double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
        }
    }

    for (int l = 0; l < k; l++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

// 6. knm permutation
void matmult_knm(int m, int n, int k,double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
        }
    }

    for (int l = 0; l < k; l++) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}


/* Blocked version */
void matmult_blk(int m,int n,int k, double **A, double **B, double **C, int bs){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;  // Initialize the result element to zero
        }
    }

    for (int i0 = 0; i0 < m; i0 += bs) {
        for (int j0 = 0; j0 < n; j0 += bs) {
            for (int l0 = 0; l0 < k; l0 += bs) {
                for (int i = i0; i < min( i0 + bs, m); i++) {
                    for (int j = j0; j < min( j0 + bs, n); j++) {
                        for (int l = l0; l < min( l0 + bs, k); l++) {
                            C[i][j] += A[i][l] * B[l][j];
                        }
                    }
                }
            }
        }
    }
}