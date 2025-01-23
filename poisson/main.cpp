/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "print.h"
#include <string.h>
#include "alloc3d.h"
#include "define_u_f.h"
#include "poisson.h"
#include <omp.h>
#include <stdbool.h>
#include <math.h>


void print_3d(double *** arr, int N)
{
    // Print array
    for (int i = 0; i < N+2; i++) {
        printf("2D Array at index %d:\n", i);
        for (int j = 0; j < N+2; j++) {
            for (int k = 0; k < N+2; k++) {
                printf("%#.6g ", arr[i][j][k]); // Print each number with a minimum width of 3
            }
            printf("\n"); // Newline for rows
        }
        printf("\n"); // Newline between 2D arrays
    }
}

// Function to check if two 3D arrays of doubles are approximately equal
bool areArraysApproximatelyEqual(double *** arr1, double *** arr2, int N, double tolerance) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                if (fabs(arr1[i][j][k] - arr2[i][j][k]) > tolerance) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Function to copy a 3D array
double ***copy3DArray(double ***source, int N) {
    // Allocate memory for the new 3D array
    double ***copy = (double ***)malloc(N * sizeof(double **));
    for (int i = 0; i < N; i++) {
        copy[i] = (double **)malloc(N * sizeof(double *));
        for (int j = 0; j < N; j++) {
            copy[i][j] = (double *)malloc(N * sizeof(double));
        }
    }

    // Copy data from source to copy
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                copy[i][j][k] = source[i][j][k];
            }
        }
    }

    return copy; // Return the new 3D array
}


int main(int argc, char *argv[]) {

    int 	N = 5;
    int 	iter_max = 1000;
    double  threshold = 0.01;
    int print_arrs = 0;
    int run_standard = 1;
    int run_threshold = 1;
    int run_dalloc = 1;
    double 	***u_0 = NULL;
    double  ***u_1 = NULL;
    double  ***f = NULL;
    double  ***u_1_seq;
    double  ***u_1_seq_norm;
    double  ***u_1_single_gpu;

    double start_time;
    double end_time;
    double arr_test_tol = 0.001;

    /* get the paramters from the command line */
    if (argc >= 2)
        N         = atoi(argv[1]);
    if (argc >= 3)
        iter_max  = atoi(argv[2]);
    if (argc >= 4)
        threshold = atoi(argv[3]);
    if (argc >= 5)
        print_arrs = atoi(argv[4]);
    if (argc >= 6)
        run_standard = atoi(argv[5]);
    if (argc >= 7)
        run_threshold = atoi(argv[6]);
    if (argc >= 8)
        run_dalloc = atoi(argv[7]);

    printf("-------------------\n");
    printf("Allocating memory on CPU:\n");
    start_time = omp_get_wtime();
    // allocate memory
    if ( (u_0 = malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u_0: allocation failed");
        exit(-1);
    }
    if ( (u_1 = malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u_1: allocation failed");
        exit(-1);
    }
    if ( (f = malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array f: allocation failed");
        exit(-1);
    }
    end_time = omp_get_wtime();
    printf("Time taken for allocating memory on CPU: %f seconds\n", end_time - start_time);

    printf("-------------------\n");
    printf("Initializing arrays on CPU:\n");
    start_time = omp_get_wtime();
    define_u(u_0, N);
    define_u(u_1, N);
    define_f(f, N);
    end_time = omp_get_wtime();
    printf("Time taken for initializing arrays on CPU: %f seconds\n", end_time - start_time);

    if (run_standard)
    {
        printf("-------------------\n");
        printf("Running Jacobi:\n");
        start_time = omp_get_wtime();
        jacobi_cpu(u_0, u_1, f, N, iter_max);
        end_time = omp_get_wtime();
        printf("Time taken for CPU: %f seconds\n", end_time - start_time);
        u_1_seq = copy3DArray(u_1, N);
        if (print_arrs)
            print_3d(u_1, N);
        define_u(u_0, N);
        define_u(u_1, N);
        start_time = omp_get_wtime();
        jacobi_gpu(u_0, u_1, f, N, iter_max);
        end_time = omp_get_wtime();
        printf("Time taken for GPU: %f seconds\n", end_time - start_time);
        if (print_arrs)
            print_3d(u_1, N);
        if (areArraysApproximatelyEqual(u_1, u_1_seq, N, arr_test_tol))
            printf("CPU and GPU outputs are IDENTICAL!\n");
        else
            printf("CPU and GPU outputs are DIFFERENT!\n");
        define_u(u_0, N);
        define_u(u_1, N);
    }

    if (run_threshold)
    {
        printf("-------------------\n");
        int iters;
        printf("Running Jacobi with norm:\n");
        start_time = omp_get_wtime();
        iters = jacobi_cpu_norm(u_0, u_1, f, N, iter_max, threshold);
        end_time = omp_get_wtime();
        printf("Time taken for CPU: %f seconds\n", end_time - start_time);
        u_1_seq_norm = copy3DArray(u_1, N);
        define_u(u_0, N);
        define_u(u_1, N);
        printf("Iterations: %d\n", iters);
        start_time = omp_get_wtime();
        iters = jacobi_gpu_norm(u_0, u_1, f, N, iter_max, threshold);
        end_time = omp_get_wtime();
        printf("Time taken for GPU: %f seconds\n", end_time - start_time);
        printf("Iterations: %d\n", iters);
        if (areArraysApproximatelyEqual(u_1, u_1_seq_norm, N, arr_test_tol))
            printf("CPU and GPU outputs are IDENTICAL!\n");
        else
            printf("CPU and GPU outputs are DIFFERENT!\n");
    }

    if (run_dalloc)
    {
        printf("-------------------\n");
        printf("Running Jacobi with d_malloc:\n");
        start_time = omp_get_wtime();
        jacobi_target(u_0, u_1, f, N, iter_max);
        end_time = omp_get_wtime();
        printf("Time taken for CPU: %f seconds\n", end_time - start_time);
        u_1_single_gpu = copy3DArray(u_1, N);
        define_u(u_0, N);
        define_u(u_1, N);
        start_time = omp_get_wtime();
        jacobi_dual_gpu(u_0, u_1, f, N, iter_max);
        end_time = omp_get_wtime();
        printf("Time taken for GPU: %f seconds\n", end_time - start_time);
        if (areArraysApproximatelyEqual(u_1, u_1_single_gpu, N, arr_test_tol))
            printf("target and dual_gpu outputs are IDENTICAL!\n");
        else
            printf("target and dual_gpu outputs are DIFFERENT!\n");
        if (areArraysApproximatelyEqual(u_1, u_1_seq, N, arr_test_tol))
            printf("dual_gpu and seq cpu version outputs are IDENTICAL!\n");
        else
            printf("dual_gpu and seq cpu version outputs are DIFFERENT!\n");
    }

    printf("-------------------\n");

    // de-allocate memory
    free_3d(u_0);
    free_3d(u_1);
    free_3d(f);

    return(0);
}
