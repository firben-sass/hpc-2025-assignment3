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

int main(int argc, char *argv[]) {

    int 	N = 5;
    int 	iter_max = 1000;
    double  threshold = 0.01;
    double 	***u_0 = NULL;
    double  ***u_1 = NULL;
    double  ***f = NULL;

    double start_time;
    double end_time;

    /* get the paramters from the command line */
    if (argc >= 2)
        N         = atoi(argv[1]);
    if (argc >= 3)
        iter_max  = atoi(argv[2]);
    if (argc >= 4)
        threshold = atoi(argv[3]);


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


    printf("-------------------\n");
    printf("Running Jacobi:\n");
    start_time = omp_get_wtime();
    jacobi_cpu(u_0, u_1, f, N, iter_max);
    end_time = omp_get_wtime();
    printf("Time taken for CPU: %f seconds\n", end_time - start_time);
    start_time = omp_get_wtime();
    jacobi_gpu(u_0, u_1, f, N, iter_max);
    end_time = omp_get_wtime();
    printf("Time taken for GPU: %f seconds\n", end_time - start_time);

    printf("-------------------\n");
    int iters;
    printf("Running Jacobi with norm:\n");
    start_time = omp_get_wtime();
    iters = jacobi_cpu_norm(u_0, u_1, f, N, iter_max, threshold);
    end_time = omp_get_wtime();
    printf("Time taken for CPU: %f seconds\n", end_time - start_time);
    printf("Iterations: %s\n", iters);
    start_time = omp_get_wtime();
    iters = jacobi_gpu_norm(u_0, u_1, f, N, iter_max, threshold);
    end_time = omp_get_wtime();
    printf("Time taken for GPU: %f seconds\n", end_time - start_time);
    printf("Iterations: %s\n", iters);

    printf("-------------------\n");


    // print_3d(u_1, N);

    // de-allocate memory
    free_3d(u_0);
    free_3d(u_1);
    free_3d(f);

    return(0);
}
