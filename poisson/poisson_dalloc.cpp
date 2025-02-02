#include <math.h>
#include <omp.h>
#include <cstdio>
#include "alloc3d.h"

#include <iostream>



int jacobi_target(double ***u0, double ***u1, double ***f, int N, int P) {
    double factor = 1.0 / 6.0;
    double delta = 2.0 / N;

     // pick the default gpu
    int dev_num = omp_get_default_device();

    // allocation memory on device
    double *a0_d1, ***u0_d = d_malloc_3d((N + 2), (N + 2), (N + 2), &a0_d1, dev_num);
    double *a0_d2, ***f_d = d_malloc_3d((N + 2), (N + 2), (N + 2), &a0_d2, dev_num);
    double *a0_d3, ***u1_d = d_malloc_3d((N + 2), (N + 2), (N + 2), &a0_d3, dev_num);
    
    // copy the data to the "device" (gpu) from the "host" (cpu)
    omp_target_memcpy(a0_d1, u0[0][0], (N + 2) * (N + 2) * (N + 2) * sizeof(double), 0, 0, dev_num, omp_get_initial_device());
    omp_target_memcpy(a0_d2, f[0][0], (N + 2) * (N + 2) * (N + 2) * sizeof(double), 0, 0, dev_num, omp_get_initial_device());
    omp_target_memcpy(a0_d3, u1[0][0], (N + 2) * (N + 2) * (N + 2) * sizeof(double), 0, 0, dev_num, omp_get_initial_device());
   
    for (int p = 0; p < P; p++) {
        #pragma omp target teams distribute parallel for is_device_ptr(u0_d, u1_d, f_d) collapse(3) device(dev_num)
        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < N + 1; j++) {
                for (int k = 1; k < N + 1; k++) {
                    //u0_d[i][j][k] = 5;
                    //u1_d[i][j][k] = 5;
                    u1_d[i][j][k] = (u0_d[i-1][j][k] + u0_d[i+1][j][k] + 
                                    u0_d[i][j-1][k] + u0_d[i][j+1][k] + 
                                    u0_d[i][j][k-1] + u0_d[i][j][k+1] + 
                                    delta * delta * f_d[i][j][k]) * factor;
                }
            }
        }

        // Swap the pointers to update u0 for the next iteration
        std::swap(u0_d, u1_d);
    }

    // copy the results back to the "host" (cpu) from the "device" (gpu)
    omp_target_memcpy(u0[0][0], a0_d1, (N + 2) * (N + 2) * (N + 2) * sizeof(double), 0, 0, omp_get_initial_device(), dev_num);
    omp_target_memcpy(u1[0][0], a0_d3, (N + 2) * (N + 2) * (N + 2) * sizeof(double), 0, 0, omp_get_initial_device(), dev_num);


    // free the memory on the device
    d_free_3d(u0_d, a0_d1, dev_num);
    d_free_3d(f_d, a0_d2, dev_num);
    d_free_3d(u1_d, a0_d3, dev_num);

    return 0;
    
 }



int jacobi_dual_gpu(double ***u0, double ***u1, double ***f, int N, int P) {
    double factor = 1.0 / 6.0;
    double delta = 2.0 / N;

    int dev0 = 0; 
    int dev1 = 1; 

    // Enable peer access between GPUs
    cudaSetDevice(dev0);
    cudaDeviceEnablePeerAccess(dev1, 0);
    cudaSetDevice(dev1);
    cudaDeviceEnablePeerAccess(dev0, 0);
    cudaSetDevice(dev0);

    // Divide the grid between two GPUs
    int halfN = (N + 2) / 2;
    int otherHalfN = (N + 2) - halfN;


    // Allocate memory on both GPUs
    double *a0_du0,***u0_d0 = d_malloc_3d(halfN, (N + 2), (N + 2), &a0_du0, dev0);
    double *a0_df,***f_d0 = d_malloc_3d(halfN, (N + 2), (N + 2), &a0_df, dev0);
    double *a0_du1,***u1_d0 = d_malloc_3d(halfN, (N + 2), (N + 2), &a0_du1, dev0);


    double *a1_du0,***u0_d1 = d_malloc_3d(otherHalfN, (N + 2), (N + 2), &a1_du0, dev1);
    double *a1_df,***f_d1 = d_malloc_3d(otherHalfN, (N + 2), (N + 2), &a1_df, dev1);
    double *a1_du1,***u1_d1 = d_malloc_3d(otherHalfN, (N + 2), (N + 2), &a1_du1, dev1);

    // Copy data to GPUs
    omp_target_memcpy(a0_du0, u0[0][0], (halfN) * (N + 2) * (N + 2) * sizeof(double), 0, 0, dev0, omp_get_initial_device());
    omp_target_memcpy(a0_df, f[0][0], (halfN) * (N + 2) * (N + 2) * sizeof(double), 0, 0, dev0, omp_get_initial_device());
    omp_target_memcpy(a0_du1, u1[0][0], (halfN) * (N + 2) * (N + 2) * sizeof(double), 0, 0, dev0, omp_get_initial_device());

    omp_target_memcpy(a1_du0, u0[halfN][0], (otherHalfN) * (N + 2) * (N + 2) * sizeof(double), 0, 0, dev1, omp_get_initial_device());
    omp_target_memcpy(a1_df, f[halfN][0], (otherHalfN) * (N + 2) * (N + 2) * sizeof(double), 0, 0, dev1, omp_get_initial_device());
    omp_target_memcpy(a1_du1, u1[halfN][0], (otherHalfN) * (N + 2) * (N + 2) * sizeof(double), 0, 0, dev0, omp_get_initial_device());

    for (int p = 0; p < P; p++) {
        // Compute on GPU 0
        omp_set_default_device(dev0);
        #pragma omp target teams distribute parallel for is_device_ptr(u0_d0, u1_d0, f_d0, u0_d1) collapse(3) nowait //device(dev0)
        for (int i = 1; i < halfN; i++) {
            for (int j = 1; j < N + 1; j++) {
                for (int k = 1; k < N + 1; k++) {
                    if (i < halfN - 1){
                        u1_d0[i][j][k] = (u0_d0[i-1][j][k] + u0_d0[i+1][j][k] +
                                      u0_d0[i][j-1][k] + u0_d0[i][j+1][k] +
                                      u0_d0[i][j][k-1] + u0_d0[i][j][k+1] +
                                      delta * delta * f_d0[i][j][k]) * factor;
                    }
                    else{
                        u1_d0[i][j][k] = (u0_d0[i-1][j][k] + u0_d1[0][j][k] +
                                      u0_d0[i][j-1][k] + u0_d0[i][j+1][k] +
                                      u0_d0[i][j][k-1] + u0_d0[i][j][k+1] +
                                      delta * delta * f_d0[i][j][k]) * factor;

                    }
                }
            }
        }
        
        // Compute on GPU 1
        omp_set_default_device(dev1);
        #pragma omp target teams distribute parallel for is_device_ptr(u0_d1, u1_d1, f_d1, u0_d0) collapse(3) nowait //device(dev1)
        for (int i = 0; i < otherHalfN-1; i++) {
            for (int j = 1; j < N + 1; j++) {
                for (int k = 1; k < N + 1; k++) {
                    if (i > 0) {
                        u1_d1[i][j][k] = (u0_d1[i-1][j][k] + u0_d1[i+1][j][k] +
                                      u0_d1[i][j-1][k] + u0_d1[i][j+1][k] +
                                      u0_d1[i][j][k-1] + u0_d1[i][j][k+1] +
                                      delta * delta * f_d1[i][j][k]) * factor;
                    }
                    else{
                        u1_d1[i][j][k] = (u0_d0[halfN-1][j][k] + u0_d1[i+1][j][k] +
                                      u0_d1[i][j-1][k] + u0_d1[i][j+1][k] +
                                      u0_d1[i][j][k-1] + u0_d1[i][j][k+1] +
                                      delta * delta * f_d1[i][j][k]) * factor;
                    }
                }
            }
        }
        #pragma omp taskwait
        // Swap device pointers
        std::swap(u0_d0, u1_d0);
        std::swap(u0_d1, u1_d1);
    }


    // Copy results back to the host
    omp_target_memcpy(u0[0][0], a0_du0, (halfN) * (N + 2) * (N + 2) * sizeof(double), 0, 0, omp_get_initial_device(), dev0);
    omp_target_memcpy(u0[halfN][0], a1_du0, (otherHalfN) * (N + 2) * (N + 2) * sizeof(double), 0, 0, omp_get_initial_device(), dev1);
    
    omp_target_memcpy(u1[0][0], a0_du1, (halfN) * (N + 2) * (N + 2) * sizeof(double), 0, 0, omp_get_initial_device(), dev0);
    omp_target_memcpy(u1[halfN][0], a1_du1, (otherHalfN) * (N + 2) * (N + 2) * sizeof(double), 0, 0, omp_get_initial_device(), dev1);

    // Free device memory
    d_free_3d(u0_d0, a0_du0, dev0);
    d_free_3d(f_d0, a0_df, dev0);
    d_free_3d(u1_d0, a0_du1, dev0);

    d_free_3d(u0_d1, a1_du0, dev1);
    d_free_3d(f_d1, a1_df, dev1);
    d_free_3d(u1_d1, a1_du1, dev1);

    return 0;
}