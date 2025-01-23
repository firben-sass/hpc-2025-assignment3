#include <math.h>
#include <omp.h>
#include "alloc3d.h"


void jacobi_cpu(double *** u_0, double *** u_1, double *** f, int N, int P)
{
    double factor = 1.0 / 6.0;
    double delta = 2.0 / N;

    for (int p = 0; p < P; p++)
    {
        for (int i = 1; i < N + 1; i++)
        {
            for (int j = 1; j < N + 1; j++)
            {
                for (int k = 1; k < N + 1; k++)
                {
                    u_1[i][j][k] = (u_0[i-1][j][k] + u_0[i+1][j][k] + 
                                    u_0[i][j-1][k] + u_0[i][j+1][k] + 
                                    u_0[i][j][k-1] + u_0[i][j][k+1] + 
                                    delta * delta * f[i][j][k]) * factor;
                }
            }
        }

        // Swap the pointers to update u_0 for the next iteration
        double ***temp = u_0;
        u_0 = u_1;
        u_1 = temp;
    }
}

void jacobi_gpu(double ***u_0, double ***u_1, double ***f, int N, int P) {
    double factor = 1.0 / 6.0;
    double delta = 2.0 / N;

    // Map data to the GPU before computation
    #pragma omp target data map(to: u_0[0:N+2][0:N+2][0:N+2], f[0:N+2][0:N+2][0:N+2]) \
                            map(tofrom: u_1[0:N+2][0:N+2][0:N+2])
    {
        for (int p = 0; p < P; p++) {
            // Perform computation on the GPU
            #pragma omp target teams distribute parallel for collapse(3)
            for (int i = 1; i < N + 1; i++) {
                for (int j = 1; j < N + 1; j++) {
                    for (int k = 1; k < N + 1; k++) {
                        u_1[i][j][k] = (u_0[i-1][j][k] + u_0[i+1][j][k] + 
                                        u_0[i][j-1][k] + u_0[i][j+1][k] + 
                                        u_0[i][j][k-1] + u_0[i][j][k+1] + 
                                        delta * delta * f[i][j][k]) * factor;
                    }
                }
            }
            //#pragma omp taskwait
            // Swap the pointers to update u_0 for the next iteration
            // #pragma omp target update to(u_1[0:N+2][0:N+2][0:N+2])
            double ***temp = u_0;
            u_0 = u_1;
            u_1 = temp;
        }
    }
}

int jacobi_gpu_norm(double ***u_0, double ***u_1, double ***f, int N, int P, double threshold) {
    double factor = 1.0 / 6.0;
    double delta = 2.0 / N;
    int p;

    // Map data to the GPU before computation
    #pragma omp target data map(to: u_0[0:N+2][0:N+2][0:N+2], f[0:N+2][0:N+2][0:N+2]) \
                            map(tofrom: u_1[0:N+2][0:N+2][0:N+2])
    {
        double norm = 1000000;
        for (p = 0; p < P; p++) {
            if (norm < threshold)
                break;
            norm = 0;

            #pragma omp target teams distribute parallel for reduction(+:norm)
            for (int i = 1; i < N + 1; i++) {
                for (int j = 1; j < N + 1; j++) {
                    for (int k = 1; k < N + 1; k++) {
                        u_1[i][j][k] = (u_0[i-1][j][k] + u_0[i+1][j][k] + 
                                        u_0[i][j-1][k] + u_0[i][j+1][k] + 
                                        u_0[i][j][k-1] + u_0[i][j][k+1] + 
                                        delta * delta * f[i][j][k]) * factor;
                        double diff = u_0[i][j][k] - u_1[i][j][k];
                        norm += diff * diff;
                    }
                }
            }

            // Swap the pointers to update u_0 for the next iteration
            // #pragma omp target update to(u_1[0:N+2][0:N+2][0:N+2])
            double ***temp = u_0;
            u_0 = u_1;
            u_1 = temp;
        }
    }

    return p;
}

int jacobi_cpu_norm(double ***u_0, double ***u_1, double ***f, int N, int P, double threshold) {
    double factor = 1.0 / 6.0;
    double delta = 2.0 / N;
    double norm = 1000000;
    int p;

    for (p = 0; p < P; p++) {
        if (norm < threshold)
            break;
        norm = 0;
        
        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < N + 1; j++) {
                for (int k = 1; k < N + 1; k++) {
                    u_1[i][j][k] = (u_0[i-1][j][k] + u_0[i+1][j][k] + 
                                    u_0[i][j-1][k] + u_0[i][j+1][k] + 
                                    u_0[i][j][k-1] + u_0[i][j][k+1] + 
                                    delta * delta * f[i][j][k]) * factor;
                    double diff = u_0[i][j][k] - u_1[i][j][k];
                    norm += diff * diff;
                }
            }
        }

        // Swap the pointers to update u_0 for the next iteration
        // #pragma omp target update to(u_1[0:N+2][0:N+2][0:N+2])
        double ***temp = u_0;
        u_0 = u_1;
        u_1 = temp;
    }

    return p;
}

int jacobi_target(double ***u0, double ***u1, double ***f, int N, int P) {
    double factor = 1.0 / 6.0;
    double delta = 2.0 / N;

    // pick the default gpu
    int dev_num = omp_get_default_device();

    double ***u0_d, ***f_d, ***u1_d;
    // allocation memory on device
    d_malloc_3d(N, N, N, dev_num);
    d_malloc_3d(N, N, N, dev_num);
    d_malloc_3d(N, N, N, dev_num);

    // copy the data to the "device" (gpu) from the "host" (cpu)
    omp_target_memcpy(u0_d, u0, N * sizeof(double), 0, 0, dev_num, omp_get_initial_device());
    omp_target_memcpy(f_d, f, N * sizeof(double), 0, 0, dev_num, omp_get_initial_device());
   
    for (int p = 0; p < P; p++) {
        #pragma omp target teams loop is_device_ptr(u0_d, u1_d, f_d) collapse(3) device(dev_num)
        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < N + 1; j++) {
                for (int k = 1; k < N + 1; k++) {
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
    omp_target_memcpy(u0_d, u0, N * sizeof(double), 0, 0, omp_get_initial_device(), dev_num);


    // free the memory on the device
    d_free_3d(u0_d, dev_num);
    d_free_3d(f_d, dev_num);
    d_free_3d(u1_d, dev_num);
    
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
    int halfN = N / 2;

    // Device pointers
    double ***u0_d0, ***u1_d0, ***f_d0;
    double ***u0_d1, ***u1_d1, ***f_d1;

    // Allocate memory on both GPUs
    // the +2 is for the halo cells (additional layer of cells to store the neighbouring values that come from the adjacent subgrids)
    d_malloc_3d(halfN + 2, N, N, dev0);
    d_malloc_3d(halfN + 2, N, N, dev0);
    d_malloc_3d(halfN + 2, N, N, dev0);

    d_malloc_3d(halfN + 2, N, N, dev1);
    d_malloc_3d(halfN + 2, N, N, dev1);
    d_malloc_3d(halfN + 2, N, N, dev1);

    // Copy data to GPUs
    omp_target_memcpy(u0_d0[0][0], u0[0][0], (halfN + 1) * N * N * sizeof(double), 0, 0, dev0, omp_get_initial_device());
    omp_target_memcpy(f_d0[0][0], f[0][0], (halfN + 1) * N * N * sizeof(double), 0, 0, dev0, omp_get_initial_device());

    omp_target_memcpy(u0_d1[0][0], u0[halfN][0], (halfN + 1) * N * N * sizeof(double), 0, 0, dev1, omp_get_initial_device());
    omp_target_memcpy(f_d1[0][0], f[halfN][0], (halfN + 1) * N * N * sizeof(double), 0, 0, dev1, omp_get_initial_device());

    for (int p = 0; p < P; p++) {
        // Compute on GPU 0
        #pragma omp target teams distribute parallel for is_device_ptr(u0_d0, u1_d0, f_d0) device(dev0)
        for (int i = 1; i <= halfN; i++) {
            for (int j = 1; j < N - 1; j++) {
                for (int k = 1; k < N - 1; k++) {
                    u1_d0[i][j][k] = (u0_d0[i-1][j][k] + u0_d0[i+1][j][k] +
                                      u0_d0[i][j-1][k] + u0_d0[i][j+1][k] +
                                      u0_d0[i][j][k-1] + u0_d0[i][j][k+1] +
                                      delta * delta * f_d0[i][j][k]) * factor;
                }
            }
        }

        // Compute on GPU 1
        #pragma omp target teams distribute parallel for is_device_ptr(u0_d1, u1_d1, f_d1) device(dev1)
        for (int i = 1; i <= halfN; i++) {
            for (int j = 1; j < N - 1; j++) {
                for (int k = 1; k < N - 1; k++) {
                    u1_d1[i][j][k] = (u0_d1[i-1][j][k] + u0_d1[i+1][j][k] +
                                      u0_d1[i][j-1][k] + u0_d1[i][j+1][k] +
                                      u0_d1[i][j][k-1] + u0_d1[i][j][k+1] +
                                      delta * delta * f_d1[i][j][k]) * factor;
                }
            }
        }

        // Exchange halos between GPU 0 and GPU 1
        omp_target_memcpy(u0_d1[0][0], u1_d0[halfN][0], N * N * sizeof(double), 0, 0, dev1, dev0);
        omp_target_memcpy(u0_d0[halfN + 1][0], u1_d1[1][0], N * N * sizeof(double), 0, 0, dev0, dev1);

        // Swap device pointers
        std::swap(u0_d0, u1_d0);
        std::swap(u0_d1, u1_d1);
    }

    // Copy results back to the host
    omp_target_memcpy(u0[0][0], u0_d0[0][0], (halfN + 1) * N * N * sizeof(double), 0, 0, omp_get_initial_device(), dev0);
    omp_target_memcpy(u0[halfN][0], u0_d1[0][0], (halfN + 1) * N * N * sizeof(double), 0, 0, omp_get_initial_device(), dev1);

    // Free device memory
    d_free_3d(u0_d0, dev0);
    d_free_3d(u1_d0, dev0);
    d_free_3d(f_d0, dev0);

    d_free_3d(u0_d1, dev1);
    d_free_3d(u1_d1, dev1);
    d_free_3d(f_d1, dev1);

    return 0;
}