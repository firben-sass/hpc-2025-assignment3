#include <math.h>
#include <omp.h>
#include "alloc3d.h"


int jacobi_cpu(double *** u_0, double *** u_1, double *** f, int N, int P)
{
    double factor = 1.0 / 6.0;
    double delta = 2.0 / N;

    #pragma omp parallel
    for (int p = 0; p < P; p++)
    {
        #pragma omp for schedule(static)
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

    return 0;
}

int jacobi_gpu(double ***u_0, double ***u_1, double ***f, int N, int P) {
    double factor = 1.0 / 6.0;
    double delta = 2.0 / N;

    // Map data to the GPU before computation
    #pragma omp target data map(to: u_0[0:N+2][0:N+2][0:N+2], f[0:N+2][0:N+2][0:N+2]) \
                            map(tofrom: u_1[0:N+2][0:N+2][0:N+2])
    {
        for (int p = 0; p < P; p++) {
            // Perform computation on the GPU
            #pragma omp target teams distribute parallel for
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

            // Swap the pointers to update u_0 for the next iteration
            #pragma omp target update to(u_1[0:N+2][0:N+2][0:N+2])
            double ***temp = u_0;
            u_0 = u_1;
            u_1 = temp;
        }
    }

    return 0;
}



int jacobi_target(double ***u_0, double ***u_1, double ***f, int N, int P) {
    double factor = 1.0 / 6.0;
    double delta = 2.0 / N;

    // pick the default gpu
    int dev_num = omp_get_default_device();

    double *u0_d, *f_d, *u1_d;
    // allocation memory on device
    d_malloc_3d(&u0, &u0_d, dev_num);
    d_malloc_3d(&u0, &u0_d, dev_num);
    d_malloc_3d(&u0, &u0_d, dev_num);

    // copy the data to the "device" (gpu) from the "host" (cpu)
    omp_target_memcpy(u0_d, u_0, N * sizeof(double), 0, 0, dev_num, omp_get_initial_device());
    omp_target_memcpy(f_d, f, N * sizeof(double), 0, 0, dev_num, omp_get_initial_device());


   
    for (int p = 0; p < P; p++) {
        #pragma omp target teams loop is_device_ptr(u0_d, u1_d, f_d)
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

        // Swap the pointers to update u_0 for the next iteration
        std::swap(u0_d, u1_d);
    }


    // copy the results back to the "host" (cpu) from the "device" (gpu)
    omp_target_memcpy(u0_d, u_0, N * sizeof(double), 0, 0, omp_get_initial_device(), dev_num);


    // free the memory on the device
    d_free_3d(u0_d, dev_num);
    d_free_3d(f_d, dev_num);
    d_free_3d(u1_d, dev_num);
    
}