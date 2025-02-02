#include <math.h>
#include <omp.h>
#include <cstdio>
#include "alloc3d.h"


void jacobi_cpu(double *** u_0, double *** u_1, double *** f, int N, int P)
{
    double factor = 1.0 / 6.0;
    double delta = 2.0 / (double)N;

    if (P % 2 == 1)
        P++;

    for (int t = 0; t < P; t++)
    {
        // #pragma omp parallel for schedule(static)
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
    double delta = 2.0 / (double)N;

    if (P % 2 == 1)
        P++;

    // Map data to the GPU before computation
    #pragma omp target data map(to: f[0:N+2][0:N+2][0:N+2]) \
                            map(tofrom: u_0[0:N+2][0:N+2][0:N+2], u_1[0:N+2][0:N+2][0:N+2])
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

            // Swap the pointers to update u_0 for the next iteration
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
    #pragma omp target data map(to: f[0:N+2][0:N+2][0:N+2]) \
                            map(tofrom: u_0[0:N+2][0:N+2][0:N+2], u_1[0:N+2][0:N+2][0:N+2])
    {
        double norm = 1000000;
        for (p = 0; p < P; p++) {
            if (norm < threshold && p % 2 == 0)
                break;
            norm = 0;

            #pragma omp target teams distribute parallel for collapse(3) reduction(+:norm)
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
        if (norm < threshold && p % 2 == 0)
            break;
        norm = 0;
        
        #pragma omp parallel for schedule(static)
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