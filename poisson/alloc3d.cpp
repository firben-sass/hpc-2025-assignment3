#include <stdlib.h>
#include <omp.h>
#include <cstdio>

#include "alloc3d.h"


double *** malloc_3d(int m, int n, int k) {

    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    double ***p = (double***) malloc(m * sizeof(double **) +
                                     m * n * sizeof(double *));
    if (p == NULL) {
        return NULL;
    }

    for(int i = 0; i < m; i++) {
        p[i] = (double **) p + m + i * n ;
    }

    double *a = (double*) malloc(m * n * k * sizeof(double));
    if (a == NULL) {
	free(p);
	return NULL;
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            p[i][j] = a + (i * n * k) + (j * k);
        }
    }

    return p;
}

void free_3d(double ***p) {
    free(p[0][0]);
    free(p);
}




double *** d_malloc_3d(int m, int n, int k, double **a_d, int dev_num) {

    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    double ***p = (double***)omp_target_alloc(m * sizeof(double **) +
                                     m * n * sizeof(double *), dev_num);
    if (p == NULL) {
        return NULL;
    }


    #pragma omp target is_device_ptr(p)
    {
    for(int i = 0; i < m; i++) {
        p[i] = (double **) p + m + i * n ;
    }
    }


    double *a = (double*)omp_target_alloc(m * n * k * sizeof(double), dev_num);
    if (a == NULL) {
	d_free_3d(p, a, dev_num);
	return NULL;
    }


    #pragma omp target is_device_ptr(p, a)
    {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            p[i][j] = a + (i * n * k) + (j * k);
        }
    }
    }

    
    *a_d = a;

    return p;
}



void d_free_3d(double ***p, double *a, int dev_num) {
    omp_target_free(a, dev_num);
    omp_target_free(p, dev_num);
}
