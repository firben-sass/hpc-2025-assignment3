/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

void jacobi_cpu(double ***, double ***, double ***, int, int);

void jacobi_gpu(double ***, double ***, double ***, int, int);

int jacobi_cpu_norm(double ***, double ***, double ***, int, int, double);

int jacobi_gpu_norm(double ***, double ***, double ***, int, int, double);

#endif
