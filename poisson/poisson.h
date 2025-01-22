/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

int jacobi_cpu(double ***, double ***, double ***, int, int);

int jacobi_gpu(double ***, double ***, double ***, int, int);

<<<<<<< HEAD
int jacobi_target(double ***, double ***, double ***, int, int); 

// double *** jacobi_LG(double ***, double ***, double ***, int, double *, double *)

=======
>>>>>>> 6e84afe2e87384d722d55c0fe991228c66f1644c
#endif
