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
    // double	tolerance = 0.01;
    // double	start_T = 0.0;
    // int		output_type = 3;
    // char    *output_ext    = "";
    // char	output_filename[FILENAME_MAX];
    double 	***u_0 = NULL;
    double  ***u_1 = NULL;
    double  ***f = NULL;

    // char * filename = strrchr(argv[0], '/');
    // if (filename != NULL) {
    //     filename++;
    // } else {
    //     filename = argv[0];
    // }
    // char * output_prefix = filename;

    /* get the paramters from the command line */
    if (argc >= 2)
        N         = atoi(argv[1]);	// grid size
    if (argc >= 3)
        iter_max  = atoi(argv[2]);  // max. no. of iterations
    // if (argc >= 4)
    //     start_T   = atof(argv[4]);  // start T for all inner grid points

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

    define_u(u_0, N);
    define_u(u_1, N);
    define_f(f, N);

    printf("Running Jacobi\n\n");
    jacobi_cpu(u_0, u_1, f, N, iter_max);

    // print_3d(u_1, N);

    // // dump  results if wanted 
    // switch(output_type) {
	// case 0:
	//     // no output at all
	//     break;
	// case 3:
	//     output_ext = ".bin";
	//     sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	//     fprintf(stderr, "Write binary dump to %s: ", output_filename);
	//     print_binary(output_filename, N, u_1);
	//     break;
	// case 4:
	//     output_ext = ".vtk";
	//     sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	//     fprintf(stderr, "Write VTK file to %s: ", output_filename);
	//     print_vtk(output_filename, N, u_1);
	//     break;
	// default:
	//     fprintf(stderr, "Non-supported output type!\n");
	//     break;
    // }

    // de-allocate memory
    free_3d(u_0);
    free_3d(u_1);
    free_3d(f);

    return(0);
}
