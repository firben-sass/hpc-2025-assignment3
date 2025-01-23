#ifndef __ALLOC_3D
#define __ALLOC_3D

double ***malloc_3d(int, int, int);

#define HAS_FREE_3D
void free_3d(double ***array3D);

double ***d_malloc_3d(int, int, int, int);

void d_free_3d(double ***, int);


#endif /* __ALLOC_3D */
