#include <stdlib.h>


void define_u(double *** u, int N)
{
    for (int i = 0; i < N+2; i++)
    {
        for (int j = 0; j < N+2; j++)
        {
            for (int k = 0; k < N+2; k++)
            {
                if (i == N+1 || j == 0 || j == N+1 || k == 0 || k == N+1)
                    u[i][j][k] = 20;
                else
                    u[i][j][k] = 0;
            }
        }
    }

    return;
}

void define_f(double *** f, int N)
{
    double delta = 2.0 / N;
    double * point = (double *) malloc(3 * sizeof(double));

    for (int i = 0; i < N+2; i++)
    {
        for (int j = 0; j < N+2; j++)
        {
            for (int k = 0; k < N+2; k++)
            {
                point[0] = (i - (N+2)/2) * delta;
                point[1] = (j - (N+2)/2) * delta;
                point[2] = (k - (N+2)/2) * delta;

                int x_cond = point[0] >= -1 && point[0] <= -3.0/8.0;
                int y_cond = point[1] >= -1 && point[1] <= -1.0/2.0;
                int z_cond = point[2] >= -2.0/3.0 && point[2] <= -0;
                if (x_cond && y_cond && z_cond)
                    f[i][j][k] = 200;
                else
                    f[i][j][k] = 0;
            }
        }
    }

    return;
}