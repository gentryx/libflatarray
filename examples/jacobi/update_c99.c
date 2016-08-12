#ifdef __ICC
#include <omp.h>
#endif

/**
 * Recommended reference for multi-dimensional array handling in C99
 * by Jeff Hammond:
 *
 *   https://github.com/jeffhammond/HPCInfo/blob/master/c99/array3d.c
 */
void update_c99(double *data_new, const double *data_old, int dim_x, int dim_y, int dim_z)
{
    // cast types here to maintain a C++-compatible signature:
    double (* const restrict grid_old)[dim_y][dim_x] = (double (* const)[dim_y][dim_x])data_old;
    double (*       restrict grid_new)[dim_y][dim_x] = (double (*      )[dim_y][dim_x])data_new;

#pragma omp parallel for schedule(static)
    for (int z = 1; z < (dim_z - 1); ++z) {
        for (int y = 1; y < (dim_y - 1); ++y) {
#ifdef __ICC
#pragma vector always nontemporal
#endif
            for (int x = 1; x < (dim_x - 1); ++x) {
                grid_new[z][y][x] =
                    (grid_old[z - 1][y    ][x    ] +
                     grid_old[z    ][y - 1][x    ] +
                     grid_old[z    ][y    ][x - 1] +
                     grid_old[z    ][y    ][x + 1] +
                     grid_old[z    ][y + 1][x    ] +
                     grid_old[z + 1][y    ][x    ]) * (1.0 / 6.0);
            }
        }
    }
}
