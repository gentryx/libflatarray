#ifdef __ICC
#include <omp.h>
#endif

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <math.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

/**
 * Computes a 2D gaussian filter with a 5x5 stencil accross the YZ-plane.
 */
void filter_c99(double *data_new, const double *data_old, int dim_x, int dim_y, int dim_z)
{
    // cast types here to maintain a C++-compatible signature:
    double (* const restrict grid_old)[dim_y][dim_x] = (double (* const)[dim_y][dim_x])data_old;
    double (*       restrict grid_new)[dim_y][dim_x] = (double (*      )[dim_y][dim_x])data_new;

    double weights[5][5];
    double sum = 0;

    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            double x_component = x - 2;
            double y_component = y - 2;
            weights[y][x] = exp(-0.5 * (x_component * x_component +
                                        y_component * y_component)) / 2 / 3.14159265358979323846;
            sum += weights[y][x];
        }
    }
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            weights[y][x] /= sum;
        }
    }

    // we exploit symmetry to avoid redudant loads of weights:
    double weight_00 = weights[2][2];
    double weight_01 = weights[2][1];
    double weight_02 = weights[2][0];
    double weight_11 = weights[1][1];
    double weight_12 = weights[1][0];
    double weight_22 = weights[0][0];

#pragma omp parallel for schedule(static)
    for (int z = 2; z < (dim_z - 2); ++z) {
        for (int y = 2; y < (dim_y - 2); ++y) {
#ifdef __ICC
#pragma vector always nontemporal
#endif
            for (int x = 0; x < dim_x; ++x) {
                grid_new[z][y][x] =
                    grid_old[z - 2][y - 2][x] * weight_22 +
                    grid_old[z - 2][y - 1][x] * weight_12 +
                    grid_old[z - 2][y + 0][x] * weight_02 +
                    grid_old[z - 2][y + 1][x] * weight_12 +
                    grid_old[z - 2][y + 2][x] * weight_22 +

                    grid_old[z - 1][y - 2][x] * weight_12 +
                    grid_old[z - 1][y - 1][x] * weight_11 +
                    grid_old[z - 1][y + 0][x] * weight_01 +
                    grid_old[z - 1][y + 1][x] * weight_11 +
                    grid_old[z - 1][y + 2][x] * weight_12 +

                    grid_old[z + 0][y - 2][x] * weight_02 +
                    grid_old[z + 0][y - 1][x] * weight_01 +
                    grid_old[z + 0][y + 0][x] * weight_00 +
                    grid_old[z + 0][y + 1][x] * weight_01 +
                    grid_old[z + 0][y + 2][x] * weight_02 +

                    grid_old[z + 1][y - 2][x] * weight_12 +
                    grid_old[z + 1][y - 1][x] * weight_11 +
                    grid_old[z + 1][y + 0][x] * weight_01 +
                    grid_old[z + 1][y + 1][x] * weight_11 +
                    grid_old[z + 1][y + 2][x] * weight_12 +

                    grid_old[z + 2][y - 2][x] * weight_22 +
                    grid_old[z + 2][y - 1][x] * weight_12 +
                    grid_old[z + 2][y + 0][x] * weight_02 +
                    grid_old[z + 2][y + 1][x] * weight_12 +
                    grid_old[z + 2][y + 2][x] * weight_22;
            }
        }
    }
}
