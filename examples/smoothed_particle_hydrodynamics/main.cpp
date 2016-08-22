#include <cmath>
#include <iostream>
#include <silo.h>
#include <stdlib.h>
#include <vector>

#include "kernels.h"

int box_indicator(float x, float y)
{
    return (x < 0.25) && (y < 1.0);
}

void place_particles(
    int count,
    float *pos_x,
    float *pos_y,
    float *v_x,
    float *v_y,
    float hh)
{
    int index = 0;

    for (float x = 0; x < 1; x += hh) {
        for (float y = 0; y < 1; y += hh) {
            if (box_indicator(x,y)) {
                pos_x[index] = x;
                pos_y[index] = y;
                v_x[index] = 0;
                v_y[index] = 0;
                ++index;
            }
        }
    }
}

int count_particles(float hh)
{
    int ret = 0;

    for (float x = 0; x < 1; x += hh) {
        for (float y = 0; y < 1; y += hh) {
            ret += box_indicator(x,y);
        }
    }

    return ret;
}

void normalize_mass(float *mass, int n, float *rho, sim_param_t params)
{
    float rho0 = params.rho0;
    float rho_squared_sum = 0;
    float rho_sum = 0;

    for (int i = 0; i < n; ++i) {
        rho_squared_sum += rho[i] * rho[i];
        rho_sum += rho[i];
    }

    *mass = *mass * rho0 * rho_sum / rho_squared_sum;
}

void dump_time_step(int cycle, int n, float* pos_x, float* pos_y)
{
    DBfile *dbfile = NULL;
    char filename[100];
    sprintf(filename, "output%04d.silo", cycle);
    dbfile = DBCreate(filename, DB_CLOBBER, DB_LOCAL,
                      "simulation time step", DB_HDF5);

    float *coords[] = {(float*)pos_x, (float*)pos_y};
    DBPutPointmesh(dbfile, "pointmesh", 2, coords, n,
                   DB_FLOAT, NULL);

    DBClose(dbfile);
}

int main(int argc, char** argv)
{
    sim_param_t params;
    params.dt = 1e-4;
    params.h = 2e-2;
    params.rho0 = 1000;
    params.k = 1e3;
    params.mu = 0.1;
    params.g = 9.8;

    float hh = params.h / 1.3;
    int count = count_particles(hh);

    std::vector<float> rho_vec(count);
    std::vector<float> pos_x_vec(count);
    std::vector<float> pos_y_vec(count);
    std::vector<float> vh_x_vec(count);
    std::vector<float> vh_y_vec(count);
    std::vector<float> v_x_vec(count);
    std::vector<float> v_y_vec(count);
    std::vector<float> a_x_vec(count);
    std::vector<float> a_y_vec(count);

    place_particles(count, pos_x_vec.data(), pos_y_vec.data(), v_x_vec.data(), v_y_vec.data(), hh);
    float mass = 1;
    compute_density(count, rho_vec.data(), pos_x_vec.data(), pos_y_vec.data(), params.h, mass);
    normalize_mass(&mass, count, rho_vec.data(), params);

    int num_steps = 20000;
    int io_period = 15;

    for (int t = 0; t < num_steps; ++t) {
        if ((t % io_period) == 0) {
            dump_time_step(t, count, pos_x_vec.data(), pos_y_vec.data());
        }

        compute_density(count, rho_vec.data(), pos_x_vec.data(), pos_y_vec.data(), params.h, mass);
        compute_accel(count,   rho_vec.data(), pos_x_vec.data(), pos_y_vec.data(), v_x_vec.data(), v_y_vec.data(), a_x_vec.data(), a_y_vec.data(), mass, params);

        leapfrog(
            count,
            pos_x_vec.data(),
            pos_y_vec.data(),
            v_x_vec.data(),
            v_y_vec.data(),
            vh_x_vec.data(),
            vh_y_vec.data(),
            a_x_vec.data(),
            a_y_vec.data(),
            params.dt);

        reflect_bc(count, pos_x_vec.data(), pos_y_vec.data(), v_x_vec.data(), v_y_vec.data(), vh_x_vec.data(), vh_y_vec.data());
    }

    dump_time_step(num_steps, count, pos_x_vec.data(), pos_y_vec.data());

    return 0;
}
