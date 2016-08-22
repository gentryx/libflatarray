#include <cmath>
#include <iostream>
#include <silo.h>
#include <stdlib.h>
#include <vector>

#include "kernels.h"

int box_indicator(float x, float y)
{
    return (x < 0.5) && (y < 0.5);
}

void place_particles(
    sim_state_t *state,
    int count,
    float hh)
{
    int index = 0;

    for (float x = 0; x < 1; x += hh) {
        for (float y = 0; y < 1; y += hh) {
            if (box_indicator(x,y)) {
                state->pos_x[index] = x;
                state->pos_y[index] = y;
                state->v_x[index] = 0;
                state->v_y[index] = 0;
                ++index;
            }
        }
    }
}

int count_particles(float hh)
{
    int ret = 0;

    for (float x = 0; x < 1; x += hh)
        for (float y = 0; y < 1; y += hh)
            ret += box_indicator(x,y);

    return ret;
}


void normalize_mass(sim_state_t* s, sim_param_t params)
{
    s->mass = 1;
    compute_density(s->n, s->rho, s->pos_x, s->pos_y, params.h, s->mass);
    float rho0 = params.rho0;
    float rho2s = 0;
    float rhos = 0;
    for (int i = 0; i < s->n; ++i) {
        rho2s += (s->rho[i])*(s->rho[i]);
        rhos += s->rho[i];

    }
    s->mass *= ( rho0*rhos / rho2s );
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

    sim_state_t state;
    std::vector<float> rho_vec(count);
    std::vector<float> pos_x_vec(count);
    std::vector<float> pos_y_vec(count);
    std::vector<float> vh_x_vec(count);
    std::vector<float> vh_y_vec(count);
    std::vector<float> v_x_vec(count);
    std::vector<float> v_y_vec(count);
    std::vector<float> a_x_vec(count);
    std::vector<float> a_y_vec(count);

    state.n = count;
    state.rho = rho_vec.data();
    state.pos_x = pos_x_vec.data();
    state.pos_y = pos_y_vec.data();
    state.vh_x = vh_x_vec.data();
    state.vh_y = vh_y_vec.data();
    state.v_x = v_x_vec.data();
    state.v_y = v_y_vec.data();
    state.a_x = a_x_vec.data();
    state.a_y = a_y_vec.data();

    place_particles(&state, count, hh);
    normalize_mass(&state, params);

    int num_steps = 6000;
    int io_period = 15;

    for (int t = 0; t < num_steps; ++t) {
        if ((t % io_period) == 0) {
            dump_time_step(t, state.n, state.pos_x, state.pos_y);
        }

        compute_density(state.n, state.rho, state.pos_x, state.pos_y, params.h, state.mass);
        compute_accel(&state, params);
        leapfrog(&state, params.dt);
    }

    dump_time_step(num_steps, state.n, state.pos_x, state.pos_y);

    return 0;
}
