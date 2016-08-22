#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "kernels.h"

int main(int argc, char** argv)
{
    sim_param_t params;
    if (get_params(argc, argv, &params) != 0)
        exit(-1);
    sim_state_t* state = init_particles(&params);
    FILE* fp = fopen(params.fname, "w");
    int nframes = params.nframes;
    int npframe = params.npframe;
    float dt = params.dt;
    int n = state->n;
    write_frame_data(0, n, state->pos_x, state->pos_y);
    compute_accel(state, &params);

    leapfrog_start(state, dt);
    for (int frame = 1; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            compute_accel(state, &params);
            leapfrog_step(state, dt);
        }
        write_frame_data(frame, n, state->pos_x, state->pos_y);
    }
    fclose(fp);
    free_state(state);
}

static void default_params(sim_param_t* params)
{
    params->fname = "run.out";
    params->nframes = 400;
    params->npframe = 15;
    params->dt = 1e-4;
    params->h = 2e-2;
    params->rho0 = 1000;
    params->k = 1e3;
    params->mu = 0.1;
    params->g = 9.8;
}

static void print_usage()
{
    sim_param_t param;
    default_params(&param);
    fprintf(stderr,
            "nbody\n"
            "\t-h: print this message\n"
            "\t-o: output file name (%s)\n"

            "\t-F: number of frames (%d)\n"
            "\t-f: steps per frame (%d)\n"
            "\t-t: time step (%e)\n"
            "\t-s: particle size (%e)\n"
            "\t-d: reference density (%g)\n"
            "\t-k: bulk modulus (%g)\n"
            "\t-v: dynamic viscosity (%g)\n"
            "\t-g: gravitational strength (%g)\n",
            param.fname, param.nframes, param.npframe,
            param.dt, param.h, param.rho0,
            param.k, param.mu, param.g);
}

int get_params(int argc, char** argv, sim_param_t* params)
{
    extern char* optarg;
    char* optstring = "ho:F:f:t:s:d:k:v:g:";
    int c;
#define get_int_arg(c, field)                   \
    case c: params->field = atoi(optarg); break
#define get_flt_arg(c, field)                           \
    case c: params->field = (float) atof(optarg); break
    default_params(params);
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
        case 'h':
            print_usage();
            return -1;
        case 'o':
            strcpy(params->fname = malloc(strlen(optarg)+1), optarg);
            break;
            get_int_arg('F', nframes);
            get_int_arg('f', npframe);
            get_flt_arg('t', dt);
            get_flt_arg('s', h);
            get_flt_arg('d', rho0);
            get_flt_arg('k', k);
            get_flt_arg('v', mu);
            get_flt_arg('g', g);
        default:
            fprintf(stderr, "Unknown option\n");
            return -1;
        }
    }
    return 0;
}
