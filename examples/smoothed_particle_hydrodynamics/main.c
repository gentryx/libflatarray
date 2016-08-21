#include <assert.h>
#include <arpa/inet.h>
#include <math.h>
#include <silo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define VERSION_TAG "SPHView01"
#define uint32_t unsigned

typedef struct sim_param_t {
    char* fname; /* File name */
    int nframes; /* Number of frames */
    int npframe; /* Steps per frame */
    float h; /* Particle size */
    float dt; /* Time step */
    float rho0; /* Reference density */
    float k; /* Bulk modulus */
    float mu; /* Viscosity */
    float g; /* Gravity strength */
} sim_param_t;
int get_params(int argc, char** argv, sim_param_t* params);

typedef struct sim_state_t {
    int n; /* Number of particles */
    float mass; /* Particle mass */
    float* restrict rho; /* Densities */
    /* fixme: use x,y,z */
    float* restrict pos_x; /* Positions */
    float* restrict pos_y; /* Positions */
    float* restrict vh_x; /* Velocities (half step) */
    float* restrict vh_y; /* Velocities (half step) */
    float* restrict v_x; /* Velocities (full step) */
    float* restrict v_y; /* Velocities (full step) */
    float* restrict a_x; /* Acceleration */
    float* restrict a_y; /* Acceleration */
} sim_state_t;

sim_state_t* alloc_state(int n)
{
    sim_state_t *state;
    state = malloc(sizeof(sim_state_t));
    state->n     = n;
    state->rho   = malloc(n * sizeof(float));
    state->pos_x = malloc(n * sizeof(float));
    state->pos_y = malloc(n * sizeof(float));
    state->vh_x  = malloc(n * sizeof(float));
    state->vh_y  = malloc(n * sizeof(float));
    state->v_x   = malloc(n * sizeof(float));
    state->v_y   = malloc(n * sizeof(float));
    state->a_x   = malloc(n * sizeof(float));
    state->a_y   = malloc(n * sizeof(float));

    return state;
}

void free_state(sim_state_t* s)
{
    /* fixme: nope, not doing that */
}

void compute_density(sim_state_t* s, sim_param_t* params)
{
    int n = s->n;
    float* restrict rho = s->rho;
    const float* restrict pos_x = s->pos_x;
    const float* restrict pos_y = s->pos_y;
    float h = params->h;
    float h2 = h*h;
    float h8 = ( h2*h2 )*( h2*h2 );
    float C = 4 * s->mass / M_PI / h8;
    memset(rho, 0, n*sizeof(float));
    for (int i = 0; i < n; ++i) {
        rho[i] += 4 * s->mass / M_PI / h2;
        for (int j = i+1; j < n; ++j) {
            float dx = pos_x[i] - pos_x[j];
            float dy = pos_y[i] - pos_y[j];
            float r2 = dx * dx + dy * dy;
            float z = h2 - r2;
            if (z > 0) {
                float rho_ij = C*z*z*z;
                rho[i] += rho_ij;
                rho[j] += rho_ij;
            }
        }
    }
}

void compute_accel(sim_state_t* state, sim_param_t* params)
{
    // Unpack basic parameters
    const float h = params->h;
    const float rho0 = params->rho0;
    const float k = params->k;
    const float mu = params->mu;
    const float g = params->g;
    const float mass = state->mass;
    const float h2 = h*h;
    // Unpack system state
    const float* restrict rho = state->rho;
    const float* restrict pos_x = state->pos_x;
    const float* restrict pos_y = state->pos_y;
    const float* restrict v_x = state->v_x;
    const float* restrict v_y = state->v_y;
    float* restrict a_x = state->a_x;
    float* restrict a_y = state->a_y;
    int n = state->n;
    // Compute density and color
    compute_density(state, params);
    // Start with gravity and surface forces
    for (int i = 0; i < n; ++i) {
        a_x[i] = 0;
        a_y[i] = -g;
    }
    // Constants for interaction term
    float C0 = mass / M_PI / ( (h2)*(h2) );
    float Cp = 15*k;
    float Cv = -40*mu;
    // Now compute interaction forces
    for (int i = 0; i < n; ++i) {
        const float rhoi = rho[i];
        for (int j = i+1; j < n; ++j) {
            float dx = pos_x[i]-pos_x[j];
            float dy = pos_y[i]-pos_y[j];
            float r2 = dx*dx + dy*dy;
            if (r2 < h2) {
                const float rhoj = rho[j];
                float q = sqrt(r2) / h;
                float u = 1 - q;
                float w0 = C0 * u / rhoi / rhoj;
                float wp = w0 * Cp * (rhoi + rhoj - 2 * rho0) * u / q;
                float wv = w0 * Cv;
                float dvx = v_x[i] - v_y[j];
                float dvy = v_y[i] - v_y[j];
                a_x[i] += (wp * dx + wv * dvx);
                a_y[i] += (wp * dy + wv * dvy);
                a_x[j] -= (wp * dx + wv * dvx);
                a_y[j] -= (wp * dy + wv * dvy);
            }
        }
    }
}

static void damp_reflect(
    int which,
    float barrier,
    float* pos_x,
    float* pos_y,
    float* v_x,
    float* v_y,
    float* vh_x,
    float* vh_y)
{
    float *v_which   = (which == 0) ? v_x   : v_y;
    float *vh_which  = (which == 0) ? vh_x  : vh_y;
    float *pos_which = (which == 0) ? pos_x : pos_y;

    // Coefficient of resitiution
    const float DAMP = 0.75;
    // Ignore degenerate cases
    if (fabs(v_which[0]) <= 1e-3)
        return;

    // Scale back the distance traveled based on time from collision
    float tbounce = (pos_which[0] - barrier) / v_which[0];
    pos_x[0] -= v_x[0]*(1-DAMP)*tbounce;
    pos_y[0] -= v_y[0]*(1-DAMP)*tbounce;

    // Reflect the position and velocity
    pos_which[0] = 2 * barrier - pos_which[0];
    v_which[0]   = -v_which[0];
    vh_which[0]  = -vh_which[0];

    // Damp the velocities
    v_x[0] *= DAMP; vh_x[0] *= DAMP;
    v_y[0] *= DAMP; vh_y[0] *= DAMP;
}

static void reflect_bc(sim_state_t* s)
{
    // Boundaries of the computational domain
    const float XMIN = 0.0;
    const float XMAX = 1.0;
    const float YMIN = 0.0;
    const float YMAX = 1.0;
    float* restrict vh_x = s->vh_x;
    float* restrict vh_y = s->vh_y;
    float* restrict v_x = s->v_x;
    float* restrict v_y = s->v_y;
    float* restrict pos_x = s->pos_x;
    float* restrict pos_y = s->pos_y;
    int n = s->n;
    for (int i = 0; i < n; ++i, pos_x += 1, pos_y += 1, v_x += 1, v_y +=1, vh_x += 1, vh_y += 1) {
        if (pos_x[0] < XMIN) damp_reflect(0, XMIN, pos_x, pos_y, v_x, v_y, vh_x, vh_y);
        if (pos_x[0] > XMAX) damp_reflect(0, XMAX, pos_x, pos_y, v_x, v_y, vh_x, vh_y);
        if (pos_y[0] < YMIN) {
            damp_reflect(1, YMIN, pos_x, pos_y, v_x, v_y, vh_x, vh_y);
            if (pos_y[0] < YMIN)
                pos_y[0] = YMIN;
        }
        if (pos_y[0] > YMAX) {
            damp_reflect(1, YMAX, pos_x, pos_y, v_x, v_y, vh_x, vh_y);
            if (pos_y[0] > YMAX)
                pos_y[0] = YMAX;
        }
    }
}

void leapfrog_step(sim_state_t* s, double dt)
{
    const float* restrict a_x = s->a_x;
    const float* restrict a_y = s->a_y;
    float* restrict vh_x = s->vh_x;
    float* restrict vh_y = s->vh_y;
    float* restrict v_x = s->v_x;
    float* restrict v_y = s->v_y;
    float* restrict pos_x = s->pos_x;
    float* restrict pos_y = s->pos_y;
    int n = s->n;
    for (int i = 0; i < n; ++i) vh_x[i] += a_x[i] * dt;
    for (int i = 0; i < n; ++i) vh_y[i] += a_y[i] * dt;
    for (int i = 0; i < n; ++i) v_x[i] = vh_x[i] + a_x[i] * dt / 2;
    for (int i = 0; i < n; ++i) v_x[i] = vh_y[i] + a_y[i] * dt / 2;
    for (int i = 0; i < n; ++i) pos_x[i] += vh_x[i] * dt;
    for (int i = 0; i < n; ++i) pos_y[i] += vh_y[i] * dt;
    reflect_bc(s);
}

void leapfrog_start(sim_state_t* s, double dt)
{
    const float* restrict a_x = s->a_x;
    const float* restrict a_y = s->a_y;
    float* restrict vh_x = s->vh_x;
    float* restrict vh_y = s->vh_y;
    float* restrict v_x = s->v_x;
    float* restrict v_y = s->v_y;
    float* restrict pos_x = s->pos_x;
    float* restrict pos_y = s->pos_y;
    int n = s->n;
    for (int i = 0; i < n; ++i) vh_x[i] = v_x[i] + a_x[i] * dt / 2;
    for (int i = 0; i < n; ++i) vh_y[i] = v_y[i] + a_y[i] * dt / 2;
    for (int i = 0; i < n; ++i) v_x[i] += a_x[i] * dt;
    for (int i = 0; i < n; ++i) v_y[i] += a_y[i] * dt;
    for (int i = 0; i < n; ++i) pos_x[i] += vh_x[i] * dt;
    for (int i = 0; i < n; ++i) pos_y[i] += vh_y[i] * dt;
    reflect_bc(s);
}

typedef int (*domain_fun_t)(float, float);
int box_indicator(float x, float y)
{
    return (x < 0.5) && (y < 0.5);
}
int circ_indicator(float x, float y)
{
    float dx = (x-0.5);
    float dy = (y-0.3);
    float r2 = dx*dx + dy*dy;
    return (r2 < 0.25*0.25);
}

sim_state_t* place_particles(sim_param_t* param,
                             domain_fun_t indicatef)
{
    printf("  place_particlesA\n");
    float h = param->h;
    float hh = h/1.3;
    // Count mesh points that fall in indicated region.
    int count = 0;
    for (float x = 0; x < 1; x += hh)
        for (float y = 0; y < 1; y += hh)
            count += indicatef(x,y);
    printf("  place_particlesB, %i\n", count);
    // Populate the particle data structure
    sim_state_t* s = alloc_state(count);
    printf("  place_particlesC\n");
    int p = 0;
    for (float x = 0; x < 1; x += hh) {
        for (float y = 0; y < 1; y += hh) {
            if (indicatef(x,y)) {
                /* printf("    p: %i, x: %f, y: %f\n", p, x, y); */
                s->pos_x[p] = x;
                s->pos_y[p] = y;
                s->v_x[p] = 0;
                s->v_y[p] = 0;
                ++p;
            }
        }
    }
    printf("  place_particlesD\n");
    return s;
}

void normalize_mass(sim_state_t* s, sim_param_t* param)
{
    s->mass = 1;
    compute_density(s, param);
    float rho0 = param->rho0;
    float rho2s = 0;
    float rhos = 0;
    for (int i = 0; i < s->n; ++i) {
        rho2s += (s->rho[i])*(s->rho[i]);
        rhos += s->rho[i];

    }
    s->mass *= ( rho0*rhos / rho2s );
}
sim_state_t* init_particles(sim_param_t* param)
{
    sim_state_t* s = place_particles(param, box_indicator);
    normalize_mass(s, param);
    return s;
}

void check_state(sim_state_t* s)
{
    for (int i = 0; i < s->n; ++i) {
        float xi = s->pos_x[i];
        float yi = s->pos_y[i];
        assert( xi >= 0 || xi <= 1 );
        assert( yi >= 0 || yi <= 1 );
    }
}

uint32_t htonf(void* data)
{
    return htonl(*(uint32_t*) data);
}

void write_header(FILE* fp, int n)
{
    float scale = 1.0;
    uint32_t nn = htonl((uint32_t) n);

    uint32_t nscale = htonf(&scale);
    fprintf(fp, "%s\n", VERSION_TAG);
    fwrite(&nn, sizeof(nn), 1, fp);
    fwrite(&nscale, sizeof(nscale), 1, fp);
}

void write_frame_data(int cycle, int n, float* pos_x, float* pos_y)
{
    DBfile *dbfile = NULL;
    /* Create a unique filename for the new Silo file*/
    char filename[100];
    sprintf(filename, "output%04d.silo", cycle);
    /* Open the Silo file */
    dbfile = DBCreate(filename, DB_CLOBBER, DB_LOCAL,
                      "simulation time step", DB_HDF5);
    /* Add other Silo calls to write data here. */

    float *coords[] = {(float*)pos_x, (float*)pos_y};

    /* Write a point mesh. */
    DBPutPointmesh(dbfile, "pointmesh", 2, coords, n,
                   DB_FLOAT, NULL);

    /* Close the Silo file. */
    DBClose(dbfile);
}

int main(int argc, char** argv)
{
    printf("pingA\n");
    sim_param_t params;
    if (get_params(argc, argv, &params) != 0)
        exit(-1);
    printf("pingB1\n");
    sim_state_t* state = init_particles(&params);
    printf("pingB2\n");
    FILE* fp = fopen(params.fname, "w");
    int nframes = params.nframes;
    int npframe = params.npframe;
    float dt = params.dt;
    int n = state->n;
    printf("pingC\n");
    /* fixme: */
    /* tic(0); */
    write_header(fp, n);
    printf("pingD1, nframes: %i, n = %i\n", nframes, n);
    write_frame_data(0, n, state->pos_x, state->pos_y);
    printf("pingD2, nframes: %i, n = %i\n", nframes, n);
    compute_accel(state, &params);

    printf("pingD3, nframes: %i, n = %i\n", nframes, n);
    leapfrog_start(state, dt);
    check_state(state);
    for (int frame = 1; frame < nframes; ++frame) {
        printf(" at frame %i, n = %i\n", frame, n);
        for (int i = 0; i < npframe; ++i) {
            compute_accel(state, &params);
            leapfrog_step(state, dt);
            check_state(state);
        }
        write_frame_data(frame, n, state->pos_x, state->pos_y);
    }
    /* fixme: */
    /* printf("Ran in %g seconds\n", toc(0)); */
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
