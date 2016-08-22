#include <assert.h>

#include "kernels.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void compute_density(int n, float *rho, float *pos_x, float *pos_y, float h, float mass)
{
    float h_squared = h * h;
    float h_pow_8 = h_squared * h_squared * h_squared * h_squared;
    float C = 4 * mass / M_PI / h_pow_8;

    for (int i = 0; i < n; ++i) {
        rho[i] = 4 * mass / M_PI / h_squared;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float delta_x = pos_x[i] - pos_x[j];
            float delta_y = pos_y[i] - pos_y[j];
            float dist_squared = delta_x * delta_x + delta_y * delta_y;
            float overlap = h_squared - dist_squared;

            if (overlap > 0) {
                float rho_ij = C * overlap * overlap * overlap;
                rho[i] += rho_ij;
                rho[j] += rho_ij;
            }
        }
    }
}

void compute_accel(sim_state_t* state, sim_param_t params)
{
    // Unpack basic parameters
    const float h = params.h;
    const float k = params.k;
    const float g = params.g;
    const float mass = state->mass;

    const float h_squared = h * h;
    const float C_0 = mass / M_PI / (h_squared * h_squared);
    const float C_p = 15 * k;
    const float C_v = -40 * params.mu;

    // Unpack system state
    const float* restrict rho = state->rho;
    const float* restrict pos_x = state->pos_x;
    const float* restrict pos_y = state->pos_y;
    const float* restrict v_x = state->v_x;
    const float* restrict v_y = state->v_y;
    float* restrict a_x = state->a_x;
    float* restrict a_y = state->a_y;
    int n = state->n;

    // gravity:
    for (int i = 0; i < n; ++i) {
        a_x[i] = 0;
        a_y[i] = -g;
    }

    // Now compute interaction forces
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            float delta_x = pos_x[i] - pos_x[j];
            float delta_y = pos_y[i] - pos_y[j];
            float dist_squared = delta_x * delta_x + delta_y * delta_y;

            if (dist_squared < h_squared) {
                float q = sqrt(dist_squared) / h;
                float u = 1 - q;
                float w_0 = C_0 * u / rho[i] / rho[j];
                float w_p = w_0 * C_p * (rho[i] + rho[j] - 2 * params.rho0) * u / q;
                float w_v = w_0 * C_v;
                float delta_v_x = v_x[i] - v_y[j];
                float delta_v_y = v_y[i] - v_y[j];
                a_x[i] += (w_p * delta_x + w_v * delta_v_x);
                a_y[i] += (w_p * delta_y + w_v * delta_v_y);
                a_x[j] -= (w_p * delta_x + w_v * delta_v_x);
                a_y[j] -= (w_p * delta_y + w_v * delta_v_y);
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
    v_x[0] *= DAMP;
    v_y[0] *= DAMP;

    vh_x[0] *= DAMP;
    vh_y[0] *= DAMP;
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

void leapfrog(sim_state_t* s, double dt)
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
