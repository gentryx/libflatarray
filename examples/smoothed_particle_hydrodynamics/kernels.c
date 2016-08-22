#include <assert.h>

#include "kernels.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void compute_density(int n, float *restrict rho, float *restrict pos_x, float *restrict pos_y, float h, float mass)
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

void compute_accel(
    int n,
    float *restrict rho,
    float *restrict pos_x,
    float *restrict pos_y,
    float *restrict v_x,
    float *restrict v_y,
    float *restrict a_x,
    float *restrict a_y,
    float mass,
    float g,
    float h,
    float k,
    float rho0,
    float mu)
{
    const float h_squared = h * h;
    const float C_0 = mass / M_PI / (h_squared * h_squared);
    const float C_p = 15 * k;
    const float C_v = -40 * mu;

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
                float w_p = w_0 * C_p * (rho[i] + rho[j] - 2 * rho0) * u / q;
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

void damp_reflect(
    int which,
    float barrier,
    float *pos_x,
    float *pos_y,
    float *v_x,
    float *v_y)
{
    float *v_which   = (which == 0) ? v_x   : v_y;
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

    // Damp the velocities
    v_x[0] *= DAMP;
    v_y[0] *= DAMP;
}

void reflect_bc(
    int n,
    float *restrict pos_x,
    float *restrict pos_y,
    float *restrict v_x,
    float *restrict v_y)
{
    // Boundaries of the computational domain
    const float XMIN = 0.0;
    const float XMAX = 1.0;
    const float YMIN = 0.0;
    const float YMAX = 1.0;

    for (int i = 0; i < n; ++i, pos_x += 1, pos_y += 1, v_x += 1, v_y +=1) {
        if (pos_x[0] < XMIN) {
            damp_reflect(0, XMIN, pos_x, pos_y, v_x, v_y);
        }
        if (pos_x[0] > XMAX) {
            damp_reflect(0, XMAX, pos_x, pos_y, v_x, v_y);
        }
        if (pos_y[0] < YMIN) {
            damp_reflect(1, YMIN, pos_x, pos_y, v_x, v_y);
        }
        if (pos_y[0] > YMAX) {
            damp_reflect(1, YMAX, pos_x, pos_y, v_x, v_y);
        }
    }
}

void leapfrog(
    int n,
    float *restrict pos_x,
    float *restrict pos_y,
    float *restrict v_x,
    float *restrict v_y,
    float *restrict a_x,
    float *restrict a_y,
    double dt)
{
    for (int i = 0; i < n; ++i) {
        v_x[i] += a_x[i] * dt;
        v_y[i] += a_y[i] * dt;

        pos_x[i] += v_x[i] * dt;
        pos_y[i] += v_y[i] * dt;
    }
}
