#include <cmath>
#include <iostream>
#include <silo.h>
#include <stdlib.h>
#include <vector>
#include <libflatarray/flat_array.hpp>

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

void normalize_mass(float *mass, int n, float *rho, float rho0)
{
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

template<typename FLOAT, typename SOA_ACCESSOR>
void compute_density_lfa_vectorized_1(long /* unused */, long end, SOA_ACCESSOR& particles, float h, float mass)
{
    // std::cout << "A start: " << start << ", end: " << end << "\n";
    float h_squared = h * h;
    FLOAT h_squared_vec(h_squared);

    for (; particles.index() < (end - FLOAT::ARITY + 1); particles += FLOAT::ARITY) {
        &particles.rho() << FLOAT(4 * mass / M_PI) / h_squared_vec;
    }
}

template<typename FLOAT, typename SOA_ACCESSOR>
void compute_density_lfa_vectorized_2(long /* unused */, long end, SOA_ACCESSOR& particles_i, SOA_ACCESSOR& particles_j, float h, float mass, FLOAT pos_x_i, FLOAT pos_y_i, float C)
{
    float h_squared = h * h;
    FLOAT h_squared_vec(h_squared);

    for (; particles_j.index() < (end - FLOAT::ARITY + 1);) {
        FLOAT delta_x = pos_x_i - FLOAT(&particles_j.pos_x());
        FLOAT delta_y = pos_y_i - FLOAT(&particles_j.pos_y());
        FLOAT dist_squared = delta_x * delta_x + delta_y * delta_y;
        FLOAT overlap = h_squared_vec - dist_squared;

        if (LibFlatArray::any(overlap > FLOAT(0.0))) {
            for (int e = 0; e < FLOAT::ARITY; ++e) {
                float o = get(overlap, e);
                if (o > 0) {
                    float rho_ij = C * o * o * o;
                    particles_i.rho() += rho_ij;
                    particles_j.rho() += rho_ij;
                }
                ++particles_j;
            }
        } else {
            particles_j += FLOAT::ARITY;
        }
    }
}

template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
void compute_density_lfa(int n, LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> particles, float h, float mass)
{
    typedef LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> soa_accessor;
    typedef typename LibFlatArray::estimate_optimum_short_vec_type<float, soa_accessor>::VALUE FLOAT;

    LIBFLATARRAY_LOOP_PEELER_TEMPLATE(FLOAT, long, particles.index(), n, compute_density_lfa_vectorized_1, particles, h, mass);
    particles.index() = 0;

    float h_squared = h * h;
    float h_pow_8 = h_squared * h_squared * h_squared * h_squared;
    float C = 4 * mass / M_PI / h_pow_8;

    soa_accessor particles_i(particles.data(), 0);
    soa_accessor particles_j(particles.data(), 0);

    for (particles_i.index() = 0; particles_i.index() < n; ++particles_i) {
        float pos_x_i = particles_i.pos_x();
        float pos_y_i = particles_i.pos_y();

        particles_j.index() = particles_i.index() + 1;
        LIBFLATARRAY_LOOP_PEELER_TEMPLATE(FLOAT, long, particles_j.index(), n, compute_density_lfa_vectorized_2, particles_i, particles_j, h, mass, pos_x_i, pos_y_i, C);
    }
}

template<typename FLOAT, typename SOA_ACCESSOR>
void compute_accel_lfa_vectorized_1(long start, long end, SOA_ACCESSOR particles, float g)
{
    FLOAT zero(0.0);
    FLOAT minus_g(-g);

    for (; particles.index() < end; particles += FLOAT::ARITY) {
        &particles.a_x() << zero;
        &particles.a_y() << minus_g;
    }
}

template<typename FLOAT, typename SOA_ACCESSOR>
void compute_accel_lfa_vectorized_2(long start, long end, SOA_ACCESSOR& particles_i, SOA_ACCESSOR& particles_j, const float h, const float rho0, const FLOAT h_squared, const float C_0, const float C_p, const float C_v)
{
    FLOAT pos_x_i = particles_i.pos_x();
    FLOAT pos_y_i = particles_i.pos_y();

    float rho_i_buf[FLOAT::ARITY];
    float v_x_i_buf[FLOAT::ARITY];
    float v_y_i_buf[FLOAT::ARITY];

    float rho_j_buf[FLOAT::ARITY];
    float v_x_j_buf[FLOAT::ARITY];
    float v_y_j_buf[FLOAT::ARITY];

    float delta_x_buf[FLOAT::ARITY];
    float delta_y_buf[FLOAT::ARITY];
    float dist_squared_buf[FLOAT::ARITY];
    int buf_index = 0;

    for (; particles_j.index() < (end - FLOAT::ARITY + 1); particles_j += FLOAT::ARITY) {
        FLOAT delta_x = pos_x_i - &particles_j.pos_x();
        FLOAT delta_y = pos_y_i - &particles_j.pos_y();
        FLOAT dist_squared = delta_x * delta_x + delta_y * delta_y;

        if (LibFlatArray::any(dist_squared < h_squared)) {
            for (int e = 0; e < FLOAT::ARITY; ++e) {
                if (get(dist_squared, e) < get(h_squared, e)) {
                    rho_i_buf[buf_index] = particles_i.rho();
                    v_x_i_buf[buf_index] = particles_i.v_x();
                    v_y_i_buf[buf_index] = particles_i.v_y();

                    rho_j_buf[buf_index] = particles_j.rho();
                    v_x_j_buf[buf_index] = particles_j.v_x();
                    v_y_j_buf[buf_index] = particles_j.v_y();

                    delta_x_buf[buf_index] = get(delta_x, e);
                    delta_y_buf[buf_index] = get(delta_y, e);
                    dist_squared_buf[buf_index] = get(dist_squared, e);
                }
                if (get(dist_squared, e) < get(h_squared, e)) {
                    float q = sqrt(dist_squared_buf[buf_index]) / h;
                    float u = 1 - q;
                    float w_0 = C_0 * u / rho_i_buf[buf_index] / rho_j_buf[buf_index];
                    float w_p = w_0 * C_p * (rho_i_buf[buf_index] + rho_j_buf[buf_index] - 2 * rho0) * u / q;
                    float w_v = w_0 * C_v;
                    float delta_v_x = v_x_i_buf[buf_index] - v_x_j_buf[buf_index];
                    float delta_v_y = v_y_i_buf[buf_index] - v_y_j_buf[buf_index];

                    particles_i.a_x() += (w_p * delta_x_buf[buf_index] + w_v * delta_v_x);
                    particles_i.a_y() += (w_p * delta_y_buf[buf_index] + w_v * delta_v_y);
                    particles_j.a_x() -= (w_p * delta_x_buf[buf_index] + w_v * delta_v_x);
                    particles_j.a_y() -= (w_p * delta_y_buf[buf_index] + w_v * delta_v_y);
                }
                ++particles_j;
            }
            particles_j.index() -= FLOAT::ARITY;
        }
    }
}


template<typename SOA_ACCESSOR>
void compute_accel_lfa(
    int n,
    SOA_ACCESSOR particles,
    float mass,
    float g,
    float h,
    float k,
    float rho0,
    float mu)
{
    typedef typename LibFlatArray::estimate_optimum_short_vec_type<float, SOA_ACCESSOR>::VALUE FLOAT;

    const float h_squared = h * h;
    const float C_0 = mass / M_PI / (h_squared * h_squared);
    const float C_p = 15 * k;
    const float C_v = -40 * mu;

    // gravity:
    LIBFLATARRAY_LOOP_PEELER_TEMPLATE(FLOAT, long, particles.index(), n, compute_accel_lfa_vectorized_1, particles, g);

    // Now compute interaction forces

    for (int i = 0; i < n; ++i) {
        SOA_ACCESSOR particles_j = particles;
        particles_j.index() = i + 1;
        LIBFLATARRAY_LOOP_PEELER_TEMPLATE(FLOAT, long, particles_j.index(), n, compute_accel_lfa_vectorized_2, particles, particles_j, h, rho0, h_squared, C_0, C_p, C_v);

        ++particles;
    }
}

class Particle
{
public:
    class API
    {
    public:
        LIBFLATARRAY_CUSTOM_SIZES(
            (32)(64)(128)(256)(512)(1024)(2048)(4096)(8192)(16384)(32768)(65536),
            (1),
            (1))
    };

    float rho;
    float pos_x;
    float pos_y;
    float v_x;
    float v_y;
    float a_x;
    float a_y;
};

LIBFLATARRAY_REGISTER_SOA(Particle, ((float)(rho))((float)(pos_x))((float)(pos_y))((float)(v_x))((float)(v_y))((float)(a_x))((float)(a_y)))

int main_c(int argc, char** argv)
{
    // time step length:
    float dt = 1e-4;
    // pitch: (size of particles)
    float h = 2e-2;
    // target density:
    float rho0 = 1000;
    // bulk modulus:
    float k = 1e3;
    // viscosity:
    float mu = 0.1;
    // gravitational acceleration:
    float g = 9.8;

    float hh = h / 1.3;
    int count = count_particles(hh);

    std::vector<float> rho(count);
    std::vector<float> pos_x(count);
    std::vector<float> pos_y(count);
    std::vector<float> v_x(count);
    std::vector<float> v_y(count);
    std::vector<float> a_x(count);
    std::vector<float> a_y(count);

    place_particles(count, pos_x.data(), pos_y.data(), v_x.data(), v_y.data(), hh);
    float mass = 1;
    compute_density(count, rho.data(), pos_x.data(), pos_y.data(), h, mass);
    normalize_mass(&mass, count, rho.data(), rho0);

    int num_steps = 20000;
    int io_period = 15;

    for (int t = 0; t < num_steps; ++t) {
        if ((t % io_period) == 0) {
            dump_time_step(t, count, pos_x.data(), pos_y.data());
        }

        compute_density(
            count,
            rho.data(),
            pos_x.data(),
            pos_y.data(),
            h,
            mass);

        compute_accel(
            count,
            rho.data(),
            pos_x.data(),
            pos_y.data(),
            v_x.data(),
            v_y.data(),
            a_x.data(),
            a_y.data(),
            mass,
            g,
            h,
            k,
            rho0,
            mu);

        leapfrog(
            count,
            pos_x.data(),
            pos_y.data(),
            v_x.data(),
            v_y.data(),
            a_x.data(),
            a_y.data(),
            dt);

        reflect_bc(
            count,
            pos_x.data(),
            pos_y.data(),
            v_x.data(),
            v_y.data());
    }

    dump_time_step(num_steps, count, pos_x.data(), pos_y.data());

    return 0;
}

class Simulate
{
public:
    Simulate(
        float dt,
        float h,
        float rho0,
        float k,
        float mu,
        float g,
        float hh,
        int count) :
        dt(dt),
        h(h),
        rho0(rho0),
        k(k),
        mu(mu),
        g(g),
        hh(hh),
        count(count)
    {}

    template<typename SOA_ACCESSOR>
    void operator()(SOA_ACCESSOR& particles)
    {
        place_particles(count, &particles.pos_x(), &particles.pos_y(), &particles.v_x(), &particles.v_y(), hh);
        float mass = 1;
        compute_density(count, &particles.rho(), &particles.pos_x(), &particles.pos_y(), h, mass);
        normalize_mass(&mass, count, &particles.rho(), rho0);

        int num_steps = 20000;
        int io_period = 15;

        for (int t = 0; t < num_steps; ++t) {
            if ((t % io_period) == 0) {
                dump_time_step(t, count, &particles.pos_x(), &particles.pos_y());
            }

            compute_density_lfa(
                count,
                particles,
                h,
                mass);

            compute_accel_lfa(
                count,
                particles,
                mass,
                g,
                h,
                k,
                rho0,
                mu);

            leapfrog(
                count,
                &particles.pos_x(),
                &particles.pos_y(),
                &particles.v_x(),
                &particles.v_y(),
                &particles.a_x(),
                &particles.a_y(),
                dt);

            reflect_bc(
                count,
                &particles.pos_x(),
                &particles.pos_y(),
                &particles.v_x(),
                &particles.v_y());
        }

        dump_time_step(num_steps, count, &particles.pos_x(), &particles.pos_y());
    }

private:
    float dt;
    float h;
    float rho0;
    float k;
    float mu;
    float g;
    float hh;
    int count;
};

int main(int argc, char** argv)
{
    // time step length:
    float dt = 1e-4;
    // pitch: (size of particles)
    float h = 2e-2;
    // target density:
    float rho0 = 1000;
    // bulk modulus:
    float k = 1e3;
    // viscosity:
    float mu = 0.1;
    // gravitational acceleration:
    float g = 9.8;

    float hh = h / 1.3;
    int count = count_particles(hh);

    LibFlatArray::soa_grid<Particle> particles(count, 1, 1);

    Simulate sim_functor(dt, h, rho0, k, mu, g, hh, count);
    particles.callback(sim_functor);

    return 0;
}
