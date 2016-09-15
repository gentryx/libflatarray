#include <cmath>
#include <iostream>
#include <silo.h>
#include <stdlib.h>
#include <vector>
#include <libflatarray/flat_array.hpp>

#include "kernels.h"

class InteractionBuffer
{
public:
    inline
    explicit InteractionBuffer(
        float rho_i = 0,
        float v_x_i = 0,
        float v_y_i = 0,
        float rho_j = 0,
        float v_x_j = 0,
        float v_y_j = 0,
        float delta_x = 0,
        float delta_y = 0,
        float dist_squared = 0,
        int i = 0,
        int j = 0) :
        rho_i(rho_i),
        v_x_i(v_x_i),
        v_y_i(v_y_i),
        rho_j(rho_j),
        v_x_j(v_x_j),
        v_y_j(v_y_j),
        delta_x(delta_x),
        delta_y(delta_y),
        dist_squared(dist_squared),
        i(i),
        j(j)
    {}

    float rho_i;
    float v_x_i;
    float v_y_i;
    float rho_j;
    float v_x_j;
    float v_y_j;
    float delta_x;
    float delta_y;
    float dist_squared;
    int i;
    int j;
};
LIBFLATARRAY_REGISTER_SOA(
    InteractionBuffer,
    ((float)(rho_i))
    ((float)(v_x_i))
    ((float)(v_y_i))
    ((float)(rho_j))
    ((float)(v_x_j))
    ((float)(v_y_j))
    ((float)(delta_x))
    ((float)(delta_y))
    ((float)(dist_squared))
    ((int)(i))
    ((int)(j)))

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

template<int ARITY, typename SOA_ACCESSOR, typename SOA_ARRAY>
void handle_interactions(SOA_ACCESSOR& particles, SOA_ARRAY& interaction_buf, const float h, const float rho0, const float C_0, const float C_p, const float C_v)
{
    typedef LibFlatArray::short_vec<float, ARITY> FLOAT;
    for (int f = 0; f < (int(interaction_buf.size()) - FLOAT::ARITY + 1); f += FLOAT::ARITY) {
        // fixme: enable this code:  FLOAT q = sqrt(FLOAT(&interaction_buf[f].dist_squared())) / h;
        FLOAT q = sqrt(FLOAT(&interaction_buf[f].dist_squared()));
        q /= h;

        FLOAT u = FLOAT(1) - q;
        FLOAT w_0 = FLOAT(C_0) * u / &interaction_buf[f].rho_i() / &interaction_buf[f].rho_j();
        FLOAT w_p = w_0 * FLOAT(C_p) * (FLOAT(&interaction_buf[f].rho_i()) + FLOAT(&interaction_buf[f].rho_j()) - 2 * rho0) * u / q;
        FLOAT w_v = w_0 * FLOAT(C_v);
        FLOAT delta_v_x = FLOAT(&interaction_buf[f].v_x_i()) - &interaction_buf[f].v_x_j();
        FLOAT delta_v_y = FLOAT(&interaction_buf[f].v_y_i()) - &interaction_buf[f].v_y_j();

        FLOAT add_x = w_p * FLOAT(&interaction_buf[f].delta_x()) + w_v * delta_v_x;
        FLOAT add_y = w_p * FLOAT(&interaction_buf[f].delta_y()) + w_v * delta_v_y;
        // scalar store to avoid simultaneous overwrites:
        for (int i = 0; i < FLOAT::ARITY; ++i) {
            float add_x_scalar = get(add_x, i);
            float add_y_scalar = get(add_y, i);
            (&particles.a_x())[interaction_buf[f + i].i()] += add_x_scalar;
            (&particles.a_y())[interaction_buf[f + i].i()] += add_y_scalar;
            (&particles.a_x())[interaction_buf[f + i].j()] -= add_x_scalar;
            (&particles.a_y())[interaction_buf[f + i].j()] -= add_y_scalar;
        }
    }
    interaction_buf.clear();
}

template<typename FLOAT, typename SOA_ACCESSOR, typename SOA_ARRAY>
void compute_accel_lfa_vectorized_2(long start, long end, SOA_ACCESSOR& particles, SOA_ACCESSOR& particles_i, SOA_ACCESSOR& particles_j, const float h, const float rho0, const FLOAT h_squared, const float C_0, const float C_p, const float C_v, SOA_ARRAY& interaction_buf)
{
    FLOAT pos_x_i = particles_i.pos_x();
    FLOAT pos_y_i = particles_i.pos_y();

    for (; particles_j.index() < (end - FLOAT::ARITY + 1); particles_j += FLOAT::ARITY) {
        FLOAT delta_x = pos_x_i - &particles_j.pos_x();
        FLOAT delta_y = pos_y_i - &particles_j.pos_y();
        FLOAT dist_squared = delta_x * delta_x + delta_y * delta_y;

        if (LibFlatArray::any(dist_squared < h_squared)) {
            for (int e = 0; e < FLOAT::ARITY; ++e) {
                if (get(dist_squared, e) < get(h_squared, e)) {
                    interaction_buf << InteractionBuffer(
                        particles_i.rho(),
                        particles_i.v_x(),
                        particles_i.v_y(),

                        particles_j.rho(),
                        particles_j.v_x(),
                        particles_j.v_y(),

                        get(delta_x, e),
                        get(delta_y, e),
                        get(dist_squared, e),
                        particles_i.index(),
                        particles_j.index());
                }
                // fixme: needs additional sweep after vector loop
                if (interaction_buf.size() == SOA_ARRAY::SIZE) {
                    handle_interactions<SOA_ARRAY::SIZE>(particles, interaction_buf, h, rho0, C_0, C_p, C_v);
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

    typedef LibFlatArray::soa_array<InteractionBuffer, FLOAT::ARITY> soa_array;
    // typedef LibFlatArray::soa_array<InteractionBuffer, 16> soa_array;
    // std::cout << "buf: " << soa_array::SIZE << "\n";
    soa_array interaction_buf;

    SOA_ACCESSOR particles_i = particles;
    SOA_ACCESSOR particles_j = particles;

    // Now compute interaction forces
    for (particles_i.index() = 0; particles_i.index() < n; ++particles_i) {
        particles_j.index() = particles_i.index() + 1;
        LIBFLATARRAY_LOOP_PEELER_TEMPLATE(FLOAT, long, particles_j.index(), n, compute_accel_lfa_vectorized_2, particles, particles_i, particles_j, h, rho0, h_squared, C_0, C_p, C_v, interaction_buf);
    }
    handle_interactions<1>(particles, interaction_buf, h, rho0, C_0, C_p, C_v);
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

LIBFLATARRAY_REGISTER_SOA(Particle, ((float)(rho))((float)(pos_x))((float)(pos_y))((float)(v_x))((float)(v_y))((float)(a_x))((float)(a_y)));


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
    int io_period = 20000;

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
