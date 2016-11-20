#ifndef LIBFLATARRAY_EXAMPLES_SMOOTHED_PARTICLE_HYDRODYNAMICS_KERNELS_HPP
#define LIBFLATARRAY_EXAMPLES_SMOOTHED_PARTICLE_HYDRODYNAMICS_KERNELS_HPP

#define PI float(M_PI)

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

template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
void compute_density_lfa(int n, LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> particles, float h, float mass)
{
    typedef LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> soa_accessor;
    typedef typename LibFlatArray::estimate_optimum_short_vec_type<float, soa_accessor>::VALUE FLOAT;

    LibFlatArray::loop_peeler<FLOAT>(&particles.index(), n, [&particles, h, mass](auto my_float, long *i, long end) {
            typedef decltype(my_float) FLOAT;
            float h_squared = h * h;
            FLOAT h_squared_vec(h_squared);

            for (; particles.index() < end; particles += FLOAT::ARITY) {
                &particles.rho() << 4.0f * mass / PI / h_squared_vec;
            }
        });

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

        LibFlatArray::loop_peeler<FLOAT>(
            &particles_j.index(), n,
            [&particles_i, &particles_j, h, mass, pos_x_i, pos_y_i, C]
            (auto my_float, long *x, int end)
            {
                typedef decltype(my_float) FLOAT;
                float h_squared = h * h;
                FLOAT h_squared_vec(h_squared);

                for (; particles_j.index() < end;) {
                    FLOAT delta_x = FLOAT(pos_x_i) - &particles_j.pos_x();
                    FLOAT delta_y = FLOAT(pos_y_i) - &particles_j.pos_y();
                    FLOAT dist_squared = delta_x * delta_x + delta_y * delta_y;
                    FLOAT overlap = h_squared_vec - dist_squared;

                    if (LibFlatArray::any(overlap > FLOAT(0.0))) {
                        for (std::size_t e = 0; e < FLOAT::ARITY; ++e) {
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
            });
    }
}

template<int ARITY, typename SOA_ACCESSOR, typename SOA_ARRAY>
void handle_interactions(SOA_ACCESSOR& particles, SOA_ARRAY& interaction_buf, const float h, const float rho0, const float C_0, const float C_p, const float C_v)
{
    typedef LibFlatArray::short_vec<float, ARITY> FLOAT;
    for (std::size_t f = 0; f < interaction_buf.size(); f += FLOAT::ARITY) {
        FLOAT q = sqrt(FLOAT(&interaction_buf[f].dist_squared())) / h;

        FLOAT u = 1.0f - q;
        FLOAT w_0 = u * C_0 / &interaction_buf[f].rho_i() / &interaction_buf[f].rho_j();
        FLOAT w_p = w_0 * C_p * (FLOAT(&interaction_buf[f].rho_i()) + &interaction_buf[f].rho_j() - 2 * rho0) * u / q;
        FLOAT w_v = w_0 * C_v;
        FLOAT delta_v_x = FLOAT(&interaction_buf[f].v_x_i()) - &interaction_buf[f].v_x_j();
        FLOAT delta_v_y = FLOAT(&interaction_buf[f].v_y_i()) - &interaction_buf[f].v_y_j();

        FLOAT add_x = w_p * &interaction_buf[f].delta_x() + w_v * delta_v_x;
        FLOAT add_y = w_p * &interaction_buf[f].delta_y() + w_v * delta_v_y;
        // scalar store to avoid simultaneous overwrites:
        for (std::size_t i = 0; i < FLOAT::ARITY; ++i) {
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
    LibFlatArray::loop_peeler<FLOAT>(
        &particles.index(), long(n),
        [&particles, g](auto float_var, long *x, long end) {

            typedef decltype(float_var) FLOAT;

            for (; particles.index() < end; particles += FLOAT::ARITY) {
                &particles.a_x() << FLOAT(0.0f);
                &particles.a_y() << FLOAT(-g);
            }
        });

    particles.index() = 0;

    typedef LibFlatArray::soa_array<InteractionBuffer, FLOAT::ARITY> soa_array;
    // typedef LibFlatArray::soa_array<InteractionBuffer, 16> soa_array;
    // std::cout << "buf: " << soa_array::SIZE << "\n";
    soa_array interaction_buf;

    SOA_ACCESSOR particles_i = particles;
    SOA_ACCESSOR particles_j = particles;

    // Now compute interaction forces
    for (particles_i.index() = 0; particles_i.index() < n; ++particles_i) {
        particles_j.index() = particles_i.index() + 1;

        LibFlatArray::loop_peeler<FLOAT>(
            &particles_j.index(), n,
            [&particles, &particles_i, &particles_j, h, rho0, h_squared, C_0, C_p, C_v, &interaction_buf]
            (auto my_float, long *x, long end) {

                typedef decltype(my_float) FLOAT;
                FLOAT pos_x_i = particles_i.pos_x();
                FLOAT pos_y_i = particles_i.pos_y();

                for (; particles_j.index() < end; particles_j += FLOAT::ARITY) {
                    FLOAT delta_x = pos_x_i - &particles_j.pos_x();
                    FLOAT delta_y = pos_y_i - &particles_j.pos_y();
                    FLOAT dist_squared = delta_x * delta_x + delta_y * delta_y;

                    if (LibFlatArray::any(dist_squared < FLOAT(h_squared))) {
                        for (std::size_t e = 0; e < FLOAT::ARITY; ++e, ++particles_j) {
                            if (get(dist_squared, e) < h_squared) {
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
                            if (interaction_buf.size() == soa_array::SIZE) {
                                handle_interactions<soa_array::SIZE>(particles, interaction_buf, h, rho0, C_0, C_p, C_v);
                            }
                        }
                        particles_j += -FLOAT::ARITY;
                    }
                }
            });
    }
    handle_interactions<1>(particles, interaction_buf, h, rho0, C_0, C_p, C_v);
}

#endif
