/**
 * Copyright 2016-2017 Andreas Schäfer
 * Copyright 2017 Google
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <libflatarray/flat_array.hpp>

// globally disable some warnings with MSVC, that are issued not for a
// specific header, but rather for the interaction of system headers
// and LibFlatArray source.  Also disable overly eager sign conversion
// and overflow warnings:
#ifdef _MSC_BUILD
#pragma warning( disable : 4244 4305 4307 4365 4456 4514 4710 4800 )
#endif

#include "test.hpp"

class Particle
{
public:
    Particle()
    {}

    Particle(int i, int j)
    {
        pos[0] = i +  0;
        pos[1] = i + 10;
        pos[2] = i + 20;
        vel[0] = i + 30;
        vel[1] = i + 40;
        vel[2] = i + 50;
        time_to_live = j;
    }

    float pos[3];
    float vel[3];
    int time_to_live;
};

class UpdateParticles
{
public:
    UpdateParticles(int count) :
        count(count)
    {}

    template<typename SOA_ACCESSOR>
    void operator()(SOA_ACCESSOR particle_iter)
    {
        // fixme: use particle_iter < count bzw !=
        for (; particle_iter.index() < count; ++particle_iter) {
            particle_iter.pos()[0] += 0.1f * particle_iter.vel()[0];
            particle_iter.pos()[1] += 0.1f * particle_iter.vel()[1];
            particle_iter.pos()[2] += 0.1f * particle_iter.vel()[2];
        }
    }

private:
    int count;
};

LIBFLATARRAY_REGISTER_SOA(
    Particle,
    ((float)(pos)(3))
    ((float)(vel)(3))
    ((int)(time_to_live)))

namespace LibFlatArray {

ADD_TEST(TestConstructor)
{
    Particle reference;
    reference.pos[0] = 1.0f;
    reference.pos[1] = 2.0f;
    reference.pos[2] = 3.0f;
    reference.vel[0] = 4.0f;
    reference.vel[1] = 5.0f;
    reference.vel[2] = 6.0f;
    reference.time_to_live = 1234;

    soa_vector<Particle> vec1;
    soa_vector<Particle> vec2(20, reference);

    BOOST_TEST_EQ(vec1.size(),  0);
    BOOST_TEST_EQ(vec2.size(), 20);

    BOOST_TEST_EQ(vec1.empty(), true);
    BOOST_TEST_EQ(vec2.empty(), false);

    BOOST_TEST(vec1.capacity() < vec2.capacity());
    BOOST_TEST_EQ(vec2.capacity(), 20);

    vec1 = vec2;
    BOOST_TEST_EQ(vec1.size(), 20);
    BOOST_TEST_EQ(vec2.size(), 20);

    BOOST_TEST_EQ(vec1.empty(), false);
    BOOST_TEST_EQ(vec2.empty(), false);

    vec2.clear();
    BOOST_TEST_EQ(vec1.size(), 20);
    BOOST_TEST_EQ(vec2.size(),  0);
    BOOST_TEST_EQ(vec1.empty(), false);
    BOOST_TEST_EQ(vec2.empty(), true);

    for (std::size_t i = 0; i < vec1.size(); ++i) {
        BOOST_TEST_EQ(1.0f, vec1.get(i).pos[0]);
        BOOST_TEST_EQ(2.0f, vec1.get(i).pos[1]);
        BOOST_TEST_EQ(3.0f, vec1.get(i).pos[2]);
        BOOST_TEST_EQ(4.0f, vec1.get(i).vel[0]);
        BOOST_TEST_EQ(5.0f, vec1.get(i).vel[1]);
        BOOST_TEST_EQ(6.0f, vec1.get(i).vel[2]);
        BOOST_TEST_EQ(1234, vec1.get(i).time_to_live);
    }
}

ADD_TEST(TestResizeAndReserve)
{
    soa_vector<Particle> vec(10);

    for (int i = 0; i < 10; ++i) {
        Particle p = vec.get(i);
        p.time_to_live = 1000 + i;
        vec.set(i, p);
    }

    BOOST_TEST_EQ(10, vec.size());

    vec.resize(50);
    BOOST_TEST_EQ(50, vec.size());

    for (int i = 0; i < 10; ++i) {
        Particle p = vec.get(i);
        BOOST_TEST_EQ(1000 + i, p.time_to_live);
    }

    vec.resize(5);
    BOOST_TEST_EQ(5, vec.size());

    for (int i = 0; i < 5; ++i) {
        Particle p = vec.get(i);
        BOOST_TEST_EQ(1000 + i, p.time_to_live);
    }

    vec.reserve(1000);
    BOOST_TEST(vec.grid.extent_x() > 1000);
    BOOST_TEST_EQ(vec.grid.extent_y(), 1);
    BOOST_TEST_EQ(vec.grid.extent_z(), 1);
    BOOST_TEST_EQ(5, vec.size());

    for (int i = 0; i < 5; ++i) {
        Particle p = vec.get(i);
        BOOST_TEST_EQ(1000 + i, p.time_to_live);
    }
}

ADD_TEST(TestPushBackAndPopBack)
{
    soa_vector<Particle> vec;
    BOOST_TEST_EQ(0, vec.size());

    Particle p;
    p.pos[0] = 10.0;
    p.pos[1] = 10.1;
    p.pos[2] = 10.2;
    vec.push_back(p);
    BOOST_TEST_EQ(1, vec.size());

    Particle q = vec.get(0);
    BOOST_TEST_EQ(10.0f, q.pos[0]);
    BOOST_TEST_EQ(10.1f, q.pos[1]);
    BOOST_TEST_EQ(10.2f, q.pos[2]);

    vec.pop_back();
    BOOST_TEST_EQ(0, vec.size());

    for (int i = 0; i < 1000; ++i) {
        Particle p;
        p.pos[0] = i + 0.0;
        p.pos[1] = i + 0.1;
        p.pos[2] = i + 0.2;
        vec.push_back(p);
    }

    BOOST_TEST_EQ(1000, vec.size());
    BOOST_TEST(1000 < vec.capacity());

    for (int i = 0; i < 1000; ++i) {
        Particle p = vec.get(i);
        BOOST_TEST_EQ(i + 0.0f, p.pos[0]);
        BOOST_TEST_EQ(i + 0.1f, p.pos[1]);
        BOOST_TEST_EQ(i + 0.2f, p.pos[2]);
    }

    vec.pop_back();
    vec.pop_back();
    vec.pop_back();
    vec.pop_back();
    BOOST_TEST_EQ(996, vec.size());
}

ADD_TEST(TestCallback)
{
    soa_vector<Particle> vec;


    for (int i = 0; i < 100; ++i) {
        Particle p;
        p.pos[0] = i * 1;
        p.pos[1] = i * 2;
        p.pos[2] = i * 3;

        p.vel[0] = i * 4;
        p.vel[1] = i * 5;
        p.vel[2] = i * 6;

        vec.push_back(p);
    }

    vec.callback(UpdateParticles(vec.size()));

    for (int i = 0; i < 100; ++i) {
        Particle p = vec.get(i);

        float expected_pos_x = (i * 1) + 0.1f * (i * 4);
        float expected_pos_y = (i * 2) + 0.1f * (i * 5);
        float expected_pos_z = (i * 3) + 0.1f * (i * 6);

        BOOST_TEST_EQ(p.pos[0], expected_pos_x);
        BOOST_TEST_EQ(p.pos[1], expected_pos_y);
        BOOST_TEST_EQ(p.pos[2], expected_pos_z);
    }
}

ADD_TEST(TestEmplace)
{
#ifdef LIBFLATARRAY_WITH_CPP14
    soa_vector<Particle> vec;
    vec.emplace_back(1, 2);
    vec.emplace_back(3, 4);

    BOOST_TEST_EQ( 1, vec.get(0).pos[0]);
    BOOST_TEST_EQ(11, vec.get(0).pos[1]);
    BOOST_TEST_EQ(21, vec.get(0).pos[2]);
    BOOST_TEST_EQ(31, vec.get(0).vel[0]);
    BOOST_TEST_EQ(41, vec.get(0).vel[1]);
    BOOST_TEST_EQ(51, vec.get(0).vel[2]);
    BOOST_TEST_EQ( 2, vec.get(0).time_to_live);

    BOOST_TEST_EQ( 3, vec.get(1).pos[0]);
    BOOST_TEST_EQ(13, vec.get(1).pos[1]);
    BOOST_TEST_EQ(23, vec.get(1).pos[2]);
    BOOST_TEST_EQ(33, vec.get(1).vel[0]);
    BOOST_TEST_EQ(43, vec.get(1).vel[1]);
    BOOST_TEST_EQ(53, vec.get(1).vel[2]);
    BOOST_TEST_EQ( 4, vec.get(1).time_to_live);
#endif
}

}

int main(int /* argc */, char ** /* argv */)
{
    return 0;
}
