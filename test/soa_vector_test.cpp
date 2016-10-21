/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <libflatarray/flat_array.hpp>

#include "test.hpp"

class Particle
{
public:
    float pos[3];
    float vel[3];
    int time_to_live;
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

    BOOST_TEST(vec1.capacity() < vec2.capacity());
    BOOST_TEST_EQ(vec2.capacity(), 20);

    vec1 = vec2;
    BOOST_TEST_EQ(vec1.size(), 20);
    BOOST_TEST_EQ(vec2.size(), 20);

    vec2.clear();
    BOOST_TEST_EQ(vec1.size(), 20);
    BOOST_TEST_EQ(vec2.size(),  0);

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

}

int main(int argc, char **argv)
{
    return 0;
}
