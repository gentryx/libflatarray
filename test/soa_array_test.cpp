/**
 * Copyright 2013 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/detail/lightweight_test.hpp>
#include <iostream>
#include <libflatarray/flat_array.hpp>
#include <vector>

#include "test.h"

class Particle
{
public:
    inline
    Particle(
        float posX = 0.0,
        float posY = 0.0,
        float posZ = 0.0,
        float velX = 0.0,
        float velY = 0.0,
        float velZ = 0.0,
        float charge = 0.0,
        float mass = 1.0) :
        posX(posX),
        posY(posY),
        posZ(posZ),
        velX(velX),
        velY(velY),
        velZ(velZ),
        charge(charge),
        mass(mass)
    {}

    float posX;
    float posY;
    float posZ;
    float velX;
    float velY;
    float velZ;
    float charge;
    float mass;
};

LIBFLATARRAY_REGISTER_SOA(Particle, ((float)(posX))((float)(posY))((float)(posZ))((float)(velX))((float)(velY))((float)(velZ))((float)(charge))((float)(mass)))

namespace LibFlatArray {

ADD_TEST(TestBasic)
{
    soa_array<Particle, 20> array;
    for (int i = 0; i < 10; ++i) {
        array << Particle(i, 20, 30, 40, 50, 60 + i, i * i, -100 + i);
    }
    for (int i = 10; i < 13; ++i) {
        array.push_back(Particle(i, 20, 30, 40, 50, 60 + i, i * i, -100 + i));
    }

    BOOST_TEST(array.size() == 13);

    for (int i = 0; i < 13; ++i) {
        BOOST_TEST(array[i].posX() == i);
        BOOST_TEST(array[i].posY() == 20);
        BOOST_TEST(array[i].posZ() == 30);

        BOOST_TEST(array[i].velX() == 40);
        BOOST_TEST(array[i].velY() == 50);
        BOOST_TEST(array[i].velZ() == (60 + i));

        BOOST_TEST(array[i].charge() == (i * i));
        BOOST_TEST(array[i].mass() == (-100 + i));
    }
}

}

int main(int argc, char **argv)
{
    return 0;
}
