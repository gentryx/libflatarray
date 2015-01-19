/**
 * Copyright 2013, 2014, 2015 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/detail/lightweight_test.hpp>
#include <iostream>
#include <libflatarray/flat_array.hpp>
#include <vector>

#include "test.hpp"

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

LIBFLATARRAY_REGISTER_SOA(
    Particle,
    ((float)(posX))
    ((float)(posY))
    ((float)(posZ))
    ((float)(velX))
    ((float)(velY))
    ((float)(velZ))
    ((float)(charge))
    ((float)(mass)))

class ArrayParticle
{
public:
    ArrayParticle(
        float mass = 0,
        float charge = 0,
        float pos0 = 0,
        float pos1 = 0,
        float pos2 = 0,
        float vel0 = 0,
        float vel1 = 0,
        float vel2 = 0,
        int state = 0) :
        mass(mass),
        charge(charge),
        state(state)
    {
        pos[0] = pos0;
        pos[1] = pos1;
        pos[2] = pos2;
        vel[0] = vel0;
        vel[1] = vel1;
        vel[2] = vel2;
    }

    float mass;
    float charge;
    float pos[3];
    float vel[3];
    int state;
};

LIBFLATARRAY_REGISTER_SOA(
    ArrayParticle,
    ((float)(mass))
    ((float)(charge))
    ((float)(pos)(3))
    ((float)(vel)(3))
    ((int)(state)))

namespace LibFlatArray {

ADD_TEST(TestBasicAccessAndConversion)
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

    Particle foo(array[5]);

    BOOST_TEST(foo.posX == 5);
    BOOST_TEST(foo.posY == 20);
    BOOST_TEST(foo.posZ == 30);
    BOOST_TEST(foo.velX == 40);
    BOOST_TEST(foo.velY == 50);
    BOOST_TEST(foo.velZ == 65);
    BOOST_TEST(foo.charge == 25);
    BOOST_TEST(foo.mass == -95);

    foo = array[10];

    BOOST_TEST(foo.posX == 10);
    BOOST_TEST(foo.posY == 20);
    BOOST_TEST(foo.posZ == 30);
    BOOST_TEST(foo.velX == 40);
    BOOST_TEST(foo.velY == 50);
    BOOST_TEST(foo.velZ == 70);
    BOOST_TEST(foo.charge == 100);
    BOOST_TEST(foo.mass == -90);
}

ADD_TEST(TestArrayMember)
{
    soa_array<ArrayParticle, 50> array;
    const int num = 50;

    for (int i = 0; i < num; ++i) {
        array << ArrayParticle(
            0.1 + i,
            0.2 + i,
            1.0 + i,
            1.1 + i,
            1.2 + i,
            2.0 + i,
            2.1 + i,
            2.2 + i,
            3 * i);
    }

    for (int i = 0; i < num; ++i) {
        ArrayParticle p = array[i];

        float expectedMass   = 0.1 + i;
        float expectedCharge = 0.2 + i;
        float expectedPos0   = 1.0 + i;
        float expectedPos1   = 1.1 + i;
        float expectedPos2   = 1.2 + i;
        float expectedVel0   = 2.0 + i;
        float expectedVel1   = 2.1 + i;
        float expectedVel2   = 2.2 + i;
        int expectedState = 3 * i;

        BOOST_TEST(p.mass   == expectedMass);
        BOOST_TEST(p.charge == expectedCharge);
        BOOST_TEST(p.pos[0] == expectedPos0);
        BOOST_TEST(p.pos[1] == expectedPos1);
        BOOST_TEST(p.pos[2] == expectedPos2);
        BOOST_TEST(p.vel[0] == expectedVel0);
        BOOST_TEST(p.vel[1] == expectedVel1);
        BOOST_TEST(p.vel[2] == expectedVel2);
        BOOST_TEST(p.state  == expectedState);
    }
}

}

int main(int argc, char **argv)
{
    return 0;
}
