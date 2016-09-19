/**
 * Copyright 2013-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <iostream>
#include <libflatarray/flat_array.hpp>
#include <map>
#include <vector>

#include "test.hpp"

class Particle
{
public:
    inline
    explicit Particle(
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
    explicit ArrayParticle(
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

class DestructionCounterClass
{
public:
    static std::size_t countConstruct;
    static std::size_t countDestruct;

    DestructionCounterClass()
    {
        ++countConstruct;
    }

    ~DestructionCounterClass()
    {
        ++countDestruct;
    }
};

std::size_t DestructionCounterClass::countConstruct = 0;
std::size_t DestructionCounterClass::countDestruct = 0;

class CellWithNonTrivialMembers
{
public:
    typedef std::map<int, std::vector<double> > MapType;
    int id;
    MapType map;
    MapType maps[4];
    DestructionCounterClass destructCounter;
};

LIBFLATARRAY_REGISTER_SOA(
    CellWithNonTrivialMembers,
    ((int)(id))
    ((CellWithNonTrivialMembers::MapType)(map))
    ((CellWithNonTrivialMembers::MapType)(maps)(4))
    ((DestructionCounterClass)(destructCounter)))


namespace LibFlatArray {

ADD_TEST(TestBasicAccessAndConversion)
{
    soa_array<Particle, 20> array;
    for (int i = 0; i < 10; ++i) {
        array << Particle(i, 20, 30, 40, 50, 60 + i, i * i, -100 + i);
    }
    for (int i = 10; i < 18; ++i) {
        array.push_back(Particle(i, 20, 30, 40, 50, 60 + i, i * i, -100 + i));
    }

    BOOST_TEST(array.size() == 18);
    BOOST_TEST(array.byte_size() == (18 * 8 * sizeof(float)));

    for (int i = 0; i < 18; ++i) {
        BOOST_TEST(array[i].posX() == i);
        BOOST_TEST(array[i].posY() == 20);
        BOOST_TEST(array[i].posZ() == 30);

        BOOST_TEST(array[i].velX() == 40);
        BOOST_TEST(array[i].velY() == 50);
        BOOST_TEST(array[i].velZ() == (60 + i));

        BOOST_TEST(array[i].charge() == (i * i));
        BOOST_TEST(array[i].mass() == (-100 + i));
    }

    BOOST_TEST((array[10].access_member<float, 0>()) ==  10);
    BOOST_TEST((array[10].access_member<float, 1>()) ==  20);
    BOOST_TEST((array[10].access_member<float, 2>()) ==  30);
    BOOST_TEST((array[10].access_member<float, 3>()) ==  40);
    BOOST_TEST((array[10].access_member<float, 4>()) ==  50);
    BOOST_TEST((array[10].access_member<float, 5>()) ==  70);
    BOOST_TEST((array[10].access_member<float, 6>()) == 100);
    BOOST_TEST((array[10].access_member<float, 7>()) == -90);

    BOOST_TEST(*(reinterpret_cast<float*>(array[15].access_member(4,  0))) ==  15);
    BOOST_TEST(*(reinterpret_cast<float*>(array[15].access_member(4,  4))) ==  20);
    BOOST_TEST(*(reinterpret_cast<float*>(array[15].access_member(4,  8))) ==  30);
    BOOST_TEST(*(reinterpret_cast<float*>(array[15].access_member(4, 12))) ==  40);
    BOOST_TEST(*(reinterpret_cast<float*>(array[15].access_member(4, 16))) ==  50);
    BOOST_TEST(*(reinterpret_cast<float*>(array[15].access_member(4, 20))) ==  75);
    BOOST_TEST(*(reinterpret_cast<float*>(array[15].access_member(4, 24))) == 225);
    BOOST_TEST(*(reinterpret_cast<float*>(array[15].access_member(4, 28))) == -85);

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

    BOOST_TEST(array.byte_size() == (num * (8 * sizeof(float) + sizeof(int))));

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

    for (int i = 0; i < num; ++i) {
        float expectedPos0 = 1.0 + i;
        float expectedPos1 = 1.1 + i;
        float expectedPos2 = 1.2 + i;
        float expectedVel0 = 2.0 + i;
        float expectedVel1 = 2.1 + i;
        float expectedVel2 = 2.2 + i;

        BOOST_TEST(array[i].pos()[0] == expectedPos0);
        BOOST_TEST(array[i].pos()[1] == expectedPos1);
        BOOST_TEST(array[i].pos()[2] == expectedPos2);
        BOOST_TEST(array[i].vel()[0] == expectedVel0);
        BOOST_TEST(array[i].vel()[1] == expectedVel1);
        BOOST_TEST(array[i].vel()[2] == expectedVel2);

        array[i].pos<0>() = expectedPos0 * 2;
        array[i].pos<1>() = expectedPos1 * 3;
        array[i].pos<2>() = expectedPos2 * 4;
        array[i].vel<0>() = expectedVel0 * 5;
        array[i].vel<1>() = expectedVel1 * 6;
        array[i].vel<2>() = expectedVel2 * 7;
    }

    for (int i = 0; i < num; ++i) {
        float expectedPos0 = float(1.0 + i) * 2;
        float expectedPos1 = float(1.1 + i) * 3;
        float expectedPos2 = float(1.2 + i) * 4;
        float expectedVel0 = float(2.0 + i) * 5;
        float expectedVel1 = float(2.1 + i) * 6;
        float expectedVel2 = float(2.2 + i) * 7;

        BOOST_TEST(array[i].pos()[0] == expectedPos0);
        BOOST_TEST(array[i].pos()[1] == expectedPos1);
        BOOST_TEST(array[i].pos()[2] == expectedPos2);
        BOOST_TEST(array[i].vel()[0] == expectedVel0);
        BOOST_TEST(array[i].vel()[1] == expectedVel1);
        BOOST_TEST(array[i].vel()[2] == expectedVel2);
    }

    std::vector<float> buf(num * 3, -1);
    for (int i = 0; i < num; ++i) {
        buf[i + 0 * num] = -i - 0.1;
        buf[i + 1 * num] = -i - 0.2;
        buf[i + 2 * num] = -i - 0.3;
    }

    std::copy(&buf[0], &buf[3 * num], &array[0].access_member<float, 2>());

    for (int i = 0; i < num; ++i) {
        float expectedPos0 = float(-i - 0.1);
        float expectedPos1 = float(-i - 0.2);
        float expectedPos2 = float(-i - 0.3);
        float expectedVel0 = float(2.0 + i) * 5;
        float expectedVel1 = float(2.1 + i) * 6;
        float expectedVel2 = float(2.2 + i) * 7;

        BOOST_TEST(array[i].pos()[0] == expectedPos0);
        BOOST_TEST(array[i].pos()[1] == expectedPos1);
        BOOST_TEST(array[i].pos()[2] == expectedPos2);
        BOOST_TEST(array[i].vel()[0] == expectedVel0);
        BOOST_TEST(array[i].vel()[1] == expectedVel1);
        BOOST_TEST(array[i].vel()[2] == expectedVel2);
    }

    for (int i = 0; i < num; ++i) {
        buf[i + 0 * num] = -i - 0.15;
        buf[i + 1 * num] = -i - 0.25;
        buf[i + 2 * num] = -i - 0.35;
    }

    std::copy(&buf[0], &buf[3 * num], reinterpret_cast<float*>(array[0].access_member(sizeof(float), 20)));

    for (int i = 0; i < num; ++i) {
        float expectedPos0 = float(-i - 0.1);
        float expectedPos1 = float(-i - 0.2);
        float expectedPos2 = float(-i - 0.3);
        float expectedVel0 = float(-i - 0.15);
        float expectedVel1 = float(-i - 0.25);
        float expectedVel2 = float(-i - 0.35);

        BOOST_TEST(array[i].pos()[0] == expectedPos0);
        BOOST_TEST(array[i].pos()[1] == expectedPos1);
        BOOST_TEST(array[i].pos()[2] == expectedPos2);
        BOOST_TEST(array[i].vel()[0] == expectedVel0);
        BOOST_TEST(array[i].vel()[1] == expectedVel1);
        BOOST_TEST(array[i].vel()[2] == expectedVel2);
    }
}

ADD_TEST(TestNonTrivialMembers)
{
    CellWithNonTrivialMembers cell1;
    CellWithNonTrivialMembers cell2;
    cell1.map[5] = std::vector<double>(4711, 47.11);
    cell2.map[7] = std::vector<double>( 666, 66.66);
    {
        // fill memory with non-zero values...
        soa_array<Particle, 2000> array(1200);
        std::fill(array.data(), array.data() + array.size(), char(1));
    }
    int counter = DestructionCounterClass::countDestruct;
    {
        soa_array<CellWithNonTrivialMembers, 30> array(20);
        // ...so that deallocation of memory upon assignment of maps
        // here will fail. Memory initialized to 0 might make the maps
        // inside not run free() at all). The effect would be that the
        // code "accidentally" works.
        array[10] = cell1;
        array[10] = cell2;
    }
    // ensure d-tor got called
    size_t expected = 30 + 1 + counter;
    BOOST_TEST(expected == DestructionCounterClass::countDestruct);
}

ADD_TEST(TestNonTrivialMembers2)
{
    CellWithNonTrivialMembers cell1;
    cell1.map[5] = std::vector<double>(4711, 47.11);
    CellWithNonTrivialMembers cell2;
    cell1.map[7] = std::vector<double>(666, 1.1);
    {
        soa_array<CellWithNonTrivialMembers, 200> array1(30);
        soa_array<CellWithNonTrivialMembers, 300> array2(30);

        array1[69] = cell1;
        array2 = array1;
        // this ensures no bit-wise copy was done in the assignment
        // above. It it had been done then the two copy assignments
        // below would cause a double free error below:
        array1[69] = cell2;
        array2[69] = cell2;
    }
}

ADD_TEST(TestNonTrivialMembers3)
{
    CellWithNonTrivialMembers cell1;
    cell1.map[5] = std::vector<double>(4711, 47.11);
    CellWithNonTrivialMembers cell2;
    cell1.map[7] = std::vector<double>(666, 1.1);
    {
        soa_array<CellWithNonTrivialMembers, 200> array1(30);
        array1[69] = cell1;
        soa_array<CellWithNonTrivialMembers, 300> array2(array1);

        // this ensures no bit-wise copy was done in the assignment
        // above. It it had been done then the two copy assignments
        // below would cause a double free error below:
        array1[69] = cell2;
        array2[69] = cell2;
    }
}

ADD_TEST(TestSwap)
{
    soa_array<Particle, 20> array1(20);
    soa_array<Particle, 20> array2(10);
    for (int i = 0; i < 20; ++i) {
        array1[i].posX() = i;
    }
    for (int i = 0; i < 10; ++i) {
        array2[i].posX() = -1;
    }

    using std::swap;
    swap(array1, array2);

    BOOST_TEST(10 == array1.size());
    BOOST_TEST(20 == array2.size());

    for (int i = 0; i < 20; ++i) {
        BOOST_TEST(i == array2[i].posX());
    }
    for (int i = 0; i < 10; ++i) {
        BOOST_TEST(-1 == array1[i].posX());
    }
}

ADD_TEST(TestCopyConstructor1)
{
    soa_array<Particle, 20> array1(10);
    for (int i = 0; i < 10; ++i) {
        array1[i].posX() = 10 + i;
        array1[i].posY() = 20 + i;
        array1[i].posZ() = 30 + i;
    }

    soa_array<Particle, 10> array2(array1);

    for (int i = 0; i < 10; ++i) {
        array1[i].posX() = -1;
        array1[i].posY() = -1;
        array1[i].posZ() = -1;
    }

    BOOST_TEST(10 == array2.size());
    for (int i = 0; i < 10; ++i) {
        BOOST_TEST(array2[i].posX() == (10 + i));
        BOOST_TEST(array2[i].posY() == (20 + i));
        BOOST_TEST(array2[i].posZ() == (30 + i));
    }
}

ADD_TEST(TestCopyConstructor2)
{
    soa_array<Particle, 20> array1(10);
    for (int i = 0; i < 10; ++i) {
        array1[i].posX() = 10 + i;
        array1[i].posY() = 20 + i;
        array1[i].posZ() = 30 + i;
    }

    const soa_array<Particle, 20>& array_reference(array1);
    soa_array<Particle, 10> array2(array_reference);

    for (int i = 0; i < 10; ++i) {
        array1[i].posX() = -1;
        array1[i].posY() = -1;
        array1[i].posZ() = -1;
    }

    BOOST_TEST(10 == array2.size());
    for (int i = 0; i < 10; ++i) {
        BOOST_TEST(array2[i].posX() == (10 + i));
        BOOST_TEST(array2[i].posY() == (20 + i));
        BOOST_TEST(array2[i].posZ() == (30 + i));
    }
}

ADD_TEST(TestClear)
{
    soa_array<Particle, 31> array(10);
    BOOST_TEST_EQ(10, array.size());

    array.clear();
    BOOST_TEST_EQ(0, array.size());

    array << Particle();
    BOOST_TEST_EQ(1, array.size());
}

ADD_TEST(TestCapacity)
{
    soa_array<Particle, 33> array(13);
    BOOST_TEST_EQ(33, array.capacity());

    array.clear();
    BOOST_TEST_EQ(33, array.capacity());
}

ADD_TEST(TestBackAndPopBack)
{
    soa_array<Particle, 22> array;
    array << Particle( 1,  2,  3,  4,  5,  6,  7,  8);
    array << Particle(11, 12, 13, 14, 15, 16, 17, 18);
    BOOST_TEST_EQ(2, array.size());

    BOOST_TEST_EQ(array.back().posX(),   11);
    BOOST_TEST_EQ(array.back().posY(),   12);
    BOOST_TEST_EQ(array.back().posZ(),   13);
    BOOST_TEST_EQ(array.back().velX(),   14);
    BOOST_TEST_EQ(array.back().velY(),   15);
    BOOST_TEST_EQ(array.back().velZ(),   16);
    BOOST_TEST_EQ(array.back().charge(), 17);
    BOOST_TEST_EQ(array.back().mass(),   18);

    array.pop_back();
    BOOST_TEST_EQ(1, array.size());

    BOOST_TEST_EQ(array.back().posX(),   1);
    BOOST_TEST_EQ(array.back().posY(),   2);
    BOOST_TEST_EQ(array.back().posZ(),   3);
    BOOST_TEST_EQ(array.back().velX(),   4);
    BOOST_TEST_EQ(array.back().velY(),   5);
    BOOST_TEST_EQ(array.back().velZ(),   6);
    BOOST_TEST_EQ(array.back().charge(), 7);
    BOOST_TEST_EQ(array.back().mass(),   8);

    array.pop_back();
    BOOST_TEST_EQ(0, array.size());
}

ADD_TEST(TestBeginEnd)
{
    soa_array<Particle, 22> array;
    BOOST_TEST_EQ(array.begin(), array.end());

    array << Particle( 1,  2,  3,  4,  5,  6,  7,  8);
    array << Particle(11, 12, 13, 14, 15, 16, 17, 18);
    BOOST_TEST_EQ(2, array.size());

    soa_array<Particle, 22>::iterator i = array.begin();
    BOOST_TEST(i == array.begin());
    BOOST_TEST(i != array.end());

    BOOST_TEST_EQ(i.posX(),   1);
    BOOST_TEST_EQ(i.posY(),   2);
    BOOST_TEST_EQ(i.posZ(),   3);
    BOOST_TEST_EQ(i.velX(),   4);
    BOOST_TEST_EQ(i.velY(),   5);
    BOOST_TEST_EQ(i.velZ(),   6);
    BOOST_TEST_EQ(i.charge(), 7);
    BOOST_TEST_EQ(i.mass(),   8);

    ++i;
    BOOST_TEST(i != array.begin());
    BOOST_TEST(i != array.end());

    BOOST_TEST_EQ(i.posX(),   11);
    BOOST_TEST_EQ(i.posY(),   12);
    BOOST_TEST_EQ(i.posZ(),   13);
    BOOST_TEST_EQ(i.velX(),   14);
    BOOST_TEST_EQ(i.velY(),   15);
    BOOST_TEST_EQ(i.velZ(),   16);
    BOOST_TEST_EQ(i.charge(), 17);
    BOOST_TEST_EQ(i.mass(),   18);

    ++i;
    BOOST_TEST(i != array.begin());
    BOOST_TEST(i == array.end());
}

}

int main(int argc, char **argv)
{
    return 0;
}
