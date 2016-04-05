/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/detail/lightweight_test.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <libflatarray/cuda_array.hpp>

#include "test.hpp"

using namespace LibFlatArray;

ADD_TEST(basic)
{
    std::vector<double> hostVec1(30, -1);
    std::vector<double> hostVec2(30, -2);
    std::vector<double> hostVec3(30, -3);
    std::vector<double> hostVec4(30, -4);

    for (int i = 0; i < 30; ++i) {
        hostVec1[i] = i + 0.5;
        BOOST_TEST(-2 == hostVec2[i]);
        BOOST_TEST(-3 == hostVec3[i]);
        BOOST_TEST(-4 == hostVec4[i]);
    }

    CUDAArray<double> deviceArray1(&hostVec1[0], 30);
    CUDAArray<double> deviceArray2(hostVec1);
    CUDAArray<double> deviceArray3(deviceArray1);
    CUDAArray<double> deviceArray4;
    deviceArray4 = CUDAArray<double>(30);
    deviceArray4.load(&hostVec1[0]);

    deviceArray2.save(&hostVec2[0]);
    deviceArray3.save(&hostVec3[0]);
    deviceArray4.save(&hostVec4[0]);

    for (int i = 0; i < 30; ++i) {
        double expected = i + 0.5;
        BOOST_TEST(expected == hostVec2[i]);
        BOOST_TEST(expected == hostVec3[i]);
        BOOST_TEST(expected == hostVec4[i]);
    }

    BOOST_TEST(deviceArray1.data() != deviceArray2.data());
    BOOST_TEST(deviceArray1.data() != deviceArray3.data());
    BOOST_TEST(deviceArray1.data() != deviceArray3.data());

    BOOST_TEST(deviceArray2.data() != deviceArray3.data());
    BOOST_TEST(deviceArray2.data() != deviceArray4.data());

    BOOST_TEST(deviceArray3.data() != deviceArray4.data());

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(error) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

ADD_TEST(initialization)
{
    int value = 4711;
    CUDAArray<int> deviceArray(3, value);

    std::vector<int> hostVec(3);
    deviceArray.save(&hostVec[0]);

    BOOST_TEST(hostVec[0] == 4711);
    BOOST_TEST(hostVec[1] == 4711);
    BOOST_TEST(hostVec[2] == 4711);
}

ADD_TEST(resize_after_assignment)
{
    CUDAArray<double> deviceArray1(20, 12.34);
    CUDAArray<double> deviceArray2(30, 666);
    CUDAArray<double> deviceArray3(25, 31);

    std::vector<double> hostVec(30);
    deviceArray2.save(hostVec.data());

    for (int i = 0; i < 30; ++i) {
        BOOST_TEST(hostVec[i] == 666);
    }

    deviceArray1 = deviceArray2;
    deviceArray2 = deviceArray3;

    BOOST_TEST(deviceArray1.size()     == 30);
    BOOST_TEST(deviceArray1.capacity() == 30);

    BOOST_TEST(deviceArray2.size()     == 25);
    BOOST_TEST(deviceArray2.capacity() == 30);

    hostVec = std::vector<double>(30, -1);
    deviceArray1.save(hostVec.data());

    for (int i = 0; i < 30; ++i) {
        BOOST_TEST(hostVec[i] == 666);
    }

    deviceArray2.save(hostVec.data());

    for (int i = 0; i < 25; ++i) {
        BOOST_TEST(hostVec[i] == 31);
    }

    BOOST_TEST(deviceArray1.data() != deviceArray2.data());
    BOOST_TEST(deviceArray1.data() != deviceArray3.data());
    BOOST_TEST(deviceArray2.data() != deviceArray3.data());
}

ADD_TEST(resize)
{
    CUDAArray<double> deviceArray(200, 1.3);
    BOOST_TEST(200 == deviceArray.size());
    BOOST_TEST(200 == deviceArray.capacity());

    deviceArray.resize(150);
    BOOST_TEST(150 == deviceArray.size());
    BOOST_TEST(200 == deviceArray.capacity());

    {
        std::vector<double> hostVec(250, 10);
        deviceArray.save(hostVec.data());

        for (int i = 0; i < 150; ++i) {
            BOOST_TEST(hostVec[i] == 1.3);
        }
        for (int i = 150; i < 250; ++i) {
            BOOST_TEST(hostVec[i] == 10);
        }
    }

    deviceArray.resize(250, 27);
    BOOST_TEST(250 == deviceArray.size());
    BOOST_TEST(250 == deviceArray.capacity());

    {
        std::vector<double> hostVec(250, -1);
        deviceArray.save(hostVec.data());

        for (int i = 0; i < 150; ++i) {
            BOOST_TEST(hostVec[i] == 1.3);
        }
        for (int i = 150; i < 250; ++i) {
            BOOST_TEST(hostVec[i] == 27);
        }
    }

    // ensure content is kept intact if shrunk and enlarged
    // afterwards sans default initialization:
    deviceArray.resize(10);
    deviceArray.resize(210);
    BOOST_TEST(210 == deviceArray.size());
    BOOST_TEST(250 == deviceArray.capacity());

    {
        std::vector<double> hostVec(250, -1);
        deviceArray.save(hostVec.data());

        for (int i = 0; i < 150; ++i) {
            BOOST_TEST(hostVec[i] == 1.3);
        }
        for (int i = 150; i < 210; ++i) {
            BOOST_TEST(hostVec[i] == 27);
        }
    }
}

ADD_TEST(resize2)
{
    CUDAArray<double> array(10, 5);
    array.resize(20, 6);
    array.resize(15, 7);
    double v = 8.0;
    array.resize(20, v);

    std::vector<double> vec(20);
    array.save(vec.data());

    for (int i = 0; i < 10; ++i) {
        BOOST_TEST(vec[i] == 5);
    }
    for (int i = 10; i < 15; ++i) {
        BOOST_TEST(vec[i] == 6);
    }
    for (int i = 15; i < 20; ++i) {
        BOOST_TEST(vec[i] == 8);
    }
}

ADD_TEST(reserve)
{
    CUDAArray<double> deviceArray(31, 1.3);
    BOOST_TEST(31 == deviceArray.size());
    BOOST_TEST(31 == deviceArray.capacity());

    deviceArray.reserve(55);
    BOOST_TEST(31 == deviceArray.size());
    BOOST_TEST(55 == deviceArray.capacity());

    std::vector<double> hostVec(31, -1);
    deviceArray.save(hostVec.data());

    for (int i = 0; i < 31; ++i) {
        BOOST_TEST(hostVec[i] == 1.3);
    }
}

int main(int argc, char **argv)
{
    return 0;
}
