/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <iostream>
#include <stdexcept>
#include <vector>
#include <libflatarray/cuda_array.hpp>

#include "test.hpp"

using namespace LibFlatArray;

ADD_TEST(basic)
{
    std::vector<double> host_vec1(30, -1);
    std::vector<double> host_vec2(30, -2);
    std::vector<double> host_vec3(30, -3);
    std::vector<double> host_vec4(30, -4);

    for (int i = 0; i < 30; ++i) {
        host_vec1[i] = i + 0.5;
        BOOST_TEST(-2 == host_vec2[i]);
        BOOST_TEST(-3 == host_vec3[i]);
        BOOST_TEST(-4 == host_vec4[i]);
    }

    cuda_array<double> device_array1(&host_vec1[0], 30);
    cuda_array<double> device_array2(host_vec1);
    cuda_array<double> device_array3(device_array1);
    cuda_array<double> device_array4;
    device_array4 = cuda_array<double>(30);
    device_array4.load(&host_vec1[0]);

    device_array2.save(&host_vec2[0]);
    device_array3.save(&host_vec3[0]);
    device_array4.save(&host_vec4[0]);

    for (int i = 0; i < 30; ++i) {
        double expected = i + 0.5;
        BOOST_TEST(expected == host_vec2[i]);
        BOOST_TEST(expected == host_vec3[i]);
        BOOST_TEST(expected == host_vec4[i]);
    }

    BOOST_TEST(device_array1.data() != device_array2.data());
    BOOST_TEST(device_array1.data() != device_array3.data());
    BOOST_TEST(device_array1.data() != device_array3.data());

    BOOST_TEST(device_array2.data() != device_array3.data());
    BOOST_TEST(device_array2.data() != device_array4.data());

    BOOST_TEST(device_array3.data() != device_array4.data());

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(error) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

ADD_TEST(initialization)
{
    int value = 4711;
    cuda_array<int> device_array(3, value);

    std::vector<int> host_vec(3);
    device_array.save(&host_vec[0]);

    BOOST_TEST(host_vec[0] == 4711);
    BOOST_TEST(host_vec[1] == 4711);
    BOOST_TEST(host_vec[2] == 4711);
}

ADD_TEST(resize_after_assignment)
{
    cuda_array<double> device_array1(20, 12.34);
    cuda_array<double> device_array2(30, 666);
    cuda_array<double> device_array3(25, 31);

    std::vector<double> host_vec(30);
    device_array2.save(host_vec.data());

    for (int i = 0; i < 30; ++i) {
        BOOST_TEST(host_vec[i] == 666);
    }

    device_array1 = device_array2;
    device_array2 = device_array3;

    BOOST_TEST(device_array1.size()     == 30);
    BOOST_TEST(device_array1.capacity() == 30);

    BOOST_TEST(device_array2.size()     == 25);
    BOOST_TEST(device_array2.capacity() == 30);

    host_vec = std::vector<double>(30, -1);
    device_array1.save(host_vec.data());

    for (int i = 0; i < 30; ++i) {
        BOOST_TEST(host_vec[i] == 666);
    }

    device_array2.save(host_vec.data());

    for (int i = 0; i < 25; ++i) {
        BOOST_TEST(host_vec[i] == 31);
    }

    BOOST_TEST(device_array1.data() != device_array2.data());
    BOOST_TEST(device_array1.data() != device_array3.data());
    BOOST_TEST(device_array2.data() != device_array3.data());
}

ADD_TEST(resize)
{
    cuda_array<double> device_array(200, 1.3);
    BOOST_TEST(200 == device_array.size());
    BOOST_TEST(200 == device_array.capacity());

    device_array.resize(150);
    BOOST_TEST(150 == device_array.size());
    BOOST_TEST(200 == device_array.capacity());

    {
        std::vector<double> host_vec(250, 10);
        device_array.save(host_vec.data());

        for (int i = 0; i < 150; ++i) {
            BOOST_TEST(host_vec[i] == 1.3);
        }
        for (int i = 150; i < 250; ++i) {
            BOOST_TEST(host_vec[i] == 10);
        }
    }

    device_array.resize(250, 27);
    BOOST_TEST(250 == device_array.size());
    BOOST_TEST(250 == device_array.capacity());

    {
        std::vector<double> host_vec(250, -1);
        device_array.save(host_vec.data());

        for (int i = 0; i < 150; ++i) {
            BOOST_TEST(host_vec[i] == 1.3);
        }
        for (int i = 150; i < 250; ++i) {
            BOOST_TEST(host_vec[i] == 27);
        }
    }

    // ensure content is kept intact if shrunk and enlarged
    // afterwards sans default initialization:
    device_array.resize(10);
    device_array.resize(210);
    BOOST_TEST(210 == device_array.size());
    BOOST_TEST(250 == device_array.capacity());

    {
        std::vector<double> host_vec(250, -1);
        device_array.save(host_vec.data());

        for (int i = 0; i < 150; ++i) {
            BOOST_TEST(host_vec[i] == 1.3);
        }
        for (int i = 150; i < 210; ++i) {
            BOOST_TEST(host_vec[i] == 27);
        }
    }
}

ADD_TEST(resize2)
{
    cuda_array<double> array(10, 5);
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
    cuda_array<double> device_array(31, 1.3);
    BOOST_TEST(31 == device_array.size());
    BOOST_TEST(31 == device_array.capacity());

    device_array.reserve(55);
    BOOST_TEST(31 == device_array.size());
    BOOST_TEST(55 == device_array.capacity());

    std::vector<double> host_vec(31, -1);
    device_array.save(host_vec.data());

    for (int i = 0; i < 31; ++i) {
        BOOST_TEST(host_vec[i] == 1.3);
    }
}

int main(int argc, char **argv)
{
    return 0;
}
