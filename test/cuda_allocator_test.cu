/**
 * Copyright 2013 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/detail/lightweight_test.hpp>
#include <vector>
#include <libflatarray/cuda_allocator.hpp>

#include "test.h"

using namespace LibFlatArray;

ADD_TEST(test_basic)
{
    cuda_allocator<double> allocator;

    double *devArray1 = allocator.allocate( 50);
    double *devArray2 = allocator.allocate(110);
    BOOST_TEST(devArray1 != devArray2);

    std::vector<double> hostArray1(120, -1);
    std::vector<double> hostArray2(130, -2);

    for (int i = 0; i < 50; ++i) {
        hostArray1[i] = i + 0.5;

        BOOST_TEST(hostArray2[i] == -2);
    }

    std::size_t byteSize = 50 * sizeof(double);
    cudaMemcpy(devArray1,      &hostArray1[0], byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devArray2,      devArray1,      byteSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(&hostArray2[0], devArray2,      byteSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 50; ++i) {
        double expected = i + 0.5;
        BOOST_TEST(hostArray2[i] == expected);
    }
}

int main(int argc, char **argv)
{
    return 0;
}
