/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_TESTBED_GPU_BENCHMARK_HPP
#define FLAT_ARRAY_TESTBED_GPU_BENCHMARK_HPP

#include <libflatarray/testbed/benchmark.hpp>

namespace LibFlatArray {

class gpu_benchmark : benchmark
{
public:
    std::string order()
    {
        return "GPU";
    }

    std::string device()
    {
        int cudaDevice = cudaGetDevice();
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, cudaDevice);
        std::string cudaDeviceID = properties.name;

        return cudaDeviceID;
    }
};


}

#endif
