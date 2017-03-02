/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_TESTBED_GPU_BENCHMARK_HPP
#define FLAT_ARRAY_TESTBED_GPU_BENCHMARK_HPP

#include <libflatarray/testbed/benchmark.hpp>

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <cuda.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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
        int cudaDevice;
        cudaGetDevice(&cudaDevice);
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, cudaDevice);
        std::string cudaDeviceID = properties.name;

        return cudaDeviceID;
    }
};


}

#endif
