/**
 * Copyright 2014-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_TESTBED_BENCHMARK_HPP
#define FLAT_ARRAY_TESTBED_BENCHMARK_HPP

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

namespace LibFlatArray {

class benchmark
{
public:
    virtual ~benchmark()
    {}

    virtual std::string order() = 0;
    virtual std::string family() = 0;
    virtual std::string species() = 0;
    virtual double performance(std::vector<int> dim) = 0;
    virtual std::string unit() = 0;
    virtual std::string device() = 0;

    static
    inline double time()
    {
#ifdef _WIN32
        LARGE_INTEGER time;
        LARGE_INTEGER freq;
        QueryPerformanceCounter(&time);
        QueryPerformanceFrequency(&freq);
        return 1.0 * time.QuadPart / freq.QuadPart;
#else
        timeval t;
        gettimeofday(&t, 0);
        return t.tv_sec + t.tv_usec * 1.0e-6;
#endif

    }

};


}

#endif
