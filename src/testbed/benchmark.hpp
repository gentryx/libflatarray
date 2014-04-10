/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_TESTBED_BENCHMARK_HPP
#define FLAT_ARRAY_TESTBED_BENCHMARK_HPP

#include <boost/date_time/posix_time/posix_time.hpp>

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

    static double time()
    {
        boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
        return now.time_of_day().total_microseconds() * 1e-6;
    }

};


}

#endif
