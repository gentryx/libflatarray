/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_TESTBED_EVALUATE_HPP
#define FLAT_ARRAY_TESTBED_EVALUATE_HPP

#include <libflatarray/testbed/benchmark.hpp>

namespace LibFlatArray {

class evaluate
{
public:
    evaluate(const std::string& name, const std::string& revision) :
        name(name),
        revision(revision)
    {}

    void print_header()
    {
        std::cout << "#rev              ; date                 ; host                            ; device                                          ; order   ; family                          ; species ; dimensions              ; perf        ; unit" << std::endl;
    }

    template<class BENCHMARK>
    void operator()(BENCHMARK benchmark, std::vector<int> dim, bool output = true)
    {
        if (benchmark.family().find(name, 0) == std::string::npos) {
            return;
        }

        boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
        std::stringstream buf;
        buf << now;
        std::string now_string = buf.str();
        now_string.resize(20);

        std::string device = benchmark.device();

        int hostname_length = 2048;
        std::string hostname(hostname_length, ' ');
        gethostname(&hostname[0], hostname_length);
        // cuts string at first 0 byte, required as gethostname returns 0-terminated strings
        hostname = std::string(hostname.c_str());

        double performance = benchmark.performance(dim);

        std::ostringstream pretty_dim;
        pretty_dim << "(" << dim[0];
        for (std::size_t i = 1; i < dim.size(); ++i) {
            pretty_dim << ", " << dim[i];
        }
        pretty_dim << ")";

        if (output) {
            std::cout << std::setiosflags(std::ios::left);
            std::cout << std::setw(18) << revision << "; "
                      << now_string << " ; "
                      << std::setw(32) << hostname << "; "
                      << std::setw(48) << device << "; "
                      << std::setw( 8) << benchmark.order() <<  "; "
                      << std::setw(32) << benchmark.family() <<  "; "
                      << std::setw( 8) << benchmark.species() <<  "; "
                      << std::setw(24) << pretty_dim.str() <<  "; "
                      << std::setw(12) << performance <<  "; "
                      << std::setw( 8) << benchmark.unit() << std::endl;
        }
    }

private:
    std::string name;
    std::string revision;
};

}

#endif
