/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_TESTBED_CPU_BENCHMARK_HPP
#define FLAT_ARRAY_TESTBED_CPU_BENCHMARK_HPP

#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <libflatarray/testbed/benchmark.hpp>

namespace LibFlatArray {

class cpu_benchmark : public benchmark
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string device()
    {
        std::ifstream file("/proc/cpuinfo");
        std::size_t bufferSize = 2048;
        std::string buffer(bufferSize, ' ');
        while (file.getline(&buffer[0], bufferSize)) {
            std::vector<std::string> tokens = tokenize(buffer, ":");
            std::vector<std::string> fields = tokenize(tokens[0], " \t");

            if ((fields[0] == "model") && (fields[1] == "name")) {
                tokens = tokenize(tokens[1], " \t");
                std::string buf = join(tokens, " ");
                if (buf[buf.size() - 1] == 0) {
                    buf.resize(buf.size() - 1);
                }

                return buf;
            }
        }

        throw std::runtime_error("could not parse /proc/cpuinfo");
    }

private:
    static std::vector<std::string> tokenize(const std::string& string, const std::string& delimiters)
    {
        std::vector<std::string> ret;
        boost::split(ret, string, boost::is_any_of(delimiters), boost::token_compress_on);
        ret.erase(std::remove(ret.begin(), ret.end(), ""), ret.end());

        return ret;
    }

    static std::string join(const std::vector<std::string>& tokens, const std::string& delimiter)
    {
        std::stringstream buf;

        for (std::vector<std::string>::const_iterator i = tokens.begin(); i != tokens.end(); ++i) {
            if (i != tokens.begin()) {
                buf << delimiter;
            }
            buf << *i;
        }

        return buf.str();
    }
};


}

#endif
