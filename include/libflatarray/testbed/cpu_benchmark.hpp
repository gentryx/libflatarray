/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2018 Google
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_TESTBED_CPU_BENCHMARK_HPP
#define FLAT_ARRAY_TESTBED_CPU_BENCHMARK_HPP

#include <libflatarray/testbed/benchmark.hpp>

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 )
#endif

#include <fstream>
#include <iostream>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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
        const std::size_t bufferSize = 1 << 12;
        char buffer[bufferSize];

        while (file.getline(&buffer[0], bufferSize)) {
            std::vector<std::string> tokens = tokenize(buffer, ':');
            std::vector<std::string> fields = tokenize(tokens[0], '\t');

            if ((fields.size() == 1) && (fields[0] == "cpu")) {
                return tokens[1];
            }

            if ((fields.size() == 1) && (fields[0] == "model name")) {
                tokens = tokenize(tokens[1], ' ');
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
    static std::string trim(const std::string& string)
    {
        if (string.size() == 0) {
            return string;
        }

        std::size_t start = 0;
        while ((string[start] == ' ') && (start < string.size())) {
            start += 1;
        }

        std::size_t end = string.size() - 1;
        while ((string[end] == ' ') && (end > 1)) {
            end -= 1;
        }
        if ((string[end] != ' ') && (end < string.size())) {
            end += 1;
        }

        return std::string(string, start, end - start);
    }

    static std::vector<std::string> tokenize(const std::string& line, char delimiter = ';')
    {
        std::vector<std::string> ret;

        std::stringstream buf(line);
        std::string item;

        while (std::getline(buf, item, delimiter)) {
            ret.push_back(trim(item));
        }

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
