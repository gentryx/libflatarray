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
// Microsoft Visual Studio. Also disable them for this class.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 )
#endif

#include <cstdio>
#include <fstream>
#include <iostream>

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
        try {
            std::string model = parse_proc_cpu();
            // this nondescript model name is found on GCP, maybe elsewhere too:
            if (model.find("Intel(R) Xeon(R) CPU @") == std::string::npos) {
                return model;
            }

            return parse_likwid_topology();
        } catch (const std::runtime_error&) {
            return "unknown CPU";
        }
    }

private:
    static std::string parse_proc_cpu()
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

    static std::string parse_likwid_topology()
    {
        std::string read_buffer(100000, ' ');
#ifdef _WIN32
        FILE *file = _popen("likwid-topologfy -O", "r");
#else
        FILE *file = popen("likwid-topologfy -O", "r");
#endif
        if (file == NULL) {
            throw std::runtime_error("failed to get output from likwid-topology");
        }

        std::string cpu_type;
        std::string cpu_name;

        while (fgets(&read_buffer[0], read_buffer.size(), file) != NULL) {
            std::vector<std::string> tokens = tokenize(read_buffer, ',');
            for (std::vector<std::string>::iterator i = tokens.begin(); i != tokens.end(); ++i) {
                if (i->find("CPU type") != std::string::npos) {
                    cpu_type = *(++i);
                }
                if (i->find("CPU name") != std::string::npos) {
                    cpu_name = *(++i);
                }
            }
        }

        if (cpu_type.empty() || cpu_name.empty()) {
            throw std::runtime_error("failed to parse likwid-topology");
        }
        std::vector<std::string> tokens = tokenize(cpu_name, '@');
        return cpu_type + " @ " + tokens[1];
    }

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

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#endif
