/**
 * Copyright 2014-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_TESTBED_EVALUATE_HPP
#define FLAT_ARRAY_TESTBED_EVALUATE_HPP

#include <libflatarray/testbed/benchmark.hpp>

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#pragma warning( disable : 4668 )
#pragma warning( disable : 4820 )
#endif

#include <ctime>
#include <iomanip>
#ifdef _WIN32
#include <windows.h>
#include <WinSock2.h>
#else
#include <unistd.h>
#endif

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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

#ifdef _WIN32
        // this charade is based on https://msdn.microsoft.com/en-us/library/windows/desktop/ms724928(v=vs.85).aspx
        FILETIME fileTime;
        GetSystemTimeAsFileTime(&fileTime);

        ULARGE_INTEGER systemTime;
        systemTime.LowPart = fileTime.dwLowDateTime;
        systemTime.HighPart = fileTime.dwHighDateTime;

        SYSTEMTIME epoch;
        epoch.wYear = 1970;
        epoch.wMonth = 1;
        epoch.wDayOfWeek = 4;
        epoch.wDay = 1;
        epoch.wHour = 0;
        epoch.wMinute = 0;
        epoch.wSecond = 1;
        epoch.wMilliseconds = 0;
        FILETIME epochFileTime;
        SystemTimeToFileTime(&epoch, &epochFileTime);

        ULARGE_INTEGER epochULargeInteger;
        epochULargeInteger.LowPart = epochFileTime.dwLowDateTime;
        epochULargeInteger.HighPart = epochFileTime.dwHighDateTime;

        time_t secondsSinceEpoch = static_cast<time_t>(systemTime.QuadPart - epochULargeInteger.QuadPart);
#else
        timeval t;
        gettimeofday(&t, 0);
        time_t secondsSinceEpoch = t.tv_sec;
#endif

        tm timeSpec;
#ifdef _WIN32
        gmtime_s(&timeSpec, &secondsSinceEpoch);
#else
        gmtime_r(&secondsSinceEpoch, &timeSpec);
#endif
        char buf[1024];
        strftime(buf, 1024, "%Y-%b-%d %H:%M:%S", &timeSpec);

        std::string now_string = buf;
        std::string device = benchmark.device();

        int hostname_length = 2048;
        std::string hostname(static_cast<std::size_t>(hostname_length), ' ');
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
