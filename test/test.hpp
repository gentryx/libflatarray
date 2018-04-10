/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2017-2018 Google
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef TEST_H
#define TEST_H

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4996 )
#endif

#include <cmath>
#include <iostream>
#include <sstream>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#include <libflatarray/detail/macros.hpp>

#ifndef BOOST_TEST
// Microsoft Visual Studio doesn't define __PRETTY_FUNCTION__:
#ifdef _MSC_VER
#define BOOST_TEST(ARG) if (!(ARG)) { std::cerr << __FILE__ << "(" << __LINE__ << "): test '" << #ARG << "' failed in function '" << __FUNCSIG__ << "'" << std::endl; }
#else
#define BOOST_TEST(ARG) if (!(ARG)) { std::cerr << __FILE__ << "(" << __LINE__ << "): test '" << #ARG << "' failed in function '" << __PRETTY_FUNCTION__ << "'" << std::endl; }
#endif

#endif


#ifndef BOOST_TEST_EQ
#define BOOST_TEST_EQ(A, B) BOOST_TEST((A) == (B))
#endif

// Runner and ADD_TEST are some convenience functions to simplify
// definition of new tests. ADD_TEST will add scaffolding that causes
// the following block to be executed once the program starts.
// Advantage: tests have no longer to be manually added to main().
template<typename TEST>
class Runner
{
public:
    Runner()
    {
        TEST()();
    }
};

#define ADD_TEST(TEST_NAME)                     \
    class TEST_NAME                             \
    {                                           \
    public:                                     \
        LIBFLATARRAY_INLINE                     \
        void operator()();                      \
                                                \
    private:                                    \
        static Runner<TEST_NAME> runner;        \
    };                                          \
                                                \
    Runner<TEST_NAME> TEST_NAME::runner;        \
                                                \
    LIBFLATARRAY_INLINE                         \
    void TEST_NAME::operator()()                \


#define TEST_REAL_ACCURACY(A, B, RELATIVE_ERROR_LIMIT)                  \
    {                                                                   \
        double a = (A);                                                 \
        double b = (B);                                                 \
        double delta = std::abs(a - b);                                 \
        double relativeError = delta / std::abs(a);                     \
        if (relativeError > RELATIVE_ERROR_LIMIT) {                     \
            std::stringstream buf;                                      \
            buf << "in file "                                           \
                << __FILE__ << ":"                                      \
                << __LINE__ << ": "                                     \
                << "difference exceeds tolerance.\n"                    \
                << "   A: " << a << "\n"                                \
                << "   B: " << b << "\n"                                \
                << "   delta: " << delta << "\n"                        \
                << "   relativeError: " << relativeError << "\n";       \
            throw std::logic_error(buf.str());                          \
        }                                                               \
    }



#ifdef _MSC_BUILD
// lazy (read: bad, inexact) test for equality. we can't use stict
// equality (operator==()), as vector units may yield
// non-IEEE-compliannt results. Single-precision accuracy (i.e. ~20
// bits for the mantissa or 6 digits) shall be suffice for functional
// testing.
#  define TEST_REAL(A, B)                                       \
    __pragma( warning( push ) )                                 \
    __pragma( warning( disable : 4710 ) )                       \
    TEST_REAL_ACCURACY(A, B, 0.000001)                          \
    __pragma( warning( pop ) )
#else
// lazy (read: bad, inexact) test for equality. we can't use stict
// equality (operator==()), as vector units may yield
// non-IEEE-compliannt results. Single-precision accuracy (i.e. ~20
// bits for the mantissa or 6 digits) shall be suffice for functional
// testing.
#  define TEST_REAL(A, B)                       \
    TEST_REAL_ACCURACY(A, B, 0.000001)
#endif

#endif
