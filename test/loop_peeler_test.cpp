/**
 * Copyright 2016-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

// globally disable some warnings with MSVC, that are issued not for a
// specific header, but rather for the interaction of system headers
// and LibFlatArray source:
#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif

#include <libflatarray/aligned_allocator.hpp>
#include <libflatarray/loop_peeler.hpp>
#include <libflatarray/short_vec.hpp>
#include <libflatarray/streaming_short_vec.hpp>
#include <vector>

#include "test.hpp"

template<typename SHORT_VEC>
void scaler(int& i, int endX, double *data, double factor)
{
    for (; i < endX; i += SHORT_VEC::ARITY) {
        SHORT_VEC vec(data + i);
        vec *= factor;
        (data + i) << vec;
    }
}

ADD_TEST(TestLoopPeelerFunctionality)
{
    std::vector<double, LibFlatArray::aligned_allocator<double, 64> > foo;
    for (int i = 0; i < 123; ++i) {
        foo.push_back(1000 + i);
    }

    int x = 3;
    typedef LibFlatArray::short_vec<double, 8> short_vec_type;
    LIBFLATARRAY_LOOP_PEELER(short_vec_type, int, x, 113, scaler, &foo[0], 2.5);

    for (std::size_t i = 0; i < 123; ++i) {
        double expected = 1000 + i;
        if ((i >= 3) && (i < 113)) {
            expected *= 2.5;
        }

        BOOST_TEST_EQ(expected, foo[i]);
    }
}

ADD_TEST(TestLoopPeelerInteroperabilityWithStreamingShortVecs)
{
    std::vector<double, LibFlatArray::aligned_allocator<double, 64> > foo;
    for (int i = 0; i < 1234; ++i) {
        foo.push_back(1000 + i);
    }

    int x = 13;
    typedef LibFlatArray::streaming_short_vec<double, 8> short_vec_type;
    LIBFLATARRAY_LOOP_PEELER(short_vec_type, int, x, 1113, scaler, &foo[0], 2.5);

    for (std::size_t i = 0; i < 1234; ++i) {
        double expected = 1000 + i;
        if ((i >= 13) && (i < 1113)) {
            expected *= 2.5;
        }

        BOOST_TEST_EQ(expected, foo[i]);
    }
}

#ifdef LIBFLATARRAY_WITH_CPP14
#ifndef LIBFLATARRAY_WITH_CUDA
#ifndef LIBFLATARRAY_WITH_FORCED_CPP11

ADD_TEST(TestCpp14StyleLoopPeeler)
{
    unsigned i = 5;
    unsigned end = 43;
    std::vector<double, LibFlatArray::aligned_allocator<double, 64> > foo(64, 0);

// Actually MSVC is wrong here to assume we're not referencing
// my_float in the following lamda. We're just not referencing its
// value, just the type:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4100 )
#endif

    LibFlatArray::loop_peeler<LibFlatArray::short_vec<double, 8> >(&i, end, [&foo](auto my_float, unsigned *i, unsigned end) {
            typedef decltype(my_float) FLOAT;
            for (; *i < end; *i += FLOAT::ARITY) {
                &foo[*i] << FLOAT(1.0);
            }
        });

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif


    for (std::size_t c = 0; c < 5; ++c) {
        BOOST_TEST_EQ(0.0, foo[c]);
    }
    for (std::size_t c = 5; c < 43; ++c) {
        BOOST_TEST_EQ(1.0, foo[c]);
    }
    for (std::size_t c = 43; c < 64; ++c) {
        BOOST_TEST_EQ(0.0, foo[c]);
    }
}

#endif
#endif
#endif

int main(int /* argc */, char** /* argv */)
{
    return 0;
}
