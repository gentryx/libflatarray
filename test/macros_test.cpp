/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <libflatarray/aligned_allocator.hpp>
#include <libflatarray/macros.hpp>
#include <libflatarray/short_vec.hpp>
#include <libflatarray/streaming_short_vec.hpp>
#include <vector>

#include "test.hpp"

template<typename SHORT_VEC>
void scaler(int& i, int endX, double *data, double factor)
{
    for (; i < endX - (SHORT_VEC::ARITY - 1); i += SHORT_VEC::ARITY) {
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

    for (int i = 0; i < 123; ++i) {
        double expected = 1000 + i;
        if ((i >= 3) && (i < 113)) {
            expected *= 2.5;
        }

        BOOST_TEST(expected == foo[i]);
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

    for (int i = 0; i < 1234; ++i) {
        double expected = 1000 + i;
        if ((i >= 13) && (i < 1113)) {
            expected *= 2.5;
        }

        BOOST_TEST(expected == foo[i]);
    }
}

int main(int argc, char **argv)
{
    return 0;
}
