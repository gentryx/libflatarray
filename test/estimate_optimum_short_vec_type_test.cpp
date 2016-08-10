/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <libflatarray/estimate_optimum_short_vec_type.hpp>

#include "test.hpp"

namespace LibFlatArray {

class fake_particle
{
    double pos_x;
    double pos_y;
    double pos_z;

    double vel_x;
    double vel_y;
    double vel_z;
};

class fake_accessor
{
public:
    typedef fake_particle element_type;
    static const int DIM_PROD = 2000;
};

ADD_TEST(TestArity)
{
    // expected arities are 2x of the vector-unit's bit width for some
    // architectures as we're doing loop-unrolling for those:

#  ifdef __VECTOR4DOUBLE__
    static const int expected_arity_for_double = 8;
    static const int expected_arity_for_float = 16;
#  endif

    // Dito for ARM NEON:
#  ifdef __ARM_NEON__
    // no double-intrinsics for NEON:
    static const int expected_arity_for_double = 2;
    static const int expected_arity_for_float = 8;
#  endif

    // Only the case of the IBM PC is complicated. No thanks to you,
    // history!
#  ifdef __AVX512F__
    static const int expected_arity_for_double = 16;
    static const int expected_arity_for_double = 32;
#  else
#    ifdef __AVX__
    static const int expected_arity_for_double = 8;
    static const int expected_arity_for_float = 16;
#    else
#      ifdef __SSE__
    static const int expected_arity_for_double = 4;
    static const int expected_arity_for_float = 8;
#      else
    static const int expected_arity_for_double = 2;
    static const int expected_arity_for_float = 2;
  #    endif
#    endif
#  endif

    typedef estimate_optimum_short_vec_type<double, fake_accessor>::VALUE selected_double_type;
    typedef estimate_optimum_short_vec_type<float,  fake_accessor>::VALUE selected_float_type;
    int actual_double = selected_double_type::ARITY;
    int actual_float  = selected_float_type::ARITY;

    BOOST_TEST_EQ(expected_arity_for_double, actual_double);
    BOOST_TEST_EQ(expected_arity_for_float,  actual_float);
};

}

int main(int argc, char **argv)
{
    return 0;
}
