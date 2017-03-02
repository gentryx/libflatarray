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

// #include <libflatarray/estimate_optimum_short_vec_type.hpp>

#include "test.hpp"

namespace LibFlatArray {

class fake_particle
{
public:
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

class fake_accessor2
{
public:
    typedef fake_particle element_type;
    static const int DIM_PROD = 20000000;
};

// ADD_TEST(TestArity)
// {
//     // expected arities are 2x of the vector-unit's bit width for some
//     // architectures as we're doing loop-unrolling for those:

// #  ifdef __VECTOR4DOUBLE__
//     static const int expected_arity_for_double = 8;
//     static const int expected_arity_for_float = 16;
// #  endif

//     // Dito for ARM NEON:
// #  ifdef __ARM_NEON__
//     // no double-intrinsics for NEON:
//     static const int expected_arity_for_double = 2;
//     static const int expected_arity_for_float = 8;
// #  endif

//     // Only the case of the IBM PC is complicated. No thanks to you,
//     // history!
// #  ifdef __AVX512F__
//     static const int expected_arity_for_double = 16;
//     static const int expected_arity_for_float  = 32;
// #  else
// #    ifdef __AVX__
//     static const int expected_arity_for_double = 8;
//     static const int expected_arity_for_float = 16;
// #    else
// #      ifdef __SSE__
//     static const int expected_arity_for_double = 4;
//     static const int expected_arity_for_float = 8;
// #      else
//     static const int expected_arity_for_double = 2;
//     static const int expected_arity_for_float = 2;
//   #    endif
// #    endif
// #  endif

//     typedef estimate_optimum_short_vec_type<double, fake_accessor>::VALUE selected_double_type;
//     typedef estimate_optimum_short_vec_type<float,  fake_accessor>::VALUE selected_float_type;
//     int actual_double = selected_double_type::ARITY;
//     int actual_float  = selected_float_type::ARITY;

//     BOOST_TEST_EQ(expected_arity_for_double, actual_double);
//     BOOST_TEST_EQ(expected_arity_for_float,  actual_float);
// };

// template<typename SHORT_VEC>
// class is_streaming_short_vec;

// template<typename CARGO, std::size_t ARITY>
// class is_streaming_short_vec<streaming_short_vec<CARGO, ARITY> >
// {
// public:
//     static const bool VALUE = true;
// };

// template<typename CARGO, std::size_t ARITY>
// class is_streaming_short_vec<short_vec<CARGO, ARITY> >
// {
// public:
//     static const bool VALUE = false;
// };

// ADD_TEST(TestStoreImplementation)
// {
// // Don't warn about const expressions not being flagged as such: we
// // don't have a suitable macro for such comparisons.
// #ifdef _MSC_BUILD
// #pragma warning( push )
// #pragma warning( disable : 4127 )
// #endif

//     // small problem size should yield normal stores:
//     typedef estimate_optimum_short_vec_type<double, fake_accessor>::VALUE selected_double_type;
//     typedef estimate_optimum_short_vec_type<float,  fake_accessor>::VALUE selected_float_type;

//     BOOST_TEST_EQ(is_streaming_short_vec<selected_double_type>::VALUE, false);
//     BOOST_TEST_EQ(is_streaming_short_vec<selected_float_type>::VALUE,  false);

//     // larger problem size should yield streaming stores:
//     typedef estimate_optimum_short_vec_type<double, fake_accessor2>::VALUE selected_double_type2;
//     typedef estimate_optimum_short_vec_type<float,  fake_accessor2>::VALUE selected_float_type2;

//     BOOST_TEST_EQ(is_streaming_short_vec<selected_double_type2>::VALUE, true);
//     BOOST_TEST_EQ(is_streaming_short_vec<selected_float_type2>::VALUE,  true);

// #ifdef _MSC_BUILD
// #pragma warning( pop )
// #endif

// };

}

int main(int /* argc */, char** /* argv */)
{
    return 0;
}
