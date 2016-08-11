/**
 * Copyright 2014-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SHORT_VEC_HPP
#define FLAT_ARRAY_SHORT_VEC_HPP

namespace LibFlatArray {

class short_vec_strategy
{
public:
    class scalar
    {};

    class avx
    {};

    class avx512
    {};

    class cuda
    {};

    class qpx
    {};

    class sse
    {};

    class mic
    {};

    class neon
    {};
};

}

#ifdef __CUDA_ARCH__
// Only use scalar short_vec implementations on CUDA devices:
#define LIBFLATARRAY_WIDEST_VECTOR_ISA SCALAR
#else
    // for IBM Blue Gene/Q's QPX, which is mutually exclusive to
    // Intel/AMD's AVX/SSE or ARM's NEON ISAs:
#  ifdef __VECTOR4DOUBLE__
#  define LIBFLATARRAY_WIDEST_VECTOR_ISA __VECTOR4DOUBLE__
#  endif

    // Dito for ARM NEON:
#  ifdef __ARM_NEON__
#  define LIBFLATARRAY_WIDEST_VECTOR_ISA __ARM_NEON__
#  endif

    // Only the case of the IBM PC is complicated. No thanks to you,
    // history!
#  ifdef __AVX512F__
#  define LIBFLATARRAY_WIDEST_VECTOR_ISA __AVX512F__
#  else
#    ifdef __AVX__
#    define LIBFLATARRAY_WIDEST_VECTOR_ISA __AVX__
#    else
#      ifdef __SSE__
#      define LIBFLATARRAY_WIDEST_VECTOR_ISA __SSE__
#      else
// fallback: scalar implementation always works and is still yields
// code that's easy to vectorize for the compiler:
#      define LIBFLATARRAY_WIDEST_VECTOR_ISA SCALAR
#      endif
#    endif
#  endif

#endif

#include <sstream>

#include <libflatarray/detail/short_vec_avx512_double_8.hpp>
#include <libflatarray/detail/short_vec_avx512_double_16.hpp>
#include <libflatarray/detail/short_vec_avx512_double_32.hpp>

#include <libflatarray/detail/short_vec_avx512_float_16.hpp>
#include <libflatarray/detail/short_vec_avx512_float_32.hpp>

#include <libflatarray/detail/short_vec_avx_double_4.hpp>
#include <libflatarray/detail/short_vec_avx_double_8.hpp>
#include <libflatarray/detail/short_vec_avx_double_16.hpp>

#include <libflatarray/detail/short_vec_avx_float_8.hpp>
#include <libflatarray/detail/short_vec_avx_float_16.hpp>
#include <libflatarray/detail/short_vec_avx_float_32.hpp>

#include <libflatarray/detail/short_vec_scalar_double_1.hpp>
#include <libflatarray/detail/short_vec_scalar_double_2.hpp>
#include <libflatarray/detail/short_vec_scalar_double_4.hpp>
#include <libflatarray/detail/short_vec_scalar_double_8.hpp>
#include <libflatarray/detail/short_vec_scalar_double_16.hpp>
#include <libflatarray/detail/short_vec_scalar_double_32.hpp>

#include <libflatarray/detail/short_vec_scalar_float_1.hpp>
#include <libflatarray/detail/short_vec_scalar_float_2.hpp>
#include <libflatarray/detail/short_vec_scalar_float_4.hpp>
#include <libflatarray/detail/short_vec_scalar_float_8.hpp>
#include <libflatarray/detail/short_vec_scalar_float_16.hpp>
#include <libflatarray/detail/short_vec_scalar_float_32.hpp>

#include <libflatarray/detail/short_vec_scalar_int_1.hpp>
#include <libflatarray/detail/short_vec_scalar_int_2.hpp>
#include <libflatarray/detail/short_vec_scalar_int_4.hpp>
#include <libflatarray/detail/short_vec_scalar_int_8.hpp>
#include <libflatarray/detail/short_vec_scalar_int_16.hpp>
#include <libflatarray/detail/short_vec_scalar_int_32.hpp>

#include <libflatarray/detail/short_vec_sse_int_4.hpp>
#include <libflatarray/detail/short_vec_sse_int_8.hpp>
#include <libflatarray/detail/short_vec_sse_int_16.hpp>

#include <libflatarray/detail/short_vec_avx_int_8.hpp>
#include <libflatarray/detail/short_vec_avx_int_16.hpp>
#include <libflatarray/detail/short_vec_avx_int_32.hpp>

#include <libflatarray/detail/short_vec_avx512_int_16.hpp>
#include <libflatarray/detail/short_vec_avx512_int_32.hpp>

#include <libflatarray/detail/short_vec_sse_double_2.hpp>
#include <libflatarray/detail/short_vec_sse_double_4.hpp>
#include <libflatarray/detail/short_vec_sse_double_8.hpp>

#include <libflatarray/detail/short_vec_sse_float_4.hpp>
#include <libflatarray/detail/short_vec_sse_float_8.hpp>
#include <libflatarray/detail/short_vec_sse_float_16.hpp>

#include <libflatarray/detail/short_vec_qpx_double_4.hpp>
#include <libflatarray/detail/short_vec_qpx_double_8.hpp>
#include <libflatarray/detail/short_vec_qpx_double_16.hpp>

#include <libflatarray/detail/short_vec_neon_float_4.hpp>
#include <libflatarray/detail/short_vec_neon_float_8.hpp>
#include <libflatarray/detail/short_vec_neon_float_16.hpp>
#include <libflatarray/detail/short_vec_neon_float_32.hpp>

#include <libflatarray/detail/short_vec_mic_double_8.hpp>
#include <libflatarray/detail/short_vec_mic_double_16.hpp>
#include <libflatarray/detail/short_vec_mic_double_32.hpp>
#include <libflatarray/detail/short_vec_mic_float_16.hpp>
#include <libflatarray/detail/short_vec_mic_float_32.hpp>

#endif
