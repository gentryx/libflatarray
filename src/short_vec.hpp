/**
 * Copyright 2014-2015 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SHORT_VEC_HPP
#define FLAT_ARRAY_SHORT_VEC_HPP

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

#include <libflatarray/detail/short_vec_neon_float_4.hpp>

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
