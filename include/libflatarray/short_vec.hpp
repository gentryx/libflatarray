/**
 * Copyright 2014-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SHORT_VEC_HPP
#define FLAT_ARRAY_SHORT_VEC_HPP

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <cstdlib>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibFlatArray {

template<typename CARGO, std::size_t ARITY>
class short_vec;

template<typename CARGO, std::size_t ARITY>
inline short_vec<CARGO, ARITY> operator+(CARGO a, const short_vec<CARGO, ARITY>& b)
{
    return short_vec<CARGO, ARITY>(a) + b;
}

template<typename CARGO, std::size_t ARITY>
inline short_vec<CARGO, ARITY> operator-(CARGO a, const short_vec<CARGO, ARITY>& b)
{
    return short_vec<CARGO, ARITY>(a) - b;
}

template<typename CARGO, std::size_t ARITY>
inline short_vec<CARGO, ARITY> operator*(CARGO a, const short_vec<CARGO, ARITY>& b)
{
    return short_vec<CARGO, ARITY>(a) * b;
}

template<typename CARGO, std::size_t ARITY>
inline short_vec<CARGO, ARITY> operator/(CARGO a, const short_vec<CARGO, ARITY>& b)
{
    return short_vec<CARGO, ARITY>(a) / b;
}

// Don't warn about these functions being stripped from an executable
// as they're not being used, that's actually expected behavior.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

template<typename CARGO, std::size_t ARITY >
inline bool any(const short_vec<CARGO, ARITY>& vec)
{
    return vec.any();
}

inline unsigned any(unsigned mask)
{
    return mask;
}

inline unsigned short any(unsigned short mask)
{
    return mask;
}

inline unsigned char any(unsigned char mask)
{
    return mask;
}

template<typename CARGO, std::size_t ARITY >
inline CARGO get(const short_vec<CARGO, ARITY>& vec, const int i)
{
    return vec[i];
}

inline bool get(unsigned mask, const int i)
{
    return (mask >> i) & 1;
}

inline bool get(unsigned short mask, const int i)
{
    return (mask >> i) & 1;
}

inline bool get(unsigned char mask, const int i)
{
    return (mask >> i) & 1;
}

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

template<typename SHORT_VEC1, typename SHORT_VEC2>
inline
SHORT_VEC1 blend(const SHORT_VEC1& v1, const SHORT_VEC2& v2, const typename SHORT_VEC1::mask_type& mask)
{
    SHORT_VEC1 ret = v1;
    ret.blend(mask, v2);
    return ret;
}

// fixme: this is slow
// fixme: replace by horizontal sum, get rid of get() alltoggether
template<typename T, std::size_t ARITY>
inline std::size_t count_mask(const typename short_vec<T, ARITY>::mask_type& mask)
{
    if (!any(mask)) {
        return 0;
    }

    short_vec<T, ARITY> v(T(0));
    v.blend(mask, short_vec<T, ARITY>(T(1)));
    std::size_t sum = 0;

    for (std::size_t i = 0; i < ARITY; ++i) {
        sum += get(v, i);
    }

    return sum;
}

class short_vec_strategy
{
public:
    class scalar
    {
    public:
        template<typename CARGO>
        class alignment
        {
        public:
            const static int ALIGNMENT = sizeof(CARGO);
        };
    };

    class avx
    {
    public:
        template<typename CARGO>
        class alignment
        {
        public:
            const static int ALIGNMENT = 32;
        };
    };

    class avx2
    {
    public:
        template<typename CARGO>
        class alignment
        {
        public:
            const static int ALIGNMENT = 32;
        };
    };

    class avx512f
    {
    public:
        template<typename CARGO>
        class alignment
        {
        public:
            const static int ALIGNMENT = 64;
        };
    };

    class cuda
    {
    public:
        template<typename CARGO>
        class alignment
        {
        public:
            const static int ALIGNMENT = sizeof(CARGO);
        };
    };

    class qpx
    {
    public:
        template<typename CARGO>
        class alignment
        {
        public:
            const static int ALIGNMENT = 32;
        };
    };

    class sse
    {
    public:
        template<typename CARGO>
        class alignment
        {
        public:
            const static int ALIGNMENT = 16;
        };
    };

    class sse2
    {
    public:
        template<typename CARGO>
        class alignment
        {
        public:
            const static int ALIGNMENT = 16;
        };
    };

    class sse4_1
    {
    public:
        template<typename CARGO>
        class alignment
        {
        public:
            const static int ALIGNMENT = 16;
        };
    };

    class mic
    {
    public:
        template<typename CARGO>
        class alignment
        {
        public:
            const static int ALIGNMENT = 32;
        };
    };

    class neon
    {
    public:
        template<typename CARGO>
        class alignment
        {
        public:
            const static int ALIGNMENT = 16;
        };
    };
};

}

#define LIBFLATARRAY_SCALAR        10
#define LIBFLATARRAY_QPX           11
#define LIBFLATARRAY_ARM_NEON      12
#define LIBFLATARRAY_MIC           13
#define LIBFLATARRAY_AVX512F       14
#define LIBFLATARRAY_AVX           15
#define LIBFLATARRAY_AVX2          16
#define LIBFLATARRAY_SSE           17
#define LIBFLATARRAY_SSE2          18
#define LIBFLATARRAY_SSE4_1        19

#ifdef __CUDA_ARCH__
// Use only scalar short_vec implementations on CUDA devices:
#define LIBFLATARRAY_WIDEST_VECTOR_ISA LIBFLATARRAY_SCALAR
#else
// for IBM Blue Gene/Q's QPX, which is mutually exclusive to
// Intel/AMD's AVX/SSE or ARM's NEON ISAs:
#  ifdef __VECTOR4DOUBLE__
#    define LIBFLATARRAY_WIDEST_VECTOR_ISA LIBFLATARRAY_QPX
#  endif

// Dito for ARM NEON:
#  ifdef __ARM_NEON__
#    define LIBFLATARRAY_WIDEST_VECTOR_ISA LIBFLATARRAY_ARM_NEON
#  endif

#  ifndef LIBFLATARRAY_WIDEST_VECTOR_ISA
// Only the case of the IBM PC is complicated. No thanks to you,
// history!
#    ifdef __MIC__
#      define LIBFLATARRAY_WIDEST_VECTOR_ISA LIBFLATARRAY_MIC
#    else
#      ifdef __AVX512F__
#        define LIBFLATARRAY_WIDEST_VECTOR_ISA LIBFLATARRAY_AVX512F
#      else
#        ifdef __AVX2__
#          define LIBFLATARRAY_WIDEST_VECTOR_ISA LIBFLATARRAY_AVX2
#        else
#          ifdef __AVX__
#            define LIBFLATARRAY_WIDEST_VECTOR_ISA LIBFLATARRAY_AVX
#          else
#            ifdef __SSE4_1__
#              define LIBFLATARRAY_WIDEST_VECTOR_ISA LIBFLATARRAY_SSE4_1
#            else
#              ifdef __SSE2__
#                define LIBFLATARRAY_WIDEST_VECTOR_ISA LIBFLATARRAY_SSE2
#              else
#                ifdef __SSE__
#                  define LIBFLATARRAY_WIDEST_VECTOR_ISA LIBFLATARRAY_SSE
#                else
// fallback: scalar implementation always works and is still yields
// code that's easy to vectorize for the compiler:
#                  define LIBFLATARRAY_WIDEST_VECTOR_ISA LIBFLATARRAY_SCALAR
#                endif
#              endif
#            endif
#          endif
#        endif
#      endif
#    endif
#  endif

#endif

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <sstream>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#include <libflatarray/detail/short_vec_avx512_double_8.hpp>
#include <libflatarray/detail/short_vec_avx512_double_16.hpp>
#include <libflatarray/detail/short_vec_avx512_double_32.hpp>

#include <libflatarray/detail/short_vec_avx512_float_16.hpp>
#include <libflatarray/detail/short_vec_avx512_float_32.hpp>

#include <libflatarray/detail/short_vec_avx_double_4.hpp>
#include <libflatarray/detail/short_vec_avx_double_8.hpp>
#include <libflatarray/detail/short_vec_avx_double_16.hpp>
#include <libflatarray/detail/short_vec_avx_double_32.hpp>

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
#include <libflatarray/detail/short_vec_sse_int_32.hpp>

#include <libflatarray/detail/short_vec_avx_int_8.hpp>
#include <libflatarray/detail/short_vec_avx_int_16.hpp>
#include <libflatarray/detail/short_vec_avx_int_32.hpp>

#include <libflatarray/detail/short_vec_avx512_int_16.hpp>
#include <libflatarray/detail/short_vec_avx512_int_32.hpp>

#include <libflatarray/detail/short_vec_sse_double_2.hpp>
#include <libflatarray/detail/short_vec_sse_double_4.hpp>
#include <libflatarray/detail/short_vec_sse_double_8.hpp>
#include <libflatarray/detail/short_vec_sse_double_16.hpp>
#include <libflatarray/detail/short_vec_sse_double_32.hpp>

#include <libflatarray/detail/short_vec_sse_float_4.hpp>
#include <libflatarray/detail/short_vec_sse_float_8.hpp>
#include <libflatarray/detail/short_vec_sse_float_16.hpp>
#include <libflatarray/detail/short_vec_sse_float_32.hpp>

#include <libflatarray/detail/short_vec_qpx_double_4.hpp>
#include <libflatarray/detail/short_vec_qpx_double_8.hpp>
#include <libflatarray/detail/short_vec_qpx_double_16.hpp>
#include <libflatarray/detail/short_vec_qpx_double_32.hpp>

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
