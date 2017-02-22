/**
 * Copyright 2016-2017 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_INT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_INT_32_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE2) ||            \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE4_1) ||          \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX)

#include <emmintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>
#include <libflatarray/short_vec_base.hpp>
#include <iostream>

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

namespace LibFlatArray {

template<typename CARGO, std::size_t ARITY>
class short_vec;

template<typename CARGO, std::size_t ARITY>
class sqrt_reference;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<int, 32> : public short_vec_base<int, 32>
{
public:
    static const std::size_t ARITY = 32;

    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 32>& vec);

    inline
    short_vec(const int data = 0) :
        val{_mm_set1_epi32(data),
            _mm_set1_epi32(data),
            _mm_set1_epi32(data),
            _mm_set1_epi32(data),
            _mm_set1_epi32(data),
            _mm_set1_epi32(data),
            _mm_set1_epi32(data),
            _mm_set1_epi32(data)}
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

    inline
    short_vec(
        const __m128i& val1,
        const __m128i& val2,
        const __m128i& val3,
        const __m128i& val4,
        const __m128i& val5,
        const __m128i& val6,
        const __m128i& val7,
        const __m128i& val8) :
        val{val1,
            val2,
            val3,
            val4,
            val5,
            val6,
            val7,
            val8}
    {}

#ifdef LIBFLATARRAY_WITH_CPP14
    inline
    short_vec(const std::initializer_list<int>& il)
    {
        const int *ptr = static_cast<const int *>(&(*il.begin()));
        load(ptr);
    }
#endif

    inline
    short_vec(const sqrt_reference<int, 32>& other);

    inline
    void operator-=(const short_vec<int, 32>& other)
    {
        val[ 0] = _mm_sub_epi32(val[ 0], other.val[ 0]);
        val[ 1] = _mm_sub_epi32(val[ 1], other.val[ 1]);
        val[ 2] = _mm_sub_epi32(val[ 2], other.val[ 2]);
        val[ 3] = _mm_sub_epi32(val[ 3], other.val[ 3]);
        val[ 4] = _mm_sub_epi32(val[ 4], other.val[ 4]);
        val[ 5] = _mm_sub_epi32(val[ 5], other.val[ 5]);
        val[ 6] = _mm_sub_epi32(val[ 6], other.val[ 6]);
        val[ 7] = _mm_sub_epi32(val[ 7], other.val[ 7]);
    }

    inline
    short_vec<int, 32> operator-(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm_sub_epi32(val[ 0], other.val[ 0]),
            _mm_sub_epi32(val[ 1], other.val[ 1]),
            _mm_sub_epi32(val[ 2], other.val[ 2]),
            _mm_sub_epi32(val[ 3], other.val[ 3]),
            _mm_sub_epi32(val[ 4], other.val[ 4]),
            _mm_sub_epi32(val[ 5], other.val[ 5]),
            _mm_sub_epi32(val[ 6], other.val[ 6]),
            _mm_sub_epi32(val[ 7], other.val[ 7]));
    }

    inline
    void operator+=(const short_vec<int, 32>& other)
    {
        val[ 0] = _mm_add_epi32(val[ 0], other.val[ 0]);
        val[ 1] = _mm_add_epi32(val[ 1], other.val[ 1]);
        val[ 2] = _mm_add_epi32(val[ 2], other.val[ 2]);
        val[ 3] = _mm_add_epi32(val[ 3], other.val[ 3]);
        val[ 4] = _mm_add_epi32(val[ 4], other.val[ 4]);
        val[ 5] = _mm_add_epi32(val[ 5], other.val[ 5]);
        val[ 6] = _mm_add_epi32(val[ 6], other.val[ 6]);
        val[ 7] = _mm_add_epi32(val[ 7], other.val[ 7]);
    }

    inline
    short_vec<int, 32> operator+(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm_add_epi32(val[ 0], other.val[ 0]),
            _mm_add_epi32(val[ 1], other.val[ 1]),
            _mm_add_epi32(val[ 2], other.val[ 2]),
            _mm_add_epi32(val[ 3], other.val[ 3]),
            _mm_add_epi32(val[ 4], other.val[ 4]),
            _mm_add_epi32(val[ 5], other.val[ 5]),
            _mm_add_epi32(val[ 6], other.val[ 6]),
            _mm_add_epi32(val[ 7], other.val[ 7]));
    }

#ifdef __SSE4_1__
    inline
    void operator*=(const short_vec<int, 32>& other)
    {
        val[ 0] = _mm_mullo_epi32(val[ 0], other.val[ 0]);
        val[ 1] = _mm_mullo_epi32(val[ 1], other.val[ 1]);
        val[ 2] = _mm_mullo_epi32(val[ 2], other.val[ 2]);
        val[ 3] = _mm_mullo_epi32(val[ 3], other.val[ 3]);
        val[ 4] = _mm_mullo_epi32(val[ 4], other.val[ 4]);
        val[ 5] = _mm_mullo_epi32(val[ 5], other.val[ 5]);
        val[ 6] = _mm_mullo_epi32(val[ 6], other.val[ 6]);
        val[ 7] = _mm_mullo_epi32(val[ 7], other.val[ 7]);
    }

    inline
    short_vec<int, 32> operator*(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm_mullo_epi32(val[ 0], other.val[ 0]),
            _mm_mullo_epi32(val[ 1], other.val[ 1]),
            _mm_mullo_epi32(val[ 2], other.val[ 2]),
            _mm_mullo_epi32(val[ 3], other.val[ 3]),
            _mm_mullo_epi32(val[ 4], other.val[ 4]),
            _mm_mullo_epi32(val[ 5], other.val[ 5]),
            _mm_mullo_epi32(val[ 6], other.val[ 6]),
            _mm_mullo_epi32(val[ 7], other.val[ 7]));
    }
#else
    inline
    void operator*=(const short_vec<int, 32>& other)
    {
        // see: https://software.intel.com/en-us/forums/intel-c-compiler/topic/288768
        __m128i tmp1 = _mm_mul_epu32(val[ 0], other.val[ 0]);
        __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 0], 4),
                                     _mm_srli_si128(other.val[ 0], 4));
        val[ 0] = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 1], other.val[ 1]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 1], 4),
                             _mm_srli_si128(other.val[ 1], 4));
        val[ 1] = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 2], other.val[ 2]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 2], 4),
                             _mm_srli_si128(other.val[ 2], 4));
        val[ 2] = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 3], other.val[ 3]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 3], 4),
                             _mm_srli_si128(other.val[ 3], 4));
        val[ 3] = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 4], other.val[ 4]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 4], 4),
                             _mm_srli_si128(other.val[ 4], 4));
        val[ 4] = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 5], other.val[ 5]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 5], 4),
                             _mm_srli_si128(other.val[ 5], 4));
        val[ 5] = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 6], other.val[ 6]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 6], 4),
                             _mm_srli_si128(other.val[ 6], 4));
        val[ 6] = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 7], other.val[ 7]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 7], 4),
                             _mm_srli_si128(other.val[ 7], 4));
        val[ 7] = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));
    }

    inline
    short_vec<int, 32> operator*(const short_vec<int, 32>& other) const
    {
        // see: https://software.intel.com/en-us/forums/intel-c-compiler/topic/288768
        __m128i tmp1 = _mm_mul_epu32(val[ 0], other.val[ 0]);
        __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 0], 4),
                                     _mm_srli_si128(other.val[ 0], 4));
        __m128i result1 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 1], other.val[ 1]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 1], 4),
                             _mm_srli_si128(other.val[ 1], 4));
        __m128i result2 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 2], other.val[ 2]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 2], 4),
                             _mm_srli_si128(other.val[ 2], 4));
        __m128i result3 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 3], other.val[ 3]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 3], 4),
                             _mm_srli_si128(other.val[ 3], 4));
        __m128i result4 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 4], other.val[ 4]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 4], 4),
                             _mm_srli_si128(other.val[ 4], 4));
        __m128i result5 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 5], other.val[ 5]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 5], 4),
                             _mm_srli_si128(other.val[ 5], 4));
        __m128i result6 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 6], other.val[ 6]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 6], 4),
                             _mm_srli_si128(other.val[ 6], 4));
        __m128i result7 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val[ 7], other.val[ 7]);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val[ 7], 4),
                             _mm_srli_si128(other.val[ 7], 4));
        __m128i result8 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        return short_vec<int, 32>(
            result1,
            result2,
            result3,
            result4,
            result5,
            result6,
            result7,
            result8);
    }
#endif

    inline
    void operator/=(const short_vec<int, 32>& other)
    {
        val[ 0] = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val[ 0]),
                                          _mm_cvtepi32_ps(other.val[ 0])));
        val[ 1] = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val[ 1]),
                                          _mm_cvtepi32_ps(other.val[ 1])));
        val[ 2] = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val[ 2]),
                                          _mm_cvtepi32_ps(other.val[ 2])));
        val[ 3] = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val[ 3]),
                                          _mm_cvtepi32_ps(other.val[ 3])));
        val[ 4] = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val[ 4]),
                                          _mm_cvtepi32_ps(other.val[ 4])));
        val[ 5] = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val[ 5]),
                                          _mm_cvtepi32_ps(other.val[ 5])));
        val[ 6] = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val[ 6]),
                                          _mm_cvtepi32_ps(other.val[ 6])));
        val[ 7] = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val[ 7]),
                                          _mm_cvtepi32_ps(other.val[ 7])));
    }

    inline
    void operator/=(const sqrt_reference<int, 32>& other);

    inline
    short_vec<int, 32> operator/(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm_cvttps_epi32(_mm_div_ps(
                                 _mm_cvtepi32_ps(val[ 0]),
                                 _mm_cvtepi32_ps(other.val[ 0]))),
            _mm_cvttps_epi32(_mm_div_ps(
                                 _mm_cvtepi32_ps(val[ 1]),
                                 _mm_cvtepi32_ps(other.val[ 1]))),
            _mm_cvttps_epi32(_mm_div_ps(
                                 _mm_cvtepi32_ps(val[ 2]),
                                 _mm_cvtepi32_ps(other.val[ 2]))),
            _mm_cvttps_epi32(_mm_div_ps(
                                 _mm_cvtepi32_ps(val[ 3]),
                                 _mm_cvtepi32_ps(other.val[ 3]))),
            _mm_cvttps_epi32(_mm_div_ps(
                                 _mm_cvtepi32_ps(val[ 4]),
                                 _mm_cvtepi32_ps(other.val[ 4]))),
            _mm_cvttps_epi32(_mm_div_ps(
                                 _mm_cvtepi32_ps(val[ 5]),
                                 _mm_cvtepi32_ps(other.val[ 5]))),
            _mm_cvttps_epi32(_mm_div_ps(
                                 _mm_cvtepi32_ps(val[ 6]),
                                 _mm_cvtepi32_ps(other.val[ 6]))),
            _mm_cvttps_epi32(_mm_div_ps(
                                 _mm_cvtepi32_ps(val[ 7]),
                                 _mm_cvtepi32_ps(other.val[ 7]))));
    }

    inline
    short_vec<int, 32> operator/(const sqrt_reference<int, 32>& other) const;

    inline
    short_vec<int, 32> sqrt() const
    {
        return short_vec<int, 32>(
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val[ 0]))),
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val[ 1]))),
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val[ 2]))),
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val[ 3]))),
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val[ 4]))),
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val[ 5]))),
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val[ 6]))),
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val[ 7]))));
    }

    inline
    void load(const int *data)
    {
        val[ 0] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data +  0));
        val[ 1] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data +  4));
        val[ 2] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data +  8));
        val[ 3] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + 12));
        val[ 4] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + 16));
        val[ 5] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + 20));
        val[ 6] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + 24));
        val[ 7] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + 28));
    }

    inline
    void load_aligned(const int *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val[ 0] = _mm_load_si128(reinterpret_cast<const __m128i *>(data +  0));
        val[ 1] = _mm_load_si128(reinterpret_cast<const __m128i *>(data +  4));
        val[ 2] = _mm_load_si128(reinterpret_cast<const __m128i *>(data +  8));
        val[ 3] = _mm_load_si128(reinterpret_cast<const __m128i *>(data + 12));
        val[ 4] = _mm_load_si128(reinterpret_cast<const __m128i *>(data + 16));
        val[ 5] = _mm_load_si128(reinterpret_cast<const __m128i *>(data + 20));
        val[ 6] = _mm_load_si128(reinterpret_cast<const __m128i *>(data + 24));
        val[ 7] = _mm_load_si128(reinterpret_cast<const __m128i *>(data + 28));
    }

    inline
    void store(int *data) const
    {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data +  0), val[ 0]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data +  4), val[ 1]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data +  8), val[ 2]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + 12), val[ 3]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + 16), val[ 4]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + 20), val[ 5]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + 24), val[ 6]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + 28), val[ 7]);
    }

    inline
    void store_aligned(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_si128(reinterpret_cast<__m128i *>(data +  0), val[ 0]);
        _mm_store_si128(reinterpret_cast<__m128i *>(data +  4), val[ 1]);
        _mm_store_si128(reinterpret_cast<__m128i *>(data +  8), val[ 2]);
        _mm_store_si128(reinterpret_cast<__m128i *>(data + 12), val[ 3]);
        _mm_store_si128(reinterpret_cast<__m128i *>(data + 16), val[ 4]);
        _mm_store_si128(reinterpret_cast<__m128i *>(data + 20), val[ 5]);
        _mm_store_si128(reinterpret_cast<__m128i *>(data + 24), val[ 6]);
        _mm_store_si128(reinterpret_cast<__m128i *>(data + 28), val[ 7]);
    }

    inline
    void store_nt(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data +  0), val[ 0]);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data +  4), val[ 1]);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data +  8), val[ 2]);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data + 12), val[ 3]);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data + 16), val[ 4]);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data + 20), val[ 5]);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data + 24), val[ 6]);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data + 28), val[ 7]);
    }

#ifdef __SSE4_1__
    inline
    void gather(const int *ptr, const int *offsets)
    {
        val[ 0] = _mm_insert_epi32(val[ 0], ptr[offsets[ 0]], 0);
        val[ 0] = _mm_insert_epi32(val[ 0], ptr[offsets[ 1]], 1);
        val[ 0] = _mm_insert_epi32(val[ 0], ptr[offsets[ 2]], 2);
        val[ 0] = _mm_insert_epi32(val[ 0], ptr[offsets[ 3]], 3);

        val[ 1] = _mm_insert_epi32(val[ 1], ptr[offsets[ 4]], 0);
        val[ 1] = _mm_insert_epi32(val[ 1], ptr[offsets[ 5]], 1);
        val[ 1] = _mm_insert_epi32(val[ 1], ptr[offsets[ 6]], 2);
        val[ 1] = _mm_insert_epi32(val[ 1], ptr[offsets[ 7]], 3);

        val[ 2] = _mm_insert_epi32(val[ 2], ptr[offsets[ 8]], 0);
        val[ 2] = _mm_insert_epi32(val[ 2], ptr[offsets[ 9]], 1);
        val[ 2] = _mm_insert_epi32(val[ 2], ptr[offsets[10]], 2);
        val[ 2] = _mm_insert_epi32(val[ 2], ptr[offsets[11]], 3);

        val[ 3] = _mm_insert_epi32(val[ 3], ptr[offsets[12]], 0);
        val[ 3] = _mm_insert_epi32(val[ 3], ptr[offsets[13]], 1);
        val[ 3] = _mm_insert_epi32(val[ 3], ptr[offsets[14]], 2);
        val[ 3] = _mm_insert_epi32(val[ 3], ptr[offsets[15]], 3);

        val[ 4] = _mm_insert_epi32(val[ 4], ptr[offsets[16]], 0);
        val[ 4] = _mm_insert_epi32(val[ 4], ptr[offsets[17]], 1);
        val[ 4] = _mm_insert_epi32(val[ 4], ptr[offsets[18]], 2);
        val[ 4] = _mm_insert_epi32(val[ 4], ptr[offsets[19]], 3);

        val[ 5] = _mm_insert_epi32(val[ 5], ptr[offsets[20]], 0);
        val[ 5] = _mm_insert_epi32(val[ 5], ptr[offsets[21]], 1);
        val[ 5] = _mm_insert_epi32(val[ 5], ptr[offsets[22]], 2);
        val[ 5] = _mm_insert_epi32(val[ 5], ptr[offsets[23]], 3);

        val[ 6] = _mm_insert_epi32(val[ 6], ptr[offsets[24]], 0);
        val[ 6] = _mm_insert_epi32(val[ 6], ptr[offsets[25]], 1);
        val[ 6] = _mm_insert_epi32(val[ 6], ptr[offsets[26]], 2);
        val[ 6] = _mm_insert_epi32(val[ 6], ptr[offsets[27]], 3);

        val[ 7] = _mm_insert_epi32(val[ 7], ptr[offsets[28]], 0);
        val[ 7] = _mm_insert_epi32(val[ 7], ptr[offsets[29]], 1);
        val[ 7] = _mm_insert_epi32(val[ 7], ptr[offsets[30]], 2);
        val[ 7] = _mm_insert_epi32(val[ 7], ptr[offsets[31]], 3);
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[ 0]] = _mm_extract_epi32(val[ 0], 0);
        ptr[offsets[ 1]] = _mm_extract_epi32(val[ 0], 1);
        ptr[offsets[ 2]] = _mm_extract_epi32(val[ 0], 2);
        ptr[offsets[ 3]] = _mm_extract_epi32(val[ 0], 3);

        ptr[offsets[ 4]] = _mm_extract_epi32(val[ 1], 0);
        ptr[offsets[ 5]] = _mm_extract_epi32(val[ 1], 1);
        ptr[offsets[ 6]] = _mm_extract_epi32(val[ 1], 2);
        ptr[offsets[ 7]] = _mm_extract_epi32(val[ 1], 3);

        ptr[offsets[ 8]] = _mm_extract_epi32(val[ 2], 0);
        ptr[offsets[ 9]] = _mm_extract_epi32(val[ 2], 1);
        ptr[offsets[10]] = _mm_extract_epi32(val[ 2], 2);
        ptr[offsets[11]] = _mm_extract_epi32(val[ 2], 3);

        ptr[offsets[12]] = _mm_extract_epi32(val[ 3], 0);
        ptr[offsets[13]] = _mm_extract_epi32(val[ 3], 1);
        ptr[offsets[14]] = _mm_extract_epi32(val[ 3], 2);
        ptr[offsets[15]] = _mm_extract_epi32(val[ 3], 3);

        ptr[offsets[16]] = _mm_extract_epi32(val[ 4], 0);
        ptr[offsets[17]] = _mm_extract_epi32(val[ 4], 1);
        ptr[offsets[18]] = _mm_extract_epi32(val[ 4], 2);
        ptr[offsets[19]] = _mm_extract_epi32(val[ 4], 3);

        ptr[offsets[20]] = _mm_extract_epi32(val[ 5], 0);
        ptr[offsets[21]] = _mm_extract_epi32(val[ 5], 1);
        ptr[offsets[22]] = _mm_extract_epi32(val[ 5], 2);
        ptr[offsets[23]] = _mm_extract_epi32(val[ 5], 3);

        ptr[offsets[24]] = _mm_extract_epi32(val[ 6], 0);
        ptr[offsets[25]] = _mm_extract_epi32(val[ 6], 1);
        ptr[offsets[26]] = _mm_extract_epi32(val[ 6], 2);
        ptr[offsets[27]] = _mm_extract_epi32(val[ 6], 3);

        ptr[offsets[28]] = _mm_extract_epi32(val[ 7], 0);
        ptr[offsets[29]] = _mm_extract_epi32(val[ 7], 1);
        ptr[offsets[30]] = _mm_extract_epi32(val[ 7], 2);
        ptr[offsets[31]] = _mm_extract_epi32(val[ 7], 3);
    }
#else
    inline
    void gather(const int *ptr, const int *offsets)
    {
        __m128i i2, i3, i4;
        val[ 0] = _mm_cvtsi32_si128(ptr[offsets[0]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[1]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[2]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[3]]);
        val[ 0] = _mm_unpacklo_epi32(val[ 0], i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val[ 0] = _mm_unpacklo_epi32(val[ 0], i3);

        val[ 1] = _mm_cvtsi32_si128(ptr[offsets[4]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[5]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[6]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[7]]);
        val[ 1] = _mm_unpacklo_epi32(val[ 1], i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val[ 1] = _mm_unpacklo_epi32(val[ 1], i3);

        val[ 2] = _mm_cvtsi32_si128(ptr[offsets[ 8]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[ 9]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[10]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[11]]);
        val[ 2] = _mm_unpacklo_epi32(val[ 2], i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val[ 2] = _mm_unpacklo_epi32(val[ 2], i3);

        val[ 3] = _mm_cvtsi32_si128(ptr[offsets[12]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[13]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[14]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[15]]);
        val[ 3] = _mm_unpacklo_epi32(val[ 3], i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val[ 3] = _mm_unpacklo_epi32(val[ 3], i3);

        val[ 4] = _mm_cvtsi32_si128(ptr[offsets[16]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[17]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[18]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[19]]);
        val[ 4] = _mm_unpacklo_epi32(val[ 4], i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val[ 4] = _mm_unpacklo_epi32(val[ 4], i3);

        val[ 5] = _mm_cvtsi32_si128(ptr[offsets[20]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[21]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[22]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[23]]);
        val[ 5] = _mm_unpacklo_epi32(val[ 5], i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val[ 5] = _mm_unpacklo_epi32(val[ 5], i3);

        val[ 6] = _mm_cvtsi32_si128(ptr[offsets[24]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[25]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[26]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[27]]);
        val[ 6] = _mm_unpacklo_epi32(val[ 6], i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val[ 6] = _mm_unpacklo_epi32(val[ 6], i3);

        val[ 7] = _mm_cvtsi32_si128(ptr[offsets[28]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[29]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[30]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[31]]);
        val[ 7] = _mm_unpacklo_epi32(val[ 7], i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val[ 7] = _mm_unpacklo_epi32(val[ 7], i3);
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[ 0]] = _mm_cvtsi128_si32(val[ 0]);
        ptr[offsets[ 1]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 0], _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[ 2]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 0], _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[ 3]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 0], _MM_SHUFFLE(2,1,0,3)));

        ptr[offsets[ 4]] = _mm_cvtsi128_si32(val[ 1]);
        ptr[offsets[ 5]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 1], _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[ 6]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 1], _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[ 7]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 1], _MM_SHUFFLE(2,1,0,3)));

        ptr[offsets[ 8]] = _mm_cvtsi128_si32(val[ 2]);
        ptr[offsets[ 9]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 2], _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[10]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 2], _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[11]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 2], _MM_SHUFFLE(2,1,0,3)));

        ptr[offsets[12]] = _mm_cvtsi128_si32(val[ 3]);
        ptr[offsets[13]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 3], _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[14]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 3], _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[15]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 3], _MM_SHUFFLE(2,1,0,3)));

        ptr[offsets[16]] = _mm_cvtsi128_si32(val[ 4]);
        ptr[offsets[17]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 4], _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[18]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 4], _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[19]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 4], _MM_SHUFFLE(2,1,0,3)));

        ptr[offsets[20]] = _mm_cvtsi128_si32(val[ 5]);
        ptr[offsets[21]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 5], _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[22]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 5], _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[23]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 5], _MM_SHUFFLE(2,1,0,3)));

        ptr[offsets[24]] = _mm_cvtsi128_si32(val[ 6]);
        ptr[offsets[25]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 6], _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[26]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 6], _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[27]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 6], _MM_SHUFFLE(2,1,0,3)));

        ptr[offsets[28]] = _mm_cvtsi128_si32(val[ 7]);
        ptr[offsets[29]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 7], _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[30]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 7], _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[31]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 7], _MM_SHUFFLE(2,1,0,3)));
    }
#endif

private:
    __m128i val[8];
};

inline
void operator<<(int *data, const short_vec<int, 32>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<int, 32>
{
public:
    template<typename OTHER_CARGO, std::size_t OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<int, 32>& vec) :
        vec(vec)
    {}

private:
    short_vec<int, 32> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<int, 32>::short_vec(const sqrt_reference<int, 32>& other) :
    val{
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 0]))),
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 1]))),
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 2]))),
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 3]))),
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 4]))),
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 5]))),
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 6]))),
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 7])))
            }
{}

inline
void short_vec<int, 32>::operator/=(const sqrt_reference<int, 32>& other)
{
    val[ 0] = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val[ 0]),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 0]))));
    val[ 1] = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val[ 1]),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 1]))));
    val[ 2] = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val[ 2]),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 2]))));
    val[ 3] = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val[ 3]),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 3]))));
    val[ 4] = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val[ 4]),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 4]))));
    val[ 5] = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val[ 5]),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 5]))));
    val[ 6] = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val[ 6]),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 6]))));
    val[ 7] = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val[ 7]),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 7]))));
}

inline
short_vec<int, 32> short_vec<int, 32>::operator/(const sqrt_reference<int, 32>& other) const
{
    return short_vec<int, 32>(
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val[ 0]),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 0])))),
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val[ 1]),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 1])))),
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val[ 2]),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 2])))),
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val[ 3]),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 3])))),
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val[ 4]),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 4])))),
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val[ 5]),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 5])))),
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val[ 6]),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 6])))),
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val[ 7]),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 7])))));
}

inline
sqrt_reference<int, 32> sqrt(const short_vec<int, 32>& vec)
{
    return sqrt_reference<int, 32>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<int, 32>& vec)
{
    const int *data1 = reinterpret_cast<const int *>(&vec.val[ 0]);
    const int *data2 = reinterpret_cast<const int *>(&vec.val[ 1]);
    const int *data3 = reinterpret_cast<const int *>(&vec.val[ 2]);
    const int *data4 = reinterpret_cast<const int *>(&vec.val[ 3]);
    const int *data5 = reinterpret_cast<const int *>(&vec.val[ 4]);
    const int *data6 = reinterpret_cast<const int *>(&vec.val[ 5]);
    const int *data7 = reinterpret_cast<const int *>(&vec.val[ 6]);
    const int *data8 = reinterpret_cast<const int *>(&vec.val[ 7]);
    __os << "["
         << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3] << ", "
         << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3] << ", "
         << data3[0] << ", " << data3[1] << ", " << data3[2] << ", " << data3[3] << ", "
         << data4[0] << ", " << data4[1] << ", " << data4[2] << ", " << data4[3] << ", "
         << data5[0] << ", " << data5[1] << ", " << data5[2] << ", " << data5[3] << ", "
         << data6[0] << ", " << data6[1] << ", " << data6[2] << ", " << data6[3] << ", "
         << data7[0] << ", " << data7[1] << ", " << data7[2] << ", " << data7[3] << ", "
         << data8[0] << ", " << data8[1] << ", " << data8[2] << ", " << data8[3]
         << "]";
    return __os;
}

}

#endif

#endif
