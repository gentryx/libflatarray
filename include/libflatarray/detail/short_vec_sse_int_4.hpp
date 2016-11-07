/**
 * Copyright 2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_INT_4_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_INT_4_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE2) ||            \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE4_1) ||          \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX2) ||            \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX512F)

#include <emmintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>
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
class short_vec<int, 4>
{
public:
    static const std::size_t ARITY = 4;

    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 4>& vec);

    inline
    short_vec(const int data = 0) :
        val1(_mm_set1_epi32(data))
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

    inline
    short_vec(const __m128i& val1) :
        val1(val1)
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
    short_vec(const sqrt_reference<int, 4>& other);

    inline
    void operator-=(const short_vec<int, 4>& other)
    {
        val1 = _mm_sub_epi32(val1, other.val1);
    }

    inline
    short_vec<int, 4> operator-(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            _mm_sub_epi32(val1, other.val1));
    }

    inline
    void operator+=(const short_vec<int, 4>& other)
    {
        val1 = _mm_add_epi32(val1, other.val1);
    }

    inline
    short_vec<int, 4> operator+(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            _mm_add_epi32(val1, other.val1));
    }

#ifdef __SSE4_1__
    inline
    void operator*=(const short_vec<int, 4>& other)
    {
        val1 = _mm_mullo_epi32(val1, other.val1);
    }

    inline
    short_vec<int, 4> operator*(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            _mm_mullo_epi32(val1, other.val1));
    }
#else
    inline
    void operator*=(const short_vec<int, 4>& other)
    {
        // see: https://software.intel.com/en-us/forums/intel-c-compiler/topic/288768
        __m128i tmp1 = _mm_mul_epu32(val1, other.val1);
        __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(val1, 4),
                                     _mm_srli_si128(other.val1, 4));
        val1 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));
    }

    inline
    short_vec<int, 4> operator*(const short_vec<int, 4>& other) const
    {
        // see: https://software.intel.com/en-us/forums/intel-c-compiler/topic/288768
        __m128i tmp1 = _mm_mul_epu32(val1, other.val1);
        __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(val1, 4),
                                     _mm_srli_si128(other.val1, 4));
        return short_vec<int, 4>(
            _mm_unpacklo_epi32(
                _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
                _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0))));
    }
#endif

    inline
    void operator/=(const short_vec<int, 4>& other)
    {
        val1 = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val1),
                                          _mm_cvtepi32_ps(other.val1)));
    }

    inline
    void operator/=(const sqrt_reference<int, 4>& other);

    inline
    short_vec<int, 4> operator/(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            _mm_cvttps_epi32(_mm_div_ps(
                                 _mm_cvtepi32_ps(val1),
                                 _mm_cvtepi32_ps(other.val1))));
    }

    inline
    short_vec<int, 4> operator/(const sqrt_reference<int, 4>& other) const;

    inline
    short_vec<int, 4> sqrt() const
    {
        return short_vec<int, 4>(
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val1))));
    }

    inline
    void load(const int *data)
    {
        val1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data));
    }

    inline
    void load_aligned(const int *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val1 = _mm_load_si128(reinterpret_cast<const __m128i *>(data));
    }

    inline
    void store(int *data) const
    {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data), val1);
    }

    inline
    void store_aligned(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_si128(reinterpret_cast<__m128i *>(data), val1);
    }

    inline
    void store_nt(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data), val1);
    }

#ifdef __SSE4_1__
    inline
    void gather(const int *ptr, const int *offsets)
    {
        val1 = _mm_insert_epi32(val1, ptr[offsets[0]], 0);
        val1 = _mm_insert_epi32(val1, ptr[offsets[1]], 1);
        val1 = _mm_insert_epi32(val1, ptr[offsets[2]], 2);
        val1 = _mm_insert_epi32(val1, ptr[offsets[3]], 3);
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = _mm_extract_epi32(val1, 0);
        ptr[offsets[1]] = _mm_extract_epi32(val1, 1);
        ptr[offsets[2]] = _mm_extract_epi32(val1, 2);
        ptr[offsets[3]] = _mm_extract_epi32(val1, 3);
    }
#else
    inline
    void gather(const int *ptr, const int *offsets)
    {
        __m128i i2, i3, i4;
        val1 = _mm_cvtsi32_si128(ptr[offsets[0]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[1]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[2]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[3]]);
        val1 = _mm_unpacklo_epi32(val1, i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val1 = _mm_unpacklo_epi32(val1, i3);
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = _mm_cvtsi128_si32(val1);
        ptr[offsets[1]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val1, _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[2]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val1, _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[3]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val1, _MM_SHUFFLE(2,1,0,3)));
    }
#endif

private:
    __m128i val1;
};

inline
void operator<<(int *data, const short_vec<int, 4>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<int, 4>
{
public:
    template<typename OTHER_CARGO, std::size_t OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<int, 4>& vec) :
        vec(vec)
    {}

private:
    short_vec<int, 4> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<int, 4>::short_vec(const sqrt_reference<int, 4>& other) :
    val1(
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val1))))
{}

inline
void short_vec<int, 4>::operator/=(const sqrt_reference<int, 4>& other)
{
    val1 = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val1),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val1))));
}

inline
short_vec<int, 4> short_vec<int, 4>::operator/(const sqrt_reference<int, 4>& other) const
{
    return short_vec<int, 4>(
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val1),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val1)))));
}

inline
sqrt_reference<int, 4> sqrt(const short_vec<int, 4>& vec)
{
    return sqrt_reference<int, 4>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<int, 4>& vec)
{
    const int *data1 = reinterpret_cast<const int *>(&vec.val1);
    __os << "[" << data1[0] << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3] << "]";
    return __os;
}

}

#endif

#endif
