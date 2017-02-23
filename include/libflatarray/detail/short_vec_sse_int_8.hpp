/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_INT_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_INT_8_HPP

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
class short_vec<int, 8> : public short_vec_base<int, 8>
{
public:
    static const std::size_t ARITY = 8;

    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 8>& vec);

    inline
    short_vec(const int data = 0) :
        val{_mm_set1_epi32(data),
            _mm_set1_epi32(data)}
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

    inline
    short_vec(const __m128i& val1, const __m128i& val2) :
        val{val1,
            val2}
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
    short_vec(const sqrt_reference<int, 8>& other);

    inline
    void operator-=(const short_vec<int, 8>& other)
    {
        val[ 0] = _mm_sub_epi32(val[ 0], other.val[ 0]);
        val[ 1] = _mm_sub_epi32(val[ 1], other.val[ 1]);
    }

    inline
    short_vec<int, 8> operator-(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            _mm_sub_epi32(val[ 0], other.val[ 0]),
            _mm_sub_epi32(val[ 1], other.val[ 1]));
    }

    inline
    void operator+=(const short_vec<int, 8>& other)
    {
        val[ 0] = _mm_add_epi32(val[ 0], other.val[ 0]);
        val[ 1] = _mm_add_epi32(val[ 1], other.val[ 1]);
    }

    inline
    short_vec<int, 8> operator+(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            _mm_add_epi32(val[ 0], other.val[ 0]),
            _mm_add_epi32(val[ 1], other.val[ 1]));
    }

#ifdef __SSE4_1__
    inline
    void operator*=(const short_vec<int, 8>& other)
    {
        val[ 0] = _mm_mullo_epi32(val[ 0], other.val[ 0]);
        val[ 1] = _mm_mullo_epi32(val[ 1], other.val[ 1]);
    }

    inline
    short_vec<int, 8> operator*(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            _mm_mullo_epi32(val[ 0], other.val[ 0]),
            _mm_mullo_epi32(val[ 1], other.val[ 1]));
    }
#else
    inline
    void operator*=(const short_vec<int, 8>& other)
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
    }

    inline
    short_vec<int, 8> operator*(const short_vec<int, 8>& other) const
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

        return short_vec<int, 8>(result1, result2);
    }
#endif

    inline
    void operator/=(const short_vec<int, 8>& other)
    {
        val[ 0] = _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val[ 0]),
                                           _mm_cvtepi32_ps(other.val[ 0])));
        val[ 1] = _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val[ 1]),
                                           _mm_cvtepi32_ps(other.val[ 1])));
    }

    inline
    void operator/=(const sqrt_reference<int, 8>& other);

    inline
    short_vec<int, 8> operator/(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            _mm_cvttps_epi32(_mm_div_ps(
                                 _mm_cvtepi32_ps(val[ 0]),
                                 _mm_cvtepi32_ps(other.val[ 0]))),
            _mm_cvttps_epi32(_mm_div_ps(
                                 _mm_cvtepi32_ps(val[ 1]),
                                 _mm_cvtepi32_ps(other.val[ 1]))));
    }

    inline
    short_vec<int, 8> operator/(const sqrt_reference<int, 8>& other) const;

    inline
    short_vec<int, 8> sqrt() const
    {
        return short_vec<int, 8>(
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val[ 0]))),
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val[ 1]))));
    }

    inline
    void load(const int *data)
    {
        val[ 0] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + 0));
        val[ 1] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + 4));
    }

    inline
    void load_aligned(const int *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val[ 0] = _mm_load_si128(reinterpret_cast<const __m128i *>(data + 0));
        val[ 1] = _mm_load_si128(reinterpret_cast<const __m128i *>(data + 4));
    }

    inline
    void store(int *data) const
    {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + 0), val[ 0]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + 4), val[ 1]);
    }

    inline
    void store_aligned(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_si128(reinterpret_cast<__m128i *>(data + 0), val[ 0]);
        _mm_store_si128(reinterpret_cast<__m128i *>(data + 4), val[ 1]);
    }

    inline
    void store_nt(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data + 0), val[ 0]);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data + 4), val[ 1]);
    }

#ifdef __SSE4_1__
    inline
    void gather(const int *ptr, const int *offsets)
    {
        val[ 0] = _mm_insert_epi32(val[ 0], ptr[offsets[0]], 0);
        val[ 0] = _mm_insert_epi32(val[ 0], ptr[offsets[1]], 1);
        val[ 0] = _mm_insert_epi32(val[ 0], ptr[offsets[2]], 2);
        val[ 0] = _mm_insert_epi32(val[ 0], ptr[offsets[3]], 3);

        val[ 1] = _mm_insert_epi32(val[ 1], ptr[offsets[4]], 0);
        val[ 1] = _mm_insert_epi32(val[ 1], ptr[offsets[5]], 1);
        val[ 1] = _mm_insert_epi32(val[ 1], ptr[offsets[6]], 2);
        val[ 1] = _mm_insert_epi32(val[ 1], ptr[offsets[7]], 3);
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = _mm_extract_epi32(val[ 0], 0);
        ptr[offsets[1]] = _mm_extract_epi32(val[ 0], 1);
        ptr[offsets[2]] = _mm_extract_epi32(val[ 0], 2);
        ptr[offsets[3]] = _mm_extract_epi32(val[ 0], 3);

        ptr[offsets[4]] = _mm_extract_epi32(val[ 1], 0);
        ptr[offsets[5]] = _mm_extract_epi32(val[ 1], 1);
        ptr[offsets[6]] = _mm_extract_epi32(val[ 1], 2);
        ptr[offsets[7]] = _mm_extract_epi32(val[ 1], 3);
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
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = _mm_cvtsi128_si32(val[ 0]);
        ptr[offsets[1]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 0], _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[2]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 0], _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[3]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 0], _MM_SHUFFLE(2,1,0,3)));

        ptr[offsets[4]] = _mm_cvtsi128_si32(val[ 1]);
        ptr[offsets[5]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 1], _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[6]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 1], _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[7]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val[ 1], _MM_SHUFFLE(2,1,0,3)));
    }
#endif

private:
    __m128i val[2];
};

inline
void operator<<(int *data, const short_vec<int, 8>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<int, 8>
{
public:
    template<typename OTHER_CARGO, std::size_t OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<int, 8>& vec) :
        vec(vec)
    {}

private:
    short_vec<int, 8> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<int, 8>::short_vec(const sqrt_reference<int, 8>& other) :
    val{_mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 0]))),
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 1])))}
{}

inline
void short_vec<int, 8>::operator/=(const sqrt_reference<int, 8>& other)
{
    val[ 0] = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val[ 0]),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 0]))));
    val[ 1] = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val[ 1]),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 1]))));
}

inline
short_vec<int, 8> short_vec<int, 8>::operator/(const sqrt_reference<int, 8>& other) const
{
    return short_vec<int, 8>(
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val[ 0]),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 0])))),
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val[ 1]),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val[ 1])))));
}

inline
sqrt_reference<int, 8> sqrt(const short_vec<int, 8>& vec)
{
    return sqrt_reference<int, 8>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<int, 8>& vec)
{
    const int *data1 = reinterpret_cast<const int *>(&vec.val[ 0]);
    const int *data2 = reinterpret_cast<const int *>(&vec.val[ 1]);
    __os << "["
         << data1[0] << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3] << ", "
         << data2[0] << ", " << data2[1]  << ", " << data2[2]  << ", " << data2[3]
         << "]";
    return __os;
}

}

#endif

#endif
