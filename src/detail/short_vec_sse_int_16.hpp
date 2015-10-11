/**
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_INT_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_INT_16_HPP

#ifdef __SSE2__

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

#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

template<typename CARGO, int ARITY>
class sqrt_reference;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<int, 16>
{
public:
    static const int ARITY = 16;

    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 16>& vec);

    inline
    short_vec(const int data = 0) :
        val1(_mm_set1_epi32(data)),
        val2(_mm_set1_epi32(data)),
        val3(_mm_set1_epi32(data)),
        val4(_mm_set1_epi32(data))
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

    inline
    short_vec(const __m128i& val1, const __m128i& val2,
              const __m128i& val3, const __m128i& val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
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
    short_vec(const sqrt_reference<int, 16>& other);

    inline
    void operator-=(const short_vec<int, 16>& other)
    {
        val1 = _mm_sub_epi32(val1, other.val1);
        val2 = _mm_sub_epi32(val2, other.val2);
        val3 = _mm_sub_epi32(val3, other.val3);
        val4 = _mm_sub_epi32(val4, other.val4);
    }

    inline
    short_vec<int, 16> operator-(const short_vec<int, 16>& other) const
    {
        return short_vec<int, 16>(
            _mm_sub_epi32(val1, other.val1),
            _mm_sub_epi32(val2, other.val2),
            _mm_sub_epi32(val3, other.val3),
            _mm_sub_epi32(val4, other.val4));
    }

    inline
    void operator+=(const short_vec<int, 16>& other)
    {
        val1 = _mm_add_epi32(val1, other.val1);
        val2 = _mm_add_epi32(val2, other.val2);
        val3 = _mm_add_epi32(val3, other.val3);
        val4 = _mm_add_epi32(val4, other.val4);
    }

    inline
    short_vec<int, 16> operator+(const short_vec<int, 16>& other) const
    {
        return short_vec<int, 16>(
            _mm_add_epi32(val1, other.val1),
            _mm_add_epi32(val2, other.val2),
            _mm_add_epi32(val3, other.val3),
            _mm_add_epi32(val4, other.val4));
    }

#ifdef __SSE4_1__
    inline
    void operator*=(const short_vec<int, 16>& other)
    {
        val1 = _mm_mullo_epi32(val1, other.val1);
        val2 = _mm_mullo_epi32(val2, other.val2);
        val3 = _mm_mullo_epi32(val3, other.val3);
        val4 = _mm_mullo_epi32(val4, other.val4);
    }

    inline
    short_vec<int, 16> operator*(const short_vec<int, 16>& other) const
    {
        return short_vec<int, 16>(
            _mm_mullo_epi32(val1, other.val1),
            _mm_mullo_epi32(val2, other.val2),
            _mm_mullo_epi32(val3, other.val3),
            _mm_mullo_epi32(val4, other.val4));
    }
#else
    inline
    void operator*=(const short_vec<int, 16>& other)
    {
        // see: https://software.intel.com/en-us/forums/intel-c-compiler/topic/288768
        __m128i tmp1 = _mm_mul_epu32(val1, other.val1);
        __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(val1, 4),
                                     _mm_srli_si128(other.val1, 4));
        val1 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val2, other.val2);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val2, 4),
                             _mm_srli_si128(other.val2, 4));
        val2 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val3, other.val3);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val3, 4),
                             _mm_srli_si128(other.val3, 4));
        val3 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val4, other.val4);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val4, 4),
                             _mm_srli_si128(other.val4, 4));
        val4 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));
    }

    inline
    short_vec<int, 16> operator*(const short_vec<int, 16>& other) const
    {
        // see: https://software.intel.com/en-us/forums/intel-c-compiler/topic/288768
        __m128i tmp1 = _mm_mul_epu32(val1, other.val1);
        __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(val1, 4),
                                     _mm_srli_si128(other.val1, 4));
        __m128i result1 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val2, other.val2);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val2, 4),
                             _mm_srli_si128(other.val2, 4));
        __m128i result2 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val3, other.val3);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val3, 4),
                             _mm_srli_si128(other.val3, 4));
        __m128i result3 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        tmp1 = _mm_mul_epu32(val4, other.val4);
        tmp2 = _mm_mul_epu32(_mm_srli_si128(val4, 4),
                             _mm_srli_si128(other.val4, 4));
        __m128i result4 = _mm_unpacklo_epi32(
            _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0,0,2,0)));

        return short_vec<int, 16>(result1, result2, result3, result4);
    }
#endif

    inline
    void operator/=(const short_vec<int, 16>& other)
    {
        val1 = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val1),
                                          _mm_cvtepi32_ps(other.val1)));
        val2 = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val2),
                                          _mm_cvtepi32_ps(other.val2)));
        val3 = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val3),
                                          _mm_cvtepi32_ps(other.val3)));
        val4 = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val4),
                                          _mm_cvtepi32_ps(other.val4)));
    }

    inline
    void operator/=(const sqrt_reference<int, 16>& other);

    inline
    short_vec<int, 16> operator/(const short_vec<int, 16>& other) const
    {
        return short_vec<int, 16>(
            _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val1),
                                       _mm_cvtepi32_ps(other.val1))),
            _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val2),
                                       _mm_cvtepi32_ps(other.val2))),
            _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val3),
                                       _mm_cvtepi32_ps(other.val3))),
            _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(val4),
                                       _mm_cvtepi32_ps(other.val4))));
    }

    inline
    short_vec<int, 16> operator/(const sqrt_reference<int, 16>& other) const;

    inline
    short_vec<int, 16> sqrt() const
    {
        return short_vec<int, 16>(
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val1))),
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val2))),
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val3))),
            _mm_cvtps_epi32(
                _mm_sqrt_ps(_mm_cvtepi32_ps(val4))));
    }

    inline
    void load(const int *data)
    {
        val1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data +  0));
        val2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data +  4));
        val3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data +  8));
        val4 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + 12));
    }

    inline
    void load_aligned(const int *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val1 = _mm_load_si128(reinterpret_cast<const __m128i *>(data +  0));
        val2 = _mm_load_si128(reinterpret_cast<const __m128i *>(data +  4));
        val3 = _mm_load_si128(reinterpret_cast<const __m128i *>(data +  8));
        val4 = _mm_load_si128(reinterpret_cast<const __m128i *>(data + 12));
    }

    inline
    void store(int *data) const
    {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data +  0), val1);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data +  4), val2);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data +  8), val3);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(data + 12), val4);
    }

    inline
    void store_aligned(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_si128(reinterpret_cast<__m128i *>(data +  0), val1);
        _mm_store_si128(reinterpret_cast<__m128i *>(data +  4), val2);
        _mm_store_si128(reinterpret_cast<__m128i *>(data +  8), val3);
        _mm_store_si128(reinterpret_cast<__m128i *>(data + 12), val4);
    }

    inline
    void store_nt(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data +  0), val1);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data +  4), val2);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data +  8), val3);
        _mm_stream_si128(reinterpret_cast<__m128i *>(data + 12), val4);
    }

#ifdef __SSE4_1__
    inline
    void gather(const int *ptr, const unsigned *offsets)
    {
        val1 = _mm_insert_epi32(val1, ptr[offsets[ 0]], 0);
        val1 = _mm_insert_epi32(val1, ptr[offsets[ 1]], 1);
        val1 = _mm_insert_epi32(val1, ptr[offsets[ 2]], 2);
        val1 = _mm_insert_epi32(val1, ptr[offsets[ 3]], 3);

        val2 = _mm_insert_epi32(val2, ptr[offsets[ 4]], 0);
        val2 = _mm_insert_epi32(val2, ptr[offsets[ 5]], 1);
        val2 = _mm_insert_epi32(val2, ptr[offsets[ 6]], 2);
        val2 = _mm_insert_epi32(val2, ptr[offsets[ 7]], 3);

        val3 = _mm_insert_epi32(val3, ptr[offsets[ 8]], 0);
        val3 = _mm_insert_epi32(val3, ptr[offsets[ 9]], 1);
        val3 = _mm_insert_epi32(val3, ptr[offsets[10]], 2);
        val3 = _mm_insert_epi32(val3, ptr[offsets[11]], 3);

        val4 = _mm_insert_epi32(val4, ptr[offsets[12]], 0);
        val4 = _mm_insert_epi32(val4, ptr[offsets[13]], 1);
        val4 = _mm_insert_epi32(val4, ptr[offsets[14]], 2);
        val4 = _mm_insert_epi32(val4, ptr[offsets[15]], 3);
    }

    inline
    void scatter(int *ptr, const unsigned *offsets) const
    {
        ptr[offsets[ 0]] = _mm_extract_epi32(val1, 0);
        ptr[offsets[ 1]] = _mm_extract_epi32(val1, 1);
        ptr[offsets[ 2]] = _mm_extract_epi32(val1, 2);
        ptr[offsets[ 3]] = _mm_extract_epi32(val1, 3);

        ptr[offsets[ 4]] = _mm_extract_epi32(val2, 0);
        ptr[offsets[ 5]] = _mm_extract_epi32(val2, 1);
        ptr[offsets[ 6]] = _mm_extract_epi32(val2, 2);
        ptr[offsets[ 7]] = _mm_extract_epi32(val2, 3);

        ptr[offsets[ 8]] = _mm_extract_epi32(val3, 0);
        ptr[offsets[ 9]] = _mm_extract_epi32(val3, 1);
        ptr[offsets[10]] = _mm_extract_epi32(val3, 2);
        ptr[offsets[11]] = _mm_extract_epi32(val3, 3);

        ptr[offsets[12]] = _mm_extract_epi32(val4, 0);
        ptr[offsets[13]] = _mm_extract_epi32(val4, 1);
        ptr[offsets[14]] = _mm_extract_epi32(val4, 2);
        ptr[offsets[15]] = _mm_extract_epi32(val4, 3);
    }
#else
    inline
    void gather(const int *ptr, const unsigned *offsets)
    {
        __m128i i2, i3, i4;
        val1 = _mm_cvtsi32_si128(ptr[offsets[0]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[1]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[2]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[3]]);
        val1 = _mm_unpacklo_epi32(val1, i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val1 = _mm_unpacklo_epi32(val1, i3);

        val2 = _mm_cvtsi32_si128(ptr[offsets[4]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[5]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[6]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[7]]);
        val2 = _mm_unpacklo_epi32(val2, i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val2 = _mm_unpacklo_epi32(val2, i3);

        val3 = _mm_cvtsi32_si128(ptr[offsets[ 8]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[ 9]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[10]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[11]]);
        val3 = _mm_unpacklo_epi32(val3, i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val3 = _mm_unpacklo_epi32(val3, i3);

        val4 = _mm_cvtsi32_si128(ptr[offsets[12]]);
        i2   = _mm_cvtsi32_si128(ptr[offsets[13]]);
        i3   = _mm_cvtsi32_si128(ptr[offsets[14]]);
        i4   = _mm_cvtsi32_si128(ptr[offsets[15]]);
        val4 = _mm_unpacklo_epi32(val4, i3);
        i3   = _mm_unpacklo_epi32(i2  , i4);
        val4 = _mm_unpacklo_epi32(val4, i3);
    }

    inline
    void scatter(int *ptr, const unsigned *offsets) const
    {
        ptr[offsets[ 0]] = _mm_cvtsi128_si32(val1);
        ptr[offsets[ 1]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val1, _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[ 2]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val1, _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[ 3]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val1, _MM_SHUFFLE(2,1,0,3)));

        ptr[offsets[ 4]] = _mm_cvtsi128_si32(val2);
        ptr[offsets[ 5]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val2, _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[ 6]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val2, _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[ 7]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val2, _MM_SHUFFLE(2,1,0,3)));

        ptr[offsets[ 8]] = _mm_cvtsi128_si32(val3);
        ptr[offsets[ 9]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val3, _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[10]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val3, _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[11]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val3, _MM_SHUFFLE(2,1,0,3)));

        ptr[offsets[12]] = _mm_cvtsi128_si32(val4);
        ptr[offsets[13]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val4, _MM_SHUFFLE(0,3,2,1)));
        ptr[offsets[14]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val4, _MM_SHUFFLE(1,0,3,2)));
        ptr[offsets[15]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(val4, _MM_SHUFFLE(2,1,0,3)));
    }
#endif

private:
    __m128i val1;
    __m128i val2;
    __m128i val3;
    __m128i val4;
};

inline
void operator<<(int *data, const short_vec<int, 16>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<int, 16>
{
public:
    template<typename OTHER_CARGO, int OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<int, 16>& vec) :
        vec(vec)
    {}

private:
    short_vec<int, 16> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<int, 16>::short_vec(const sqrt_reference<int, 16>& other) :
    val1(
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val1)))),
    val2(
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val2)))),
    val3(
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val3)))),
    val4(
        _mm_cvtps_epi32(
            _mm_sqrt_ps(_mm_cvtepi32_ps(other.vec.val4))))
{}

inline
void short_vec<int, 16>::operator/=(const sqrt_reference<int, 16>& other)
{
    val1 = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val1),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val1))));
    val2 = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val2),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val2))));
    val3 = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val2),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val3))));
    val4 = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(val2),
                   _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val4))));
}

inline
short_vec<int, 16> short_vec<int, 16>::operator/(const sqrt_reference<int, 16>& other) const
{
    return short_vec<int, 16>(
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val1),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val1)))),
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val2),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val2)))),
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val3),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val3)))),
        _mm_cvtps_epi32(
            _mm_mul_ps(_mm_cvtepi32_ps(val4),
                       _mm_rsqrt_ps(_mm_cvtepi32_ps(other.vec.val4)))));
}

inline
sqrt_reference<int, 16> sqrt(const short_vec<int, 16>& vec)
{
    return sqrt_reference<int, 16>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<int, 16>& vec)
{
    const int *data1 = reinterpret_cast<const int *>(&vec.val1);
    const int *data2 = reinterpret_cast<const int *>(&vec.val2);
    const int *data3 = reinterpret_cast<const int *>(&vec.val3);
    const int *data4 = reinterpret_cast<const int *>(&vec.val4);
    __os << "["
         << data1[0] << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3] << ", "
         << data2[0] << ", " << data2[1]  << ", " << data2[2]  << ", " << data2[3] << ", "
         << data3[0] << ", " << data3[1]  << ", " << data3[2]  << ", " << data3[3] << ", "
         << data4[0] << ", " << data4[1]  << ", " << data4[2]  << ", " << data4[3]
         << "]";
    return __os;
}

}

#endif
#endif

#endif
