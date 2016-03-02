/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_INT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_INT_32_HPP

#ifdef __AVX2__

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>
#include <iostream>

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

#ifndef __AVX512F__
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
class short_vec<int, 32>
{
public:
    static const int ARITY = 32;

    typedef short_vec_strategy::avx strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 32>& vec);

    inline
    short_vec(const int data = 0) :
        val1(_mm256_set1_epi32(data)),
        val2(_mm256_set1_epi32(data)),
        val3(_mm256_set1_epi32(data)),
        val4(_mm256_set1_epi32(data))
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

    inline
    short_vec(const __m256i& val1, const __m256i& val2,
              const __m256i& val3, const __m256i& val4) :
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
    short_vec(const sqrt_reference<int, 32>& other);

    inline
    void operator-=(const short_vec<int, 32>& other)
    {
        val1 = _mm256_sub_epi32(val1, other.val1);
        val2 = _mm256_sub_epi32(val2, other.val2);
        val3 = _mm256_sub_epi32(val3, other.val3);
        val4 = _mm256_sub_epi32(val4, other.val4);
    }

    inline
    short_vec<int, 32> operator-(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm256_sub_epi32(val1, other.val1),
            _mm256_sub_epi32(val2, other.val2),
            _mm256_sub_epi32(val3, other.val3),
            _mm256_sub_epi32(val4, other.val4));
    }

    inline
    void operator+=(const short_vec<int, 32>& other)
    {
        val1 = _mm256_add_epi32(val1, other.val1);
        val2 = _mm256_add_epi32(val2, other.val2);
        val3 = _mm256_add_epi32(val3, other.val3);
        val4 = _mm256_add_epi32(val4, other.val4);
    }

    inline
    short_vec<int, 32> operator+(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm256_add_epi32(val1, other.val1),
            _mm256_add_epi32(val2, other.val2),
            _mm256_add_epi32(val3, other.val3),
            _mm256_add_epi32(val4, other.val4));
    }

    inline
    void operator*=(const short_vec<int, 32>& other)
    {
        val1 = _mm256_mullo_epi32(val1, other.val1);
        val2 = _mm256_mullo_epi32(val2, other.val2);
        val3 = _mm256_mullo_epi32(val3, other.val3);
        val4 = _mm256_mullo_epi32(val4, other.val4);
    }

    inline
    short_vec<int, 32> operator*(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm256_mullo_epi32(val1, other.val1),
            _mm256_mullo_epi32(val2, other.val2),
            _mm256_mullo_epi32(val3, other.val3),
            _mm256_mullo_epi32(val4, other.val4));
    }

    inline
    void operator/=(const short_vec<int, 32>& other)
    {
        val1 = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(val1),
                                                _mm256_cvtepi32_ps(other.val1)));
        val2 = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(val2),
                                                _mm256_cvtepi32_ps(other.val2)));
        val3 = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(val3),
                                                _mm256_cvtepi32_ps(other.val3)));
        val4 = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(val4),
                                                _mm256_cvtepi32_ps(other.val4)));
    }

    inline
    void operator/=(const sqrt_reference<int, 32>& other);

    inline
    short_vec<int, 32> operator/(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(val1),
                                             _mm256_cvtepi32_ps(other.val1))),
            _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(val2),
                                             _mm256_cvtepi32_ps(other.val2))),
            _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(val3),
                                             _mm256_cvtepi32_ps(other.val3))),
            _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(val4),
                                             _mm256_cvtepi32_ps(other.val4))));
    }

    inline
    short_vec<int, 32> operator/(const sqrt_reference<int, 32>& other) const;

    inline
    short_vec<int, 32> sqrt() const
    {
        return short_vec<int, 32>(
            _mm256_cvtps_epi32(
                _mm256_sqrt_ps(_mm256_cvtepi32_ps(val1))),
            _mm256_cvtps_epi32(
                _mm256_sqrt_ps(_mm256_cvtepi32_ps(val2))),
            _mm256_cvtps_epi32(
                _mm256_sqrt_ps(_mm256_cvtepi32_ps(val3))),
            _mm256_cvtps_epi32(
                _mm256_sqrt_ps(_mm256_cvtepi32_ps(val4))));
    }

    inline
    void load(const int *data)
    {
        val1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data +  0));
        val2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data +  8));
        val3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + 16));
        val4 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + 24));
    }

    inline
    void load_aligned(const int *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        val1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(data +  0));
        val2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(data +  8));
        val3 = _mm256_load_si256(reinterpret_cast<const __m256i *>(data + 16));
        val4 = _mm256_load_si256(reinterpret_cast<const __m256i *>(data + 24));
    }

    inline
    void store(int *data) const
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(data +  0), val1);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(data +  8), val2);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(data + 16), val3);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(data + 24), val4);
    }

    inline
    void store_aligned(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_store_si256(reinterpret_cast<__m256i *>(data +  0), val1);
        _mm256_store_si256(reinterpret_cast<__m256i *>(data +  8), val2);
        _mm256_store_si256(reinterpret_cast<__m256i *>(data + 16), val3);
        _mm256_store_si256(reinterpret_cast<__m256i *>(data + 24), val4);
    }

    inline
    void store_nt(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(data +  0), val1);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(data +  8), val2);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(data + 16), val3);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(data + 24), val4);
    }

    inline
    void gather(const int *ptr, const int *offsets)
    {
        __m256i indices1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets +  0));
        __m256i indices2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets +  8));
        __m256i indices3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets + 16));
        __m256i indices4 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets + 24));
        val1 = _mm256_i32gather_epi32(ptr, indices1, 4);
        val2 = _mm256_i32gather_epi32(ptr, indices2, 4);
        val3 = _mm256_i32gather_epi32(ptr, indices3, 4);
        val4 = _mm256_i32gather_epi32(ptr, indices4, 4);
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[ 0]] = _mm256_extract_epi32(val1, 0);
        ptr[offsets[ 1]] = _mm256_extract_epi32(val1, 1);
        ptr[offsets[ 2]] = _mm256_extract_epi32(val1, 2);
        ptr[offsets[ 3]] = _mm256_extract_epi32(val1, 3);
        ptr[offsets[ 4]] = _mm256_extract_epi32(val1, 4);
        ptr[offsets[ 5]] = _mm256_extract_epi32(val1, 5);
        ptr[offsets[ 6]] = _mm256_extract_epi32(val1, 6);
        ptr[offsets[ 7]] = _mm256_extract_epi32(val1, 7);
        ptr[offsets[ 8]] = _mm256_extract_epi32(val2, 0);
        ptr[offsets[ 9]] = _mm256_extract_epi32(val2, 1);
        ptr[offsets[10]] = _mm256_extract_epi32(val2, 2);
        ptr[offsets[11]] = _mm256_extract_epi32(val2, 3);
        ptr[offsets[12]] = _mm256_extract_epi32(val2, 4);
        ptr[offsets[13]] = _mm256_extract_epi32(val2, 5);
        ptr[offsets[14]] = _mm256_extract_epi32(val2, 6);
        ptr[offsets[15]] = _mm256_extract_epi32(val2, 7);
        ptr[offsets[16]] = _mm256_extract_epi32(val3, 0);
        ptr[offsets[17]] = _mm256_extract_epi32(val3, 1);
        ptr[offsets[18]] = _mm256_extract_epi32(val3, 2);
        ptr[offsets[19]] = _mm256_extract_epi32(val3, 3);
        ptr[offsets[20]] = _mm256_extract_epi32(val3, 4);
        ptr[offsets[21]] = _mm256_extract_epi32(val3, 5);
        ptr[offsets[22]] = _mm256_extract_epi32(val3, 6);
        ptr[offsets[23]] = _mm256_extract_epi32(val3, 7);
        ptr[offsets[24]] = _mm256_extract_epi32(val4, 0);
        ptr[offsets[25]] = _mm256_extract_epi32(val4, 1);
        ptr[offsets[26]] = _mm256_extract_epi32(val4, 2);
        ptr[offsets[27]] = _mm256_extract_epi32(val4, 3);
        ptr[offsets[28]] = _mm256_extract_epi32(val4, 4);
        ptr[offsets[29]] = _mm256_extract_epi32(val4, 5);
        ptr[offsets[30]] = _mm256_extract_epi32(val4, 6);
        ptr[offsets[31]] = _mm256_extract_epi32(val4, 7);
    }

private:
    __m256i val1;
    __m256i val2;
    __m256i val3;
    __m256i val4;
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
    template<typename OTHER_CARGO, int OTHER_ARITY>
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
    val1(
        _mm256_cvtps_epi32(
            _mm256_sqrt_ps(_mm256_cvtepi32_ps(other.vec.val1)))),
    val2(
        _mm256_cvtps_epi32(
            _mm256_sqrt_ps(_mm256_cvtepi32_ps(other.vec.val2)))),
    val3(
        _mm256_cvtps_epi32(
            _mm256_sqrt_ps(_mm256_cvtepi32_ps(other.vec.val3)))),
    val4(
        _mm256_cvtps_epi32(
            _mm256_sqrt_ps(_mm256_cvtepi32_ps(other.vec.val4))))
{}

inline
void short_vec<int, 32>::operator/=(const sqrt_reference<int, 32>& other)
{
    val1 = _mm256_cvtps_epi32(
        _mm256_mul_ps(_mm256_cvtepi32_ps(val1),
                   _mm256_rsqrt_ps(_mm256_cvtepi32_ps(other.vec.val1))));
    val2 = _mm256_cvtps_epi32(
        _mm256_mul_ps(_mm256_cvtepi32_ps(val2),
                   _mm256_rsqrt_ps(_mm256_cvtepi32_ps(other.vec.val2))));
    val3 = _mm256_cvtps_epi32(
        _mm256_mul_ps(_mm256_cvtepi32_ps(val3),
                   _mm256_rsqrt_ps(_mm256_cvtepi32_ps(other.vec.val3))));
    val4 = _mm256_cvtps_epi32(
        _mm256_mul_ps(_mm256_cvtepi32_ps(val4),
                   _mm256_rsqrt_ps(_mm256_cvtepi32_ps(other.vec.val4))));
}

inline
short_vec<int, 32> short_vec<int, 32>::operator/(const sqrt_reference<int, 32>& other) const
{
    return short_vec<int, 32>(
        _mm256_cvtps_epi32(
            _mm256_mul_ps(_mm256_cvtepi32_ps(val1),
                          _mm256_rsqrt_ps(_mm256_cvtepi32_ps(other.vec.val1)))),
        _mm256_cvtps_epi32(
            _mm256_mul_ps(_mm256_cvtepi32_ps(val2),
                          _mm256_rsqrt_ps(_mm256_cvtepi32_ps(other.vec.val2)))),
        _mm256_cvtps_epi32(
            _mm256_mul_ps(_mm256_cvtepi32_ps(val3),
                          _mm256_rsqrt_ps(_mm256_cvtepi32_ps(other.vec.val3)))),
        _mm256_cvtps_epi32(
            _mm256_mul_ps(_mm256_cvtepi32_ps(val4),
                          _mm256_rsqrt_ps(_mm256_cvtepi32_ps(other.vec.val4)))));
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
    const int *data1 = reinterpret_cast<const int *>(&vec.val1);
    const int *data2 = reinterpret_cast<const int *>(&vec.val2);
    const int *data3 = reinterpret_cast<const int *>(&vec.val3);
    const int *data4 = reinterpret_cast<const int *>(&vec.val4);
    __os << "["
         << data1[0] << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3] << ", "
         << data1[4] << ", " << data1[5]  << ", " << data1[6]  << ", " << data1[7] << ", "
         << data2[0] << ", " << data2[1]  << ", " << data2[2]  << ", " << data2[3] << ", "
         << data2[4] << ", " << data2[5]  << ", " << data2[6]  << ", " << data2[7] << ", "
         << data3[0] << ", " << data3[1]  << ", " << data3[2]  << ", " << data3[3] << ", "
         << data3[4] << ", " << data3[5]  << ", " << data3[6]  << ", " << data3[7] << ", "
         << data4[0] << ", " << data4[1]  << ", " << data4[2]  << ", " << data4[3] << ", "
         << data4[4] << ", " << data4[5]  << ", " << data4[6]  << ", " << data4[7]
         << "]";
    return __os;
}

}

#endif
#endif
#endif

#endif
