/**
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_INT_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_INT_8_HPP

#ifdef __AVX2__

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>
#include <iostream>

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
class short_vec<int, 8>
{
public:
    static const int ARITY = 8;

    typedef short_vec_strategy::avx strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 8>& vec);

    inline
    short_vec(const int data = 0) :
        val1(_mm256_set1_epi32(data))
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

    inline
    short_vec(const __m256i& val1) :
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
    short_vec(const sqrt_reference<int, 8>& other);

    inline
    void operator-=(const short_vec<int, 8>& other)
    {
        val1 = _mm256_sub_epi32(val1, other.val1);
    }

    inline
    short_vec<int, 8> operator-(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            _mm256_sub_epi32(val1, other.val1));
    }

    inline
    void operator+=(const short_vec<int, 8>& other)
    {
        val1 = _mm256_add_epi32(val1, other.val1);
    }

    inline
    short_vec<int, 8> operator+(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            _mm256_add_epi32(val1, other.val1));
    }

    inline
    void operator*=(const short_vec<int, 8>& other)
    {
        val1 = _mm256_mullo_epi32(val1, other.val1);
    }

    inline
    short_vec<int, 8> operator*(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            _mm256_mullo_epi32(val1, other.val1));
    }

    inline
    void operator/=(const short_vec<int, 8>& other)
    {
        val1 = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(val1),
                                                _mm256_cvtepi32_ps(other.val1)));
    }

    inline
    void operator/=(const sqrt_reference<int, 8>& other);

    inline
    short_vec<int, 8> operator/(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(val1),
                                             _mm256_cvtepi32_ps(other.val1))));
    }

    inline
    short_vec<int, 8> operator/(const sqrt_reference<int, 8>& other) const;

    inline
    short_vec<int, 8> sqrt() const
    {
        return short_vec<int, 8>(
            _mm256_cvtps_epi32(
                _mm256_sqrt_ps(_mm256_cvtepi32_ps(val1))));
    }

    inline
    void load(const int *data)
    {
        val1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data));
    }

    inline
    void load_aligned(const int *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        val1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(data));
    }

    inline
    void store(int *data) const
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(data), val1);
    }

    inline
    void store_aligned(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_store_si256(reinterpret_cast<__m256i *>(data), val1);
    }

    inline
    void store_nt(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(data), val1);
    }

    inline
    void gather(const int *ptr, const unsigned *offsets)
    {
        __m256i indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets));
        val1 = _mm256_i32gather_epi32(ptr, indices, 4);
    }

    inline
    void scatter(int *ptr, const unsigned *offsets) const
    {
        ptr[offsets[0]] = _mm256_extract_epi32(val1, 0);
        ptr[offsets[1]] = _mm256_extract_epi32(val1, 1);
        ptr[offsets[2]] = _mm256_extract_epi32(val1, 2);
        ptr[offsets[3]] = _mm256_extract_epi32(val1, 3);
        ptr[offsets[4]] = _mm256_extract_epi32(val1, 4);
        ptr[offsets[5]] = _mm256_extract_epi32(val1, 5);
        ptr[offsets[6]] = _mm256_extract_epi32(val1, 6);
        ptr[offsets[7]] = _mm256_extract_epi32(val1, 7);
    }

private:
    __m256i val1;
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
    template<typename OTHER_CARGO, int OTHER_ARITY>
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
    val1(
        _mm256_cvtps_epi32(
            _mm256_sqrt_ps(_mm256_cvtepi32_ps(other.vec.val1))))
{}

inline
void short_vec<int, 8>::operator/=(const sqrt_reference<int, 8>& other)
{
    val1 = _mm256_cvtps_epi32(
        _mm256_mul_ps(_mm256_cvtepi32_ps(val1),
                   _mm256_rsqrt_ps(_mm256_cvtepi32_ps(other.vec.val1))));
}

inline
short_vec<int, 8> short_vec<int, 8>::operator/(const sqrt_reference<int, 8>& other) const
{
    return short_vec<int, 8>(
        _mm256_cvtps_epi32(
            _mm256_mul_ps(_mm256_cvtepi32_ps(val1),
                       _mm256_rsqrt_ps(_mm256_cvtepi32_ps(other.vec.val1)))));
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
    const int *data1 = reinterpret_cast<const int *>(&vec.val1);
    __os << "["
         << data1[0] << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3] << ", "
         << data1[4] << ", " << data1[5]  << ", " << data1[6]  << ", " << data1[7]
         << "]";
    return __os;
}

}

#endif
#endif

#endif
