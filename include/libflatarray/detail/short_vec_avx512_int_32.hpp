/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_INT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_INT_32_HPP

#if LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX512F

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>
#include <libflatarray/short_vec_base.hpp>
#include <iostream>

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

    typedef short_vec_strategy::avx512f strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 32>& vec);

    inline
    short_vec(const int data = 0) :
        val{_mm512_set1_epi32(data),
            _mm512_set1_epi32(data)}
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

    inline
    short_vec(const __m512i& val1, const __m512i& val2) :
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
    short_vec(const sqrt_reference<int, 32>& other);

    inline
    void operator-=(const short_vec<int, 32>& other)
    {
        val[ 0] = _mm512_sub_epi32(val[ 0], other.val[ 0]);
        val[ 1] = _mm512_sub_epi32(val[ 1], other.val[ 1]);
    }

    inline
    short_vec<int, 32> operator-(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm512_sub_epi32(val[ 0], other.val[ 0]),
            _mm512_sub_epi32(val[ 1], other.val[ 1]));
    }

    inline
    void operator+=(const short_vec<int, 32>& other)
    {
        val[ 0] = _mm512_add_epi32(val[ 0], other.val[ 0]);
        val[ 1] = _mm512_add_epi32(val[ 1], other.val[ 1]);
    }

    inline
    short_vec<int, 32> operator+(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm512_add_epi32(val[ 0], other.val[ 0]),
            _mm512_add_epi32(val[ 1], other.val[ 1]));
    }

    inline
    void operator*=(const short_vec<int, 32>& other)
    {
        val[ 0] = _mm512_mullo_epi32(val[ 0], other.val[ 0]);
        val[ 1] = _mm512_mullo_epi32(val[ 1], other.val[ 1]);
    }

    inline
    short_vec<int, 32> operator*(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm512_mullo_epi32(val[ 0], other.val[ 0]),
            _mm512_mullo_epi32(val[ 1], other.val[ 1]));
    }

    inline
    void operator/=(const short_vec<int, 32>& other)
    {
        val[ 0] = _mm512_cvtps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(val[ 0]),
                                                _mm512_cvtepi32_ps(other.val[ 0])));
        val[ 1] = _mm512_cvtps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(val[ 1]),
                                                _mm512_cvtepi32_ps(other.val[ 1])));
    }

    inline
    void operator/=(const sqrt_reference<int, 32>& other);

    inline
    short_vec<int, 32> operator/(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            _mm512_cvttps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(val[ 0]),
                                              _mm512_cvtepi32_ps(other.val[ 0]))),
            _mm512_cvttps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(val[ 1]),
                                              _mm512_cvtepi32_ps(other.val[ 1]))));
    }

    inline
    short_vec<int, 32> operator/(const sqrt_reference<int, 32>& other) const;

    inline
    short_vec<int, 32> sqrt() const
    {
        return short_vec<int, 32>(
            _mm512_cvtps_epi32(
                _mm512_sqrt_ps(_mm512_cvtepi32_ps(val[ 0]))),
            _mm512_cvtps_epi32(
                _mm512_sqrt_ps(_mm512_cvtepi32_ps(val[ 1]))));
    }

    inline
    void load(const int *data)
    {
        val[ 0] = _mm512_loadu_si512(data +  0);
        val[ 1] = _mm512_loadu_si512(data + 16);
    }

    inline
    void load_aligned(const int *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        val[ 0] = _mm512_load_epi32(data +  0);
        val[ 1] = _mm512_load_epi32(data + 16);
    }

    inline
    void store(int *data) const
    {
        _mm512_storeu_si512(data +  0, val[ 0]);
        _mm512_storeu_si512(data + 16, val[ 1]);
    }

    inline
    void store_aligned(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_store_epi32(data +  0, val[ 0]);
        _mm512_store_epi32(data + 16, val[ 1]);
    }

    inline
    void store_nt(int *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_stream_si512(reinterpret_cast<__m512i *>(data +  0), val[ 0]);
        _mm512_stream_si512(reinterpret_cast<__m512i *>(data + 16), val[ 1]);
    }

    inline
    void gather(const int *ptr, const int *offsets)
    {
        __m512i indices1 = _mm512_loadu_si512(offsets +  0);
        __m512i indices2 = _mm512_loadu_si512(offsets + 16);
        val[ 0] = _mm512_i32gather_epi32(indices1, ptr, 4);
        val[ 1] = _mm512_i32gather_epi32(indices2, ptr, 4);
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        __m512i indices1 = _mm512_loadu_si512(offsets +  0);
        __m512i indices2 = _mm512_loadu_si512(offsets + 16);
        _mm512_i32scatter_epi32(ptr, indices1, val[ 0], 4);
        _mm512_i32scatter_epi32(ptr, indices2, val[ 1], 4);
    }

private:
    __m512i val[2];
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
        _mm512_cvtps_epi32(
            _mm512_sqrt_ps(_mm512_cvtepi32_ps(other.vec.val[ 0]))),
        _mm512_cvtps_epi32(
            _mm512_sqrt_ps(_mm512_cvtepi32_ps(other.vec.val[ 1])))
            }
{}

inline
void short_vec<int, 32>::operator/=(const sqrt_reference<int, 32>& other)
{
    val[ 0] = _mm512_cvtps_epi32(
        _mm512_mul_ps(_mm512_cvtepi32_ps(val[ 0]),
                      _mm512_rsqrt14_ps(_mm512_cvtepi32_ps(other.vec.val[ 0]))));
    val[ 1] = _mm512_cvtps_epi32(
        _mm512_mul_ps(_mm512_cvtepi32_ps(val[ 1]),
                      _mm512_rsqrt14_ps(_mm512_cvtepi32_ps(other.vec.val[ 1]))));
}

inline
short_vec<int, 32> short_vec<int, 32>::operator/(const sqrt_reference<int, 32>& other) const
{
    return short_vec<int, 32>(
        _mm512_cvtps_epi32(
            _mm512_mul_ps(_mm512_cvtepi32_ps(val[ 0]),
                          _mm512_rsqrt14_ps(_mm512_cvtepi32_ps(other.vec.val[ 0])))),
        _mm512_cvtps_epi32(
            _mm512_mul_ps(_mm512_cvtepi32_ps(val[ 1]),
                          _mm512_rsqrt14_ps(_mm512_cvtepi32_ps(other.vec.val[ 1])))));
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
    __os << "["
         << data1[ 0] << ", " << data1[ 1]  << ", " << data1[ 2]  << ", " << data1[ 3] << ", "
         << data1[ 4] << ", " << data1[ 5]  << ", " << data1[ 6]  << ", " << data1[ 7] << ", "
         << data1[ 8] << ", " << data1[ 9]  << ", " << data1[10]  << ", " << data1[11] << ", "
         << data1[12] << ", " << data1[13]  << ", " << data1[14]  << ", " << data1[15] << ", "
         << data2[ 0] << ", " << data2[ 1]  << ", " << data2[ 2]  << ", " << data2[ 3] << ", "
         << data2[ 4] << ", " << data2[ 5]  << ", " << data2[ 6]  << ", " << data2[ 7] << ", "
         << data2[ 8] << ", " << data2[ 9]  << ", " << data2[10]  << ", " << data2[11] << ", "
         << data2[12] << ", " << data2[13]  << ", " << data2[14]  << ", " << data2[15]
         << "]";
    return __os;
}

}

#endif

#endif
