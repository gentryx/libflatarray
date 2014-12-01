/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_FLOAT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_FLOAT_32_HPP

#ifdef __AVX__

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>

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
class short_vec<float, 32>
{
public:
    static const int ARITY = 32;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 32>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm256_broadcast_ss(&data)),
        val2(_mm256_broadcast_ss(&data)),
        val3(_mm256_broadcast_ss(&data)),
        val4(_mm256_broadcast_ss(&data))
    {}

    inline
    short_vec(const float *data) :
        val1(_mm256_loadu_ps(data + 0)),
        val2(_mm256_loadu_ps(data + 8)),
        val3(_mm256_loadu_ps(data + 16)),
        val4(_mm256_loadu_ps(data + 24))
    {}

    inline
    short_vec(const __m256& val1, const __m256& val2, const __m256& val3, const __m256& val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
    {}

    inline
    short_vec(const sqrt_reference<float, 32> other);

    inline
    void operator-=(const short_vec<float, 32>& other)
    {
        val1 = _mm256_sub_ps(val1, other.val1);
        val2 = _mm256_sub_ps(val2, other.val2);
        val3 = _mm256_sub_ps(val3, other.val3);
        val4 = _mm256_sub_ps(val4, other.val4);
    }

    inline
    short_vec<float, 32> operator-(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm256_sub_ps(val1, other.val1),
            _mm256_sub_ps(val2, other.val2),
            _mm256_sub_ps(val3, other.val3),
            _mm256_sub_ps(val4, other.val4));
    }

    inline
    void operator+=(const short_vec<float, 32>& other)
    {
        val1 = _mm256_add_ps(val1, other.val1);
        val2 = _mm256_add_ps(val2, other.val2);
        val3 = _mm256_add_ps(val3, other.val3);
        val4 = _mm256_add_ps(val4, other.val4);
    }

    inline
    short_vec<float, 32> operator+(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm256_add_ps(val1, other.val1),
            _mm256_add_ps(val2, other.val2),
            _mm256_add_ps(val3, other.val3),
            _mm256_add_ps(val4, other.val4));
    }

    inline
    void operator*=(const short_vec<float, 32>& other)
    {
        val1 = _mm256_mul_ps(val1, other.val1);
        val2 = _mm256_mul_ps(val2, other.val2);
        val3 = _mm256_mul_ps(val3, other.val3);
        val4 = _mm256_mul_ps(val4, other.val4);
    }

    inline
    short_vec<float, 32> operator*(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm256_mul_ps(val1, other.val1),
            _mm256_mul_ps(val2, other.val2),
            _mm256_mul_ps(val3, other.val3),
            _mm256_mul_ps(val4, other.val4));
    }

    inline
    void operator/=(const short_vec<float, 32>& other)
    {
        val1 = _mm256_mul_ps(val1, _mm256_rcp_ps(other.val1));
        val2 = _mm256_mul_ps(val2, _mm256_rcp_ps(other.val2));
        val3 = _mm256_mul_ps(val3, _mm256_rcp_ps(other.val3));
        val4 = _mm256_mul_ps(val4, _mm256_rcp_ps(other.val4));
    }

    inline
    void operator/=(const sqrt_reference<float, 32>& other);

    inline
    short_vec<float, 32> operator/(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm256_mul_ps(val1, _mm256_rcp_ps(other.val1)),
            _mm256_mul_ps(val2, _mm256_rcp_ps(other.val2)),
            _mm256_mul_ps(val3, _mm256_rcp_ps(other.val3)),
            _mm256_mul_ps(val4, _mm256_rcp_ps(other.val4)));
    }

    inline
    short_vec<float, 32> operator/(const sqrt_reference<float, 32>& other) const;

    inline
    short_vec<float, 32> sqrt() const
    {
        return short_vec<float, 32>(
            _mm256_sqrt_ps(val1),
            _mm256_sqrt_ps(val2),
            _mm256_sqrt_ps(val3),
            _mm256_sqrt_ps(val4));
    }

    inline
    void store(float *data) const
    {
        _mm256_storeu_ps(data +  0, val1);
        _mm256_storeu_ps(data +  8, val2);
        _mm256_storeu_ps(data + 16, val3);
        _mm256_storeu_ps(data + 24, val4);
    }

private:
    __m256 val1;
    __m256 val2;
    __m256 val3;
    __m256 val4;
};

inline
void operator<<(float *data, const short_vec<float, 32>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<float, 32>
{
public:
    template<typename OTHER_CARGO, int OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<float, 32>& vec) :
        vec(vec)
    {}

private:
    short_vec<float, 32> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<float, 32>::short_vec(const sqrt_reference<float, 32> other) :
    val1(_mm256_sqrt_ps(other.vec.val1)),
    val2(_mm256_sqrt_ps(other.vec.val2)),
    val3(_mm256_sqrt_ps(other.vec.val3)),
    val4(_mm256_sqrt_ps(other.vec.val4))
{}

inline
void short_vec<float, 32>::operator/=(const sqrt_reference<float, 32>& other)
{
    val1 = _mm256_mul_ps(val1, _mm256_rsqrt_ps(other.vec.val1));
    val2 = _mm256_mul_ps(val2, _mm256_rsqrt_ps(other.vec.val2));
    val3 = _mm256_mul_ps(val3, _mm256_rsqrt_ps(other.vec.val3));
    val4 = _mm256_mul_ps(val4, _mm256_rsqrt_ps(other.vec.val4));
}

inline
short_vec<float, 32> short_vec<float, 32>::operator/(const sqrt_reference<float, 32>& other) const
{
    return short_vec<float, 32>(
        _mm256_mul_ps(val1, _mm256_rsqrt_ps(other.vec.val1)),
        _mm256_mul_ps(val2, _mm256_rsqrt_ps(other.vec.val2)),
        _mm256_mul_ps(val3, _mm256_rsqrt_ps(other.vec.val3)),
        _mm256_mul_ps(val4, _mm256_rsqrt_ps(other.vec.val4)));
}

sqrt_reference<float, 32> sqrt(const short_vec<float, 32>& vec)
{
    return sqrt_reference<float, 32>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 32>& vec)
{
    const float *data1 = reinterpret_cast<const float *>(&vec.val1);
    const float *data2 = reinterpret_cast<const float *>(&vec.val2);
    const float *data3 = reinterpret_cast<const float *>(&vec.val3);
    const float *data4 = reinterpret_cast<const float *>(&vec.val4);

    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << ", " << data1[4] << ", " << data1[5] << ", " << data1[6] << ", " << data1[7]
         << ", " << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
         << ", " << data2[4] << ", " << data2[5] << ", " << data2[6] << ", " << data2[7]
         << ", " << data3[0] << ", " << data3[1] << ", " << data3[2] << ", " << data3[3]
         << ", " << data3[4] << ", " << data3[5] << ", " << data3[6] << ", " << data3[7]
         << ", " << data4[0] << ", " << data4[1] << ", " << data4[2] << ", " << data4[3]
         << ", " << data4[4] << ", " << data4[5] << ", " << data4[6] << ", " << data4[7]
         << "]";
    return __os;
}

}

#endif
#endif

#endif
