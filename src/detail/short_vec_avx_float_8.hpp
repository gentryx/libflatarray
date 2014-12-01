/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_HPP

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
class short_vec<float, 8>
{
public:
    static const int ARITY = 8;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 8>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm256_broadcast_ss(&data))
    {}

    inline
    short_vec(const float *data) :
        val1(_mm256_loadu_ps(data))
    {}

    inline
    short_vec(const __m256& val1) :
        val1(val1)
    {}

    inline
    short_vec(const sqrt_reference<float, 8> other);

    inline
    void operator-=(const short_vec<float, 8>& other)
    {
        val1 = _mm256_sub_ps(val1, other.val1);
    }

    inline
    short_vec<float, 8> operator-(const short_vec<float, 8>& other) const
    {
        return _mm256_sub_ps(val1, other.val1);
    }

    inline
    void operator+=(const short_vec<float, 8>& other)
    {
        val1 = _mm256_add_ps(val1, other.val1);
    }

    inline
    short_vec<float, 8> operator+(const short_vec<float, 8>& other) const
    {
        return _mm256_add_ps(val1, other.val1);
    }

    inline
    void operator*=(const short_vec<float, 8>& other)
    {
        val1 = _mm256_mul_ps(val1, other.val1);
    }

    inline
    short_vec<float, 8> operator*(const short_vec<float, 8>& other) const
    {
        return _mm256_mul_ps(val1, other.val1);
    }

    inline
    void operator/=(const short_vec<float, 8>& other)
    {
        val1 = _mm256_mul_ps(val1, _mm256_rcp_ps(other.val1));
    }

    inline
    void operator/=(const sqrt_reference<float, 8>& other);

    inline
    short_vec<float, 8> operator/(const short_vec<float, 8>& other) const
    {
        return _mm256_mul_ps(val1, _mm256_rcp_ps(other.val1));
    }

    inline
    short_vec<float, 8> operator/(const sqrt_reference<float, 8>& other) const;

    inline
    short_vec<float, 8> sqrt() const
    {
        return _mm256_sqrt_ps(val1);
    }

    inline
    void store(float *data) const
    {
        _mm256_storeu_ps(data, val1);
    }

private:
    __m256 val1;
};

inline
void operator<<(float *data, const short_vec<float, 8>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<float, 8>
{
public:
    template<typename OTHER_CARGO, int OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<float, 8>& vec) :
        vec(vec)
    {}

private:
    short_vec<float, 8> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<float, 8>::short_vec(const sqrt_reference<float, 8> other) :
    val1(_mm256_sqrt_ps(other.vec.val1))
{}

inline
void short_vec<float, 8>::operator/=(const sqrt_reference<float, 8>& other)
{
    val1 = _mm256_mul_ps(val1, _mm256_rsqrt_ps(other.vec.val1));
}

inline
short_vec<float, 8> short_vec<float, 8>::operator/(const sqrt_reference<float, 8>& other) const
{
    return _mm256_mul_ps(val1, _mm256_rsqrt_ps(other.vec.val1));
}

sqrt_reference<float, 8> sqrt(const short_vec<float, 8>& vec)
{
    return sqrt_reference<float, 8>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 8>& vec)
{
    const float *data1 = reinterpret_cast<const float *>(&vec.val1);
    __os << "[" << data1[0] << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3]  << ", " << data1[4]  << ", " << data1[5]  << ", " << data1[6]  << ", " << data1[7] << "]";
    return __os;
}

}

#endif
#endif

#endif
