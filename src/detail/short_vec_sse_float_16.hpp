/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_16_HPP

#ifdef __SSE__

#include <pmmintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>

#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

template<typename CARGO, int ARITY>
class sqrt_reference;

template<>
class short_vec<float, 16>
{
public:
    static const int ARITY = 16;

    inline
    short_vec(const float& data) :
        val1(_mm_set1_ps(data)),
        val2(_mm_set1_ps(data)),
        val3(_mm_set1_ps(data)),
        val4(_mm_set1_ps(data))
    {}

    inline
    short_vec(const float *data) :
        val1(_mm_loadu_ps(data +  0)),
        val2(_mm_loadu_ps(data +  4)),
        val3(_mm_loadu_ps(data +  8)),
        val4(_mm_loadu_ps(data + 12))
    {}

    inline
    short_vec(const __m128& val1, const __m128& val2, const __m128& val3, const __m128& val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
    {}

    inline
    short_vec(const sqrt_reference<float, 16> other);

    inline
    void operator-=(const short_vec<float, 16>& other)
    {
        val1 = _mm_sub_ps(val1, other.val1);
        val2 = _mm_sub_ps(val1, other.val2);
        val3 = _mm_sub_ps(val1, other.val3);
        val4 = _mm_sub_ps(val1, other.val4);
    }

    inline
    short_vec<float, 16> operator-(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_sub_ps(val1, other.val1),
            _mm_sub_ps(val2, other.val2),
            _mm_sub_ps(val3, other.val3),
            _mm_sub_ps(val4, other.val4));
    }

    inline
    void operator+=(const short_vec<float, 16>& other)
    {
        val1 = _mm_add_ps(val1, other.val1);
        val2 = _mm_add_ps(val2, other.val2);
        val3 = _mm_add_ps(val3, other.val3);
        val4 = _mm_add_ps(val4, other.val4);
    }

    inline
    short_vec<float, 16> operator+(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_add_ps(val1, other.val1),
            _mm_add_ps(val2, other.val2),
            _mm_add_ps(val3, other.val3),
            _mm_add_ps(val4, other.val4));
    }

    inline
    void operator*=(const short_vec<float, 16>& other)
    {
        val1 = _mm_mul_ps(val1, other.val1);
        val2 = _mm_mul_ps(val2, other.val2);
        val3 = _mm_mul_ps(val3, other.val3);
        val4 = _mm_mul_ps(val4, other.val4);
    }

    inline
    short_vec<float, 16> operator*(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_mul_ps(val1, other.val1),
            _mm_mul_ps(val2, other.val2),
            _mm_mul_ps(val3, other.val3),
            _mm_mul_ps(val4, other.val4));
    }

    inline
    void operator/=(const short_vec<float, 16>& other)
    {
        val1 = _mm_div_ps(val1, other.val1);
        val1 = _mm_div_ps(val2, other.val2);
        val1 = _mm_div_ps(val3, other.val3);
        val1 = _mm_div_ps(val4, other.val4);
    }

    inline
    void operator/=(const sqrt_reference<float, 16>& other);

    inline
    short_vec<float, 16> operator/(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_div_ps(val1, other.val1),
            _mm_div_ps(val2, other.val2),
            _mm_div_ps(val3, other.val3),
            _mm_div_ps(val4, other.val4));
    }

    inline
    short_vec<float, 16> operator/(const sqrt_reference<float, 16>& other) const;

    inline
    short_vec<float, 16> sqrt() const
    {
        return short_vec<float, 16>(
            _mm_sqrt_ps(val1),
            _mm_sqrt_ps(val2),
            _mm_sqrt_ps(val3),
            _mm_sqrt_ps(val4));
    }

    inline
    void store(float *data) const
    {
        _mm_storeu_ps(data +  0, val1);
        _mm_storeu_ps(data +  4, val2);
        _mm_storeu_ps(data +  8, val3);
        _mm_storeu_ps(data + 12, val4);
    }

private:
    __m128 val1;
    __m128 val2;
    __m128 val3;
    __m128 val4;
};

inline
void operator<<(float *data, const short_vec<float, 16>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<float, 16>
{
public:
    template<typename OTHER_CARGO, int OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<float, 16>& vec) :
        vec(vec)
    {}

private:
    short_vec<float, 16> vec;
};

inline
short_vec<float, 16>::short_vec(const sqrt_reference<float, 16> other) :
    val1(_mm_sqrt_ps(other.vec.val1)),
    val2(_mm_sqrt_ps(other.vec.val2)),
    val3(_mm_sqrt_ps(other.vec.val3)),
    val4(_mm_sqrt_ps(other.vec.val4))
{}

inline
void short_vec<float, 16>::operator/=(const sqrt_reference<float, 16>& other)
{
    val1 = _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1));
    val2 = _mm_mul_ps(val2, _mm_rsqrt_ps(other.vec.val2));
    val3 = _mm_mul_ps(val3, _mm_rsqrt_ps(other.vec.val3));
    val4 = _mm_mul_ps(val4, _mm_rsqrt_ps(other.vec.val4));
}

inline
short_vec<float, 16> short_vec<float, 16>::operator/(const sqrt_reference<float, 16>& other) const
{
    return short_vec<float, 16>(
        _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1)),
        _mm_mul_ps(val2, _mm_rsqrt_ps(other.vec.val2)),
        _mm_mul_ps(val3, _mm_rsqrt_ps(other.vec.val3)),
        _mm_mul_ps(val4, _mm_rsqrt_ps(other.vec.val4)));
}

sqrt_reference<float, 16> sqrt(const short_vec<float, 16>& vec)
{
    return sqrt_reference<float, 16>(vec);
}

}

#endif
#endif

#endif
