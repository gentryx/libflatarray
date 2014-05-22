/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_8_HPP

#ifdef __SSE__

#include <emmintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>

#ifndef __AVX__
#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

template<typename CARGO, int ARITY>
class sqrt_reference;

template<>
class short_vec<float, 8>
{
public:
    static const int ARITY = 8;

    inline
    short_vec(const float& data) :
        val1(_mm_set1_ps(data)),
        val2(_mm_set1_ps(data))
    {}

    inline
    short_vec(const float *data) :
        val1(_mm_loadu_ps(data +  0)),
        val2(_mm_loadu_ps(data +  4))
    {}

    inline
    short_vec(const __m128& val1, const __m128& val2) :
        val1(val1),
        val2(val2)
    {}

    inline
    short_vec(const sqrt_reference<float, 8> other);

    inline
    void operator-=(const short_vec<float, 8>& other)
    {
        val1 = _mm_sub_ps(val1, other.val1);
        val2 = _mm_sub_ps(val1, other.val2);
    }

    inline
    short_vec<float, 8> operator-(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm_sub_ps(val1, other.val1),
            _mm_sub_ps(val2, other.val2));
    }

    inline
    void operator+=(const short_vec<float, 8>& other)
    {
        val1 = _mm_add_ps(val1, other.val1);
        val2 = _mm_add_ps(val2, other.val2);
    }

    inline
    short_vec<float, 8> operator+(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm_add_ps(val1, other.val1),
            _mm_add_ps(val2, other.val2));
    }

    inline
    void operator*=(const short_vec<float, 8>& other)
    {
        val1 = _mm_mul_ps(val1, other.val1);
        val2 = _mm_mul_ps(val2, other.val2);
    }

    inline
    short_vec<float, 8> operator*(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm_mul_ps(val1, other.val1),
            _mm_mul_ps(val2, other.val2));
    }

    inline
    void operator/=(const short_vec<float, 8>& other)
    {
        val1 = _mm_div_ps(val1, other.val1);
        val1 = _mm_div_ps(val2, other.val2);
    }

    inline
    void operator/=(const sqrt_reference<float, 8>& other);

    inline
    short_vec<float, 8> operator/(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm_div_ps(val1, other.val1),
            _mm_div_ps(val2, other.val2));
    }

    inline
    short_vec<float, 8> operator/(const sqrt_reference<float, 8>& other) const;

    inline
    short_vec<float, 8> sqrt() const
    {
        return short_vec<float, 8>(
            _mm_sqrt_ps(val1),
            _mm_sqrt_ps(val2));
    }

    inline
    void store(float *data) const
    {
        _mm_storeu_ps(data +  0, val1);
        _mm_storeu_ps(data +  4, val2);
    }

private:
    __m128 val1;
    __m128 val2;
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

inline
short_vec<float, 8>::short_vec(const sqrt_reference<float, 8> other) :
    val1(_mm_sqrt_ps(other.vec.val1)),
    val2(_mm_sqrt_ps(other.vec.val2))
{}

inline
void short_vec<float, 8>::operator/=(const sqrt_reference<float, 8>& other)
{
    val1 = _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1));
    val2 = _mm_mul_ps(val2, _mm_rsqrt_ps(other.vec.val2));
}

inline
short_vec<float, 8> short_vec<float, 8>::operator/(const sqrt_reference<float, 8>& other) const
{
    return short_vec<float, 8>(
        _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1)),
        _mm_mul_ps(val2, _mm_rsqrt_ps(other.vec.val2)));
}

sqrt_reference<float, 8> sqrt(const short_vec<float, 8>& vec)
{
    return sqrt_reference<float, 8>(vec);
}

}

#endif
#endif
#endif

#endif
