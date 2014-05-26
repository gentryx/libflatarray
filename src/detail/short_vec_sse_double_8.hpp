/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_8_HPP

#ifdef __SSE__

#include <emmintrin.h>

#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<double, 8>
{
public:
    static const int ARITY = 8;

    inline
    short_vec(const double& data) :
        val1(_mm_set1_pd(data)),
        val2(_mm_set1_pd(data)),
        val3(_mm_set1_pd(data)),
        val4(_mm_set1_pd(data))
    {}

    inline
    short_vec(const double *data) :
        val1(_mm_loadu_pd(data + 0)),
        val2(_mm_loadu_pd(data + 2)),
        val3(_mm_loadu_pd(data + 4)),
        val4(_mm_loadu_pd(data + 6))
    {}

    inline
    short_vec(const __m128d& val1, const __m128d& val2, const __m128d& val3, const __m128d& val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
    {}

    inline
    void operator-=(const short_vec<double, 8>& other)
    {
        val1 = _mm_sub_pd(val1, other.val1);
        val2 = _mm_sub_pd(val2, other.val2);
        val3 = _mm_sub_pd(val3, other.val3);
        val4 = _mm_sub_pd(val4, other.val4);
    }

    inline
    short_vec<double, 8> operator-(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm_sub_pd(val1, other.val1),
            _mm_sub_pd(val2, other.val2),
            _mm_sub_pd(val3, other.val3),
            _mm_sub_pd(val4, other.val4));
    }

    inline
    void operator+=(const short_vec<double, 8>& other)
    {
        val1 = _mm_add_pd(val1, other.val1);
        val2 = _mm_add_pd(val2, other.val2);
        val3 = _mm_add_pd(val3, other.val3);
        val4 = _mm_add_pd(val4, other.val4);
    }

    inline
    short_vec<double, 8> operator+(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm_add_pd(val1, other.val1),
            _mm_add_pd(val2, other.val2),
            _mm_add_pd(val3, other.val3),
            _mm_add_pd(val4, other.val4));
    }

    inline
    void operator*=(const short_vec<double, 8>& other)
    {
        val1 = _mm_mul_pd(val1, other.val1);
        val2 = _mm_mul_pd(val2, other.val2);
        val3 = _mm_mul_pd(val3, other.val3);
        val4 = _mm_mul_pd(val4, other.val4);
    }

    inline
    short_vec<double, 8> operator*(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm_mul_pd(val1, other.val1),
            _mm_mul_pd(val2, other.val2),
            _mm_mul_pd(val3, other.val3),
            _mm_mul_pd(val4, other.val4));
    }

    inline
    void operator/=(const short_vec<double, 8>& other)
    {
        val1 = _mm_div_pd(val1, other.val1);
        val2 = _mm_div_pd(val2, other.val2);
        val3 = _mm_div_pd(val3, other.val3);
        val4 = _mm_div_pd(val4, other.val4);
    }

    inline
    short_vec<double, 8> operator/(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm_div_pd(val1, other.val1),
            _mm_div_pd(val2, other.val2),
            _mm_div_pd(val3, other.val3),
            _mm_div_pd(val4, other.val4));
    }

    inline
    short_vec<double, 8> sqrt() const
    {
        return short_vec<double, 8>(
            _mm_sqrt_pd(val1),
            _mm_sqrt_pd(val2),
            _mm_sqrt_pd(val3),
            _mm_sqrt_pd(val4));
    }

    inline
    void store(double *data) const
    {
        _mm_storeu_pd(data + 0, val1);
        _mm_storeu_pd(data + 2, val2);
        _mm_storeu_pd(data + 4, val3);
        _mm_storeu_pd(data + 6, val4);
    }

private:
    __m128d val1;
    __m128d val2;
    __m128d val3;
    __m128d val4;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(double *data, const short_vec<double, 8>& vec)
{
    vec.store(data);
}

short_vec<double, 8> sqrt(const short_vec<double, 8>& vec)
{
    return vec.sqrt();
}

}

#endif
#endif

#endif
