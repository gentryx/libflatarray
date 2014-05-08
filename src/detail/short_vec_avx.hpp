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
    static const int Arity = 8;

    inline
    short_vec(const float& data) :
        val1(_mm256_broadcast_ss(&data))
    {}

    inline
    short_vec(const float *data) :
        val1(_mm256_loadu_ps(data))
    {}

    inline
    short_vec(const sqrt_reference<float, 8> other);

    inline
    short_vec(const __m256& val1) :
        val1(val1)
    {}

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

template<typename CARGO, int ARITY>
class sqrt_reference
{
public:
    template<typename OTHER_CARGO, int OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<CARGO, ARITY>& vec) :
        vec(vec)
    {}

private:
    short_vec<CARGO, ARITY> vec;
};

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

}

#endif
#endif

#endif
