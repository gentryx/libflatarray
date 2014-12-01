/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_DOUBLE_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_DOUBLE_16_HPP

#ifdef __AVX__

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>

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
class short_vec<double, 16>
{
public:
    static const int ARITY = 16;

    typedef short_vec_strategy::avx strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 16>& vec);

    inline
    short_vec(const double data = 0) :
        val1(_mm256_broadcast_sd(&data)),
        val2(_mm256_broadcast_sd(&data)),
        val3(_mm256_broadcast_sd(&data)),
        val4(_mm256_broadcast_sd(&data))
    {}

    inline
    short_vec(const double *data) :
        val1(_mm256_loadu_pd(data + 0)),
        val2(_mm256_loadu_pd(data + 4)),
        val3(_mm256_loadu_pd(data + 8)),
        val4(_mm256_loadu_pd(data + 12))
    {}

    inline
    short_vec(const __m256d& val1, const __m256d& val2, const __m256d& val3, const __m256d& val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
    {}

    inline
    void operator-=(const short_vec<double, 16>& other)
    {
        val1 = _mm256_sub_pd(val1, other.val1);
        val2 = _mm256_sub_pd(val2, other.val2);
        val3 = _mm256_sub_pd(val3, other.val3);
        val4 = _mm256_sub_pd(val4, other.val4);
    }

    inline
    short_vec<double, 16> operator-(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm256_sub_pd(val1, other.val1),
            _mm256_sub_pd(val2, other.val2),
            _mm256_sub_pd(val3, other.val3),
            _mm256_sub_pd(val4, other.val4));
    }

    inline
    void operator+=(const short_vec<double, 16>& other)
    {
        val1 = _mm256_add_pd(val1, other.val1);
        val2 = _mm256_add_pd(val2, other.val2);
        val3 = _mm256_add_pd(val3, other.val3);
        val4 = _mm256_add_pd(val4, other.val4);
    }

    inline
    short_vec<double, 16> operator+(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm256_add_pd(val1, other.val1),
            _mm256_add_pd(val2, other.val2),
            _mm256_add_pd(val3, other.val3),
            _mm256_add_pd(val4, other.val4));
    }

    inline
    void operator*=(const short_vec<double, 16>& other)
    {
        val1 = _mm256_mul_pd(val1, other.val1);
        val2 = _mm256_mul_pd(val2, other.val2);
        val3 = _mm256_mul_pd(val3, other.val3);
        val4 = _mm256_mul_pd(val4, other.val4);
    }

    inline
    short_vec<double, 16> operator*(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm256_mul_pd(val1, other.val1),
            _mm256_mul_pd(val2, other.val2),
            _mm256_mul_pd(val3, other.val3),
            _mm256_mul_pd(val4, other.val4));
    }

    inline
    void operator/=(const short_vec<double, 16>& other)
    {
        val1 = _mm256_div_pd(val1, other.val1);
        val2 = _mm256_div_pd(val2, other.val2);
        val3 = _mm256_div_pd(val3, other.val3);
        val4 = _mm256_div_pd(val4, other.val4);
    }

    inline
    short_vec<double, 16> operator/(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm256_div_pd(val1, other.val1),
            _mm256_div_pd(val2, other.val2),
            _mm256_div_pd(val3, other.val3),
            _mm256_div_pd(val4, other.val4));
    }

    inline
    short_vec<double, 16> sqrt() const
    {
        return short_vec<double, 16>(
            _mm256_sqrt_pd(val1),
            _mm256_sqrt_pd(val2),
            _mm256_sqrt_pd(val3),
            _mm256_sqrt_pd(val4));
    }

    inline
    void store(double *data) const
    {
        _mm256_storeu_pd(data +  0, val1);
        _mm256_storeu_pd(data +  4, val2);
        _mm256_storeu_pd(data +  8, val3);
        _mm256_storeu_pd(data + 12, val4);
    }

private:
    __m256d val1;
    __m256d val2;
    __m256d val3;
    __m256d val4;
};

inline
void operator<<(double *data, const short_vec<double, 16>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

short_vec<double, 16> sqrt(const short_vec<double, 16>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 16>& vec)
{
    const double *data1 = reinterpret_cast<const double *>(&vec.val1);
    const double *data2 = reinterpret_cast<const double *>(&vec.val2);
    const double *data3 = reinterpret_cast<const double *>(&vec.val3);
    const double *data4 = reinterpret_cast<const double *>(&vec.val4);

    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << ", " << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
         << ", " << data3[0] << ", " << data3[1] << ", " << data3[2] << ", " << data3[3]
         << ", " << data4[0] << ", " << data4[1] << ", " << data4[2] << ", " << data4[3]
         << "]";
    return __os;
}

}

#endif
#endif

#endif
