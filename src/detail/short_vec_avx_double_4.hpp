/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_DOUBLE_4_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_DOUBLE_4_HPP

#ifdef __AVX__

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>

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
class short_vec<double, 4>
{
public:
    static const int ARITY = 4;

    typedef short_vec_strategy::avx strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 4>& vec);

    inline
    short_vec(const double data = 0) :
        val1(_mm256_broadcast_sd(&data))
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(const __m256d& val1) :
        val1(val1)
    {}

    inline
    void operator-=(const short_vec<double, 4>& other)
    {
        val1 = _mm256_sub_pd(val1, other.val1);
    }

    inline
    short_vec<double, 4> operator-(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm256_sub_pd(val1, other.val1));
    }

    inline
    void operator+=(const short_vec<double, 4>& other)
    {
        val1 = _mm256_add_pd(val1, other.val1);
    }

    inline
    short_vec<double, 4> operator+(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm256_add_pd(val1, other.val1));
    }

    inline
    void operator*=(const short_vec<double, 4>& other)
    {
        val1 = _mm256_mul_pd(val1, other.val1);
    }

    inline
    short_vec<double, 4> operator*(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm256_mul_pd(val1, other.val1));
    }

    inline
    void operator/=(const short_vec<double, 4>& other)
    {
        val1 = _mm256_div_pd(val1, other.val1);
    }

    inline
    short_vec<double, 4> operator/(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm256_div_pd(val1, other.val1));
    }

    inline
    short_vec<double, 4> sqrt() const
    {
        return short_vec<double, 4>(
            _mm256_sqrt_pd(val1));
    }

    inline
    void load(const double *data)
    {
        val1 = _mm256_loadu_pd(data);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        val1 = _mm256_load_pd(data);
    }

    inline
    void store(double *data) const
    {
        _mm256_storeu_pd(data +  0, val1);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_store_pd(data, val1);
    }

    inline
    void store_nt(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_stream_pd(data, val1);
    }

#ifdef __AVX2__
    inline
    void gather(const double *ptr, const unsigned *offsets)
    {
        __m128i indices;
        indices = _mm_loadu_si128(reinterpret_cast<const __m128i *>(offsets));
        val1    = _mm256_i32gather_pd(ptr, indices, 8);
    }
#else
    inline
    void gather(const double *ptr, const unsigned *offsets)
    {
        __m128d tmp;
        tmp  = _mm_loadl_pd(tmp, ptr + offsets[0]);
        tmp  = _mm_loadh_pd(tmp, ptr + offsets[1]);
        val1 = _mm256_insertf128_pd(val1, tmp, 0);
        tmp  = _mm_loadl_pd(tmp, ptr + offsets[2]);
        tmp  = _mm_loadh_pd(tmp, ptr + offsets[3]);
        val1 = _mm256_insertf128_pd(val1, tmp, 1);
    }
#endif

    inline
    void scatter(double *ptr, const unsigned *offsets) const
    {
        __m128d tmp;
        tmp = _mm256_extractf128_pd(val1, 0);
        _mm_storel_pd(ptr + offsets[0], tmp);
        _mm_storeh_pd(ptr + offsets[1], tmp);
        tmp = _mm256_extractf128_pd(val1, 1);
        _mm_storel_pd(ptr + offsets[2], tmp);
        _mm_storeh_pd(ptr + offsets[3], tmp);
    }

private:
    __m256d val1;
};

inline
void operator<<(double *data, const short_vec<double, 4>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<double, 4> sqrt(const short_vec<double, 4>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 4>& vec)
{
    const double *data1 = reinterpret_cast<const double *>(&vec.val1);

    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << "]";
    return __os;
}

}

#endif
#endif

#endif
