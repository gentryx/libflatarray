/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_4_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_4_HPP

#ifdef __SSE__

#include <emmintrin.h>
#include <libflatarray/detail/short_vec_helpers.hpp>

#ifndef __AVX__
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

    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 4>& vec);

    inline
    short_vec(const double data = 0) :
        val1(_mm_set1_pd(data)),
        val2(_mm_set1_pd(data))
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(const __m128d& val1, const __m128d& val2) :
        val1(val1),
        val2(val2)
    {}

    inline
    void operator-=(const short_vec<double, 4>& other)
    {
        val1 = _mm_sub_pd(val1, other.val1);
        val2 = _mm_sub_pd(val2, other.val2);
    }

    inline
    short_vec<double, 4> operator-(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm_sub_pd(val1, other.val1),
            _mm_sub_pd(val2, other.val2));
    }

    inline
    void operator+=(const short_vec<double, 4>& other)
    {
        val1 = _mm_add_pd(val1, other.val1);
        val2 = _mm_add_pd(val2, other.val2);
    }

    inline
    short_vec<double, 4> operator+(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm_add_pd(val1, other.val1),
            _mm_add_pd(val2, other.val2));
    }

    inline
    void operator*=(const short_vec<double, 4>& other)
    {
        val1 = _mm_mul_pd(val1, other.val1);
        val2 = _mm_mul_pd(val2, other.val2);
    }

    inline
    short_vec<double, 4> operator*(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm_mul_pd(val1, other.val1),
            _mm_mul_pd(val2, other.val2));
    }

    inline
    void operator/=(const short_vec<double, 4>& other)
    {
        val1 = _mm_div_pd(val1, other.val1);
        val2 = _mm_div_pd(val2, other.val2);
    }

    inline
    short_vec<double, 4> operator/(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm_div_pd(val1, other.val1),
            _mm_div_pd(val2, other.val2));
    }

    inline
    short_vec<double, 4> sqrt() const
    {
        return short_vec<double, 4>(
            _mm_sqrt_pd(val1),
            _mm_sqrt_pd(val2));
    }

    inline
    void load(const double *data)
    {
        val1 = _mm_loadu_pd(data + 0);
        val2 = _mm_loadu_pd(data + 2);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val1 = _mm_load_pd(data + 0);
        val2 = _mm_load_pd(data + 2);
    }

    inline
    void store(double *data) const
    {
        _mm_storeu_pd(data + 0, val1);
        _mm_storeu_pd(data + 2, val2);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_pd(data + 0, val1);
        _mm_store_pd(data + 2, val2);
    }

    inline
    void store_nt(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_pd(data + 0, val1);
        _mm_stream_pd(data + 2, val2);
    }

    inline
    void gather(const double *ptr, const unsigned *offsets)
    {
        val1 = _mm_loadl_pd(val1, ptr + offsets[0]);
        val1 = _mm_loadh_pd(val1, ptr + offsets[1]);
        val2 = _mm_loadl_pd(val2, ptr + offsets[2]);
        val2 = _mm_loadh_pd(val2, ptr + offsets[3]);
    }

    inline
    void scatter(double *ptr, const unsigned *offsets) const
    {
        _mm_storel_pd(ptr + offsets[0], val1);
        _mm_storeh_pd(ptr + offsets[1], val1);
        _mm_storel_pd(ptr + offsets[2], val2);
        _mm_storeh_pd(ptr + offsets[3], val2);
    }

private:
    __m128d val1;
    __m128d val2;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(double *data, const short_vec<double, 4>& vec)
{
    vec.store(data);
}

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
    const double *data2 = reinterpret_cast<const double *>(&vec.val2);
    __os << "[" << data1[0] << ", " << data1[1]  << ", " << data2[0]  << ", " << data2[1]  << "]";
    return __os;
}

}

#endif
#endif
#endif

#endif
