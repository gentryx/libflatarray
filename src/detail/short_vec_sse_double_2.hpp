/**
 * Copyright 2014-2015 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_2_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_2_HPP

#ifdef __SSE__

#include <emmintrin.h>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

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
class short_vec<double, 2>
{
public:
    static const int ARITY = 2;

    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 2>& vec);

    inline
    short_vec(const double data = 0) :
        val1(_mm_set1_pd(data))
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(const __m128d& val1) :
        val1(val1)
    {}

#ifdef LIBFLATARRAY_WITH_CPP14
    inline
    short_vec(const std::initializer_list<double>& il)
    {
        const double *ptr = static_cast<const double *>(&(*il.begin()));
        load(ptr);
    }
#endif

    inline
    void operator-=(const short_vec<double, 2>& other)
    {
        val1 = _mm_sub_pd(val1, other.val1);
    }

    inline
    short_vec<double, 2> operator-(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_sub_pd(val1, other.val1));
    }

    inline
    void operator+=(const short_vec<double, 2>& other)
    {
        val1 = _mm_add_pd(val1, other.val1);
    }

    inline
    short_vec<double, 2> operator+(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_add_pd(val1, other.val1));
    }

    inline
    void operator*=(const short_vec<double, 2>& other)
    {
        val1 = _mm_mul_pd(val1, other.val1);
    }

    inline
    short_vec<double, 2> operator*(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_mul_pd(val1, other.val1));
    }

    inline
    void operator/=(const short_vec<double, 2>& other)
    {
        val1 = _mm_div_pd(val1, other.val1);
    }

    inline
    short_vec<double, 2> operator/(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_div_pd(val1, other.val1));
    }

    inline
    short_vec<double, 2> sqrt() const
    {
        return short_vec<double, 2>(
            _mm_sqrt_pd(val1));
    }

    inline
    void load(const double *data)
    {
        val1 = _mm_loadu_pd(data);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val1 = _mm_load_pd(data);
    }

    inline
    void store(double *data) const
    {
        _mm_storeu_pd(data + 0, val1);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_pd(data + 0, val1);
    }

    inline
    void store_nt(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_pd(data + 0, val1);
    }

    inline
    void gather(const double *ptr, const unsigned *offsets)
    {
        val1 = _mm_loadl_pd(val1, ptr + offsets[0]);
        val1 = _mm_loadh_pd(val1, ptr + offsets[1]);
    }

    inline
    void scatter(double *ptr, const unsigned *offsets) const
    {
        _mm_storel_pd(ptr + offsets[0], val1);
        _mm_storeh_pd(ptr + offsets[1], val1);
    }

private:
    __m128d val1;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(double *data, const short_vec<double, 2>& vec)
{
    vec.store(data);
}

inline
short_vec<double, 2> sqrt(const short_vec<double, 2>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 2>& vec)
{
    const double *data1 = reinterpret_cast<const double *>(&vec.val1);
    __os << "[" << data1[0] << ", " << data1[1]  << "]";
    return __os;
}

}

#endif
#endif

#endif
