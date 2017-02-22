/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_32_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE2) ||            \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE4_1)

#include <emmintrin.h>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>
#include <libflatarray/short_vec_base.hpp>

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

namespace LibFlatArray {

template<typename CARGO, std::size_t ARITY>
class short_vec;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<double, 32> : public short_vec_base<double, 32>
{
public:
    static const std::size_t ARITY = 32;
    typedef short_vec<double, 32> mask_type;
    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 32>& vec);

    inline
    short_vec(const double data = 0) :
        val{_mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data),
            _mm_set1_pd(data)}
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(
        const __m128d& val1,
        const __m128d& val2,
        const __m128d& val3,
        const __m128d& val4,
        const __m128d& val5,
        const __m128d& val6,
        const __m128d& val7,
        const __m128d& val8,
        const __m128d& val9,
        const __m128d& val10,
        const __m128d& val11,
        const __m128d& val12,
        const __m128d& val13,
        const __m128d& val14,
        const __m128d& val15,
        const __m128d& val16) :
        val{val1,
            val2,
            val3,
            val4,
            val5,
            val6,
            val7,
            val8,
            val9,
            val10,
            val11,
            val12,
            val13,
            val14,
            val15,
            val16}
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
    bool any() const
    {
        __m128d buf0 = _mm_or_pd(
            _mm_or_pd(
                _mm_or_pd(
                    _mm_or_pd(val[ 0], val[ 1]),
                    _mm_or_pd(val[ 2], val[ 3])),
                _mm_or_pd(
                    _mm_or_pd(val[ 4], val[ 5]),
                    _mm_or_pd(val[ 6], val[ 7]))),
            _mm_or_pd(
                _mm_or_pd(
                    _mm_or_pd(val[ 8], val[ 9]),
                    _mm_or_pd(val[10], val[11])),
                _mm_or_pd(
                    _mm_or_pd(val[12], val[13]),
                    _mm_or_pd(val[14], val[15]))));

#ifdef __SSE4_1__
        return (0 == _mm_testz_si128(
                    _mm_castpd_si128(buf0),
                    _mm_castpd_si128(buf0)));
#else
        __m128d buf1 = _mm_shuffle_pd(buf0, buf0, 1);
        return _mm_cvtsd_f64(buf0) || _mm_cvtsd_f64(buf1);
#endif
    }

    inline
    double operator[](int i) const
    {
        __m128d buf = val[i >> 1];
        i &= 1;

        if (i == 0) {
            return _mm_cvtsd_f64(buf);
        }

        buf = _mm_shuffle_pd(buf, buf, 1);
        return _mm_cvtsd_f64(buf);
    }

    inline
    void operator-=(const short_vec<double, 32>& other)
    {
        val[ 0] = _mm_sub_pd(val[ 0], other.val[ 0]);
        val[ 1] = _mm_sub_pd(val[ 1], other.val[ 1]);
        val[ 2] = _mm_sub_pd(val[ 2], other.val[ 2]);
        val[ 3] = _mm_sub_pd(val[ 3], other.val[ 3]);
        val[ 4] = _mm_sub_pd(val[ 4], other.val[ 4]);
        val[ 5] = _mm_sub_pd(val[ 5], other.val[ 5]);
        val[ 6] = _mm_sub_pd(val[ 6], other.val[ 6]);
        val[ 7] = _mm_sub_pd(val[ 7], other.val[ 7]);
        val[ 8] = _mm_sub_pd(val[ 8], other.val[ 8]);
        val[ 9] = _mm_sub_pd(val[ 9], other.val[ 9]);
        val[10] = _mm_sub_pd(val[10], other.val[10]);
        val[11] = _mm_sub_pd(val[11], other.val[11]);
        val[12] = _mm_sub_pd(val[12], other.val[12]);
        val[13] = _mm_sub_pd(val[13], other.val[13]);
        val[14] = _mm_sub_pd(val[14], other.val[14]);
        val[15] = _mm_sub_pd(val[15], other.val[15]);
    }

    inline
    short_vec<double, 32> operator-(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_sub_pd(val[ 0], other.val[ 0]),
            _mm_sub_pd(val[ 1], other.val[ 1]),
            _mm_sub_pd(val[ 2], other.val[ 2]),
            _mm_sub_pd(val[ 3], other.val[ 3]),
            _mm_sub_pd(val[ 4], other.val[ 4]),
            _mm_sub_pd(val[ 5], other.val[ 5]),
            _mm_sub_pd(val[ 6], other.val[ 6]),
            _mm_sub_pd(val[ 7], other.val[ 7]),
            _mm_sub_pd(val[ 8], other.val[ 8]),
            _mm_sub_pd(val[ 9], other.val[ 9]),
            _mm_sub_pd(val[10], other.val[10]),
            _mm_sub_pd(val[11], other.val[11]),
            _mm_sub_pd(val[12], other.val[12]),
            _mm_sub_pd(val[13], other.val[13]),
            _mm_sub_pd(val[14], other.val[14]),
            _mm_sub_pd(val[15], other.val[15]));
    }

    inline
    void operator+=(const short_vec<double, 32>& other)
    {
        val[ 0] = _mm_add_pd(val[ 0], other.val[ 0]);
        val[ 1] = _mm_add_pd(val[ 1], other.val[ 1]);
        val[ 2] = _mm_add_pd(val[ 2], other.val[ 2]);
        val[ 3] = _mm_add_pd(val[ 3], other.val[ 3]);
        val[ 4] = _mm_add_pd(val[ 4], other.val[ 4]);
        val[ 5] = _mm_add_pd(val[ 5], other.val[ 5]);
        val[ 6] = _mm_add_pd(val[ 6], other.val[ 6]);
        val[ 7] = _mm_add_pd(val[ 7], other.val[ 7]);
        val[ 8] = _mm_add_pd(val[ 8], other.val[ 8]);
        val[ 9] = _mm_add_pd(val[ 9], other.val[ 9]);
        val[10] = _mm_add_pd(val[10], other.val[10]);
        val[11] = _mm_add_pd(val[11], other.val[11]);
        val[12] = _mm_add_pd(val[12], other.val[12]);
        val[13] = _mm_add_pd(val[13], other.val[13]);
        val[14] = _mm_add_pd(val[14], other.val[14]);
        val[15] = _mm_add_pd(val[15], other.val[15]);
    }

    inline
    short_vec<double, 32> operator+(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_add_pd(val[ 0], other.val[ 0]),
            _mm_add_pd(val[ 1], other.val[ 1]),
            _mm_add_pd(val[ 2], other.val[ 2]),
            _mm_add_pd(val[ 3], other.val[ 3]),
            _mm_add_pd(val[ 4], other.val[ 4]),
            _mm_add_pd(val[ 5], other.val[ 5]),
            _mm_add_pd(val[ 6], other.val[ 6]),
            _mm_add_pd(val[ 7], other.val[ 7]),
            _mm_add_pd(val[ 8], other.val[ 8]),
            _mm_add_pd(val[ 9], other.val[ 9]),
            _mm_add_pd(val[10], other.val[10]),
            _mm_add_pd(val[11], other.val[11]),
            _mm_add_pd(val[12], other.val[12]),
            _mm_add_pd(val[13], other.val[13]),
            _mm_add_pd(val[14], other.val[14]),
            _mm_add_pd(val[15], other.val[15]));
    }

    inline
    void operator*=(const short_vec<double, 32>& other)
    {
        val[ 0] = _mm_mul_pd(val[ 0], other.val[ 0]);
        val[ 1] = _mm_mul_pd(val[ 1], other.val[ 1]);
        val[ 2] = _mm_mul_pd(val[ 2], other.val[ 2]);
        val[ 3] = _mm_mul_pd(val[ 3], other.val[ 3]);
        val[ 4] = _mm_mul_pd(val[ 4], other.val[ 4]);
        val[ 5] = _mm_mul_pd(val[ 5], other.val[ 5]);
        val[ 6] = _mm_mul_pd(val[ 6], other.val[ 6]);
        val[ 7] = _mm_mul_pd(val[ 7], other.val[ 7]);
        val[ 8] = _mm_mul_pd(val[ 8], other.val[ 8]);
        val[ 9] = _mm_mul_pd(val[ 9], other.val[ 9]);
        val[10] = _mm_mul_pd(val[10], other.val[10]);
        val[11] = _mm_mul_pd(val[11], other.val[11]);
        val[12] = _mm_mul_pd(val[12], other.val[12]);
        val[13] = _mm_mul_pd(val[13], other.val[13]);
        val[14] = _mm_mul_pd(val[14], other.val[14]);
        val[15] = _mm_mul_pd(val[15], other.val[15]);
    }

    inline
    short_vec<double, 32> operator*(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_mul_pd(val[ 0], other.val[ 0]),
            _mm_mul_pd(val[ 1], other.val[ 1]),
            _mm_mul_pd(val[ 2], other.val[ 2]),
            _mm_mul_pd(val[ 3], other.val[ 3]),
            _mm_mul_pd(val[ 4], other.val[ 4]),
            _mm_mul_pd(val[ 5], other.val[ 5]),
            _mm_mul_pd(val[ 6], other.val[ 6]),
            _mm_mul_pd(val[ 7], other.val[ 7]),
            _mm_mul_pd(val[ 8], other.val[ 8]),
            _mm_mul_pd(val[ 9], other.val[ 9]),
            _mm_mul_pd(val[10], other.val[10]),
            _mm_mul_pd(val[11], other.val[11]),
            _mm_mul_pd(val[12], other.val[12]),
            _mm_mul_pd(val[13], other.val[13]),
            _mm_mul_pd(val[14], other.val[14]),
            _mm_mul_pd(val[15], other.val[15]));
    }

    inline
    void operator/=(const short_vec<double, 32>& other)
    {
        val[ 0] = _mm_div_pd(val[ 0], other.val[ 0]);
        val[ 1] = _mm_div_pd(val[ 1], other.val[ 1]);
        val[ 2] = _mm_div_pd(val[ 2], other.val[ 2]);
        val[ 3] = _mm_div_pd(val[ 3], other.val[ 3]);
        val[ 4] = _mm_div_pd(val[ 4], other.val[ 4]);
        val[ 5] = _mm_div_pd(val[ 5], other.val[ 5]);
        val[ 6] = _mm_div_pd(val[ 6], other.val[ 6]);
        val[ 7] = _mm_div_pd(val[ 7], other.val[ 7]);
        val[ 8] = _mm_div_pd(val[ 8], other.val[ 8]);
        val[ 9] = _mm_div_pd(val[ 9], other.val[ 9]);
        val[10] = _mm_div_pd(val[10], other.val[10]);
        val[11] = _mm_div_pd(val[11], other.val[11]);
        val[12] = _mm_div_pd(val[12], other.val[12]);
        val[13] = _mm_div_pd(val[13], other.val[13]);
        val[14] = _mm_div_pd(val[14], other.val[14]);
        val[15] = _mm_div_pd(val[15], other.val[15]);
    }

    inline
    short_vec<double, 32> operator/(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_div_pd(val[ 0], other.val[ 0]),
            _mm_div_pd(val[ 1], other.val[ 1]),
            _mm_div_pd(val[ 2], other.val[ 2]),
            _mm_div_pd(val[ 3], other.val[ 3]),
            _mm_div_pd(val[ 4], other.val[ 4]),
            _mm_div_pd(val[ 5], other.val[ 5]),
            _mm_div_pd(val[ 6], other.val[ 6]),
            _mm_div_pd(val[ 7], other.val[ 7]),
            _mm_div_pd(val[ 8], other.val[ 8]),
            _mm_div_pd(val[ 9], other.val[ 9]),
            _mm_div_pd(val[10], other.val[10]),
            _mm_div_pd(val[11], other.val[11]),
            _mm_div_pd(val[12], other.val[12]),
            _mm_div_pd(val[13], other.val[13]),
            _mm_div_pd(val[14], other.val[14]),
            _mm_div_pd(val[15], other.val[15]));
    }

    inline
    short_vec<double, 32> operator/(const sqrt_reference<double, 32>& other) const;

    inline
    short_vec<double, 32> operator<(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_cmplt_pd(val[ 0], other.val[ 0]),
            _mm_cmplt_pd(val[ 1], other.val[ 1]),
            _mm_cmplt_pd(val[ 2], other.val[ 2]),
            _mm_cmplt_pd(val[ 3], other.val[ 3]),
            _mm_cmplt_pd(val[ 4], other.val[ 4]),
            _mm_cmplt_pd(val[ 5], other.val[ 5]),
            _mm_cmplt_pd(val[ 6], other.val[ 6]),
            _mm_cmplt_pd(val[ 7], other.val[ 7]),
            _mm_cmplt_pd(val[ 8], other.val[ 8]),
            _mm_cmplt_pd(val[ 9], other.val[ 9]),
            _mm_cmplt_pd(val[10], other.val[10]),
            _mm_cmplt_pd(val[11], other.val[11]),
            _mm_cmplt_pd(val[12], other.val[12]),
            _mm_cmplt_pd(val[13], other.val[13]),
            _mm_cmplt_pd(val[14], other.val[14]),
            _mm_cmplt_pd(val[15], other.val[15]));
    }

    inline
    short_vec<double, 32> operator<=(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_cmple_pd(val[ 0], other.val[ 0]),
            _mm_cmple_pd(val[ 1], other.val[ 1]),
            _mm_cmple_pd(val[ 2], other.val[ 2]),
            _mm_cmple_pd(val[ 3], other.val[ 3]),
            _mm_cmple_pd(val[ 4], other.val[ 4]),
            _mm_cmple_pd(val[ 5], other.val[ 5]),
            _mm_cmple_pd(val[ 6], other.val[ 6]),
            _mm_cmple_pd(val[ 7], other.val[ 7]),
            _mm_cmple_pd(val[ 8], other.val[ 8]),
            _mm_cmple_pd(val[ 9], other.val[ 9]),
            _mm_cmple_pd(val[10], other.val[10]),
            _mm_cmple_pd(val[11], other.val[11]),
            _mm_cmple_pd(val[12], other.val[12]),
            _mm_cmple_pd(val[13], other.val[13]),
            _mm_cmple_pd(val[14], other.val[14]),
            _mm_cmple_pd(val[15], other.val[15]));
    }

    inline
    short_vec<double, 32> operator==(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_cmpeq_pd(val[ 0], other.val[ 0]),
            _mm_cmpeq_pd(val[ 1], other.val[ 1]),
            _mm_cmpeq_pd(val[ 2], other.val[ 2]),
            _mm_cmpeq_pd(val[ 3], other.val[ 3]),
            _mm_cmpeq_pd(val[ 4], other.val[ 4]),
            _mm_cmpeq_pd(val[ 5], other.val[ 5]),
            _mm_cmpeq_pd(val[ 6], other.val[ 6]),
            _mm_cmpeq_pd(val[ 7], other.val[ 7]),
            _mm_cmpeq_pd(val[ 8], other.val[ 8]),
            _mm_cmpeq_pd(val[ 9], other.val[ 9]),
            _mm_cmpeq_pd(val[10], other.val[10]),
            _mm_cmpeq_pd(val[11], other.val[11]),
            _mm_cmpeq_pd(val[12], other.val[12]),
            _mm_cmpeq_pd(val[13], other.val[13]),
            _mm_cmpeq_pd(val[14], other.val[14]),
            _mm_cmpeq_pd(val[15], other.val[15]));
    }

    inline
    short_vec<double, 32> operator>(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_cmpgt_pd(val[ 0], other.val[ 0]),
            _mm_cmpgt_pd(val[ 1], other.val[ 1]),
            _mm_cmpgt_pd(val[ 2], other.val[ 2]),
            _mm_cmpgt_pd(val[ 3], other.val[ 3]),
            _mm_cmpgt_pd(val[ 4], other.val[ 4]),
            _mm_cmpgt_pd(val[ 5], other.val[ 5]),
            _mm_cmpgt_pd(val[ 6], other.val[ 6]),
            _mm_cmpgt_pd(val[ 7], other.val[ 7]),
            _mm_cmpgt_pd(val[ 8], other.val[ 8]),
            _mm_cmpgt_pd(val[ 9], other.val[ 9]),
            _mm_cmpgt_pd(val[10], other.val[10]),
            _mm_cmpgt_pd(val[11], other.val[11]),
            _mm_cmpgt_pd(val[12], other.val[12]),
            _mm_cmpgt_pd(val[13], other.val[13]),
            _mm_cmpgt_pd(val[14], other.val[14]),
            _mm_cmpgt_pd(val[15], other.val[15]));
    }

    inline
    short_vec<double, 32> operator>=(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_cmpge_pd(val[ 0], other.val[ 0]),
            _mm_cmpge_pd(val[ 1], other.val[ 1]),
            _mm_cmpge_pd(val[ 2], other.val[ 2]),
            _mm_cmpge_pd(val[ 3], other.val[ 3]),
            _mm_cmpge_pd(val[ 4], other.val[ 4]),
            _mm_cmpge_pd(val[ 5], other.val[ 5]),
            _mm_cmpge_pd(val[ 6], other.val[ 6]),
            _mm_cmpge_pd(val[ 7], other.val[ 7]),
            _mm_cmpge_pd(val[ 8], other.val[ 8]),
            _mm_cmpge_pd(val[ 9], other.val[ 9]),
            _mm_cmpge_pd(val[10], other.val[10]),
            _mm_cmpge_pd(val[11], other.val[11]),
            _mm_cmpge_pd(val[12], other.val[12]),
            _mm_cmpge_pd(val[13], other.val[13]),
            _mm_cmpge_pd(val[14], other.val[14]),
            _mm_cmpge_pd(val[15], other.val[15]));
    }

    inline
    short_vec<double, 32> sqrt() const
    {
        return short_vec<double, 32>(
            _mm_sqrt_pd(val[ 0]),
            _mm_sqrt_pd(val[ 1]),
            _mm_sqrt_pd(val[ 2]),
            _mm_sqrt_pd(val[ 3]),
            _mm_sqrt_pd(val[ 4]),
            _mm_sqrt_pd(val[ 5]),
            _mm_sqrt_pd(val[ 6]),
            _mm_sqrt_pd(val[ 7]),
            _mm_sqrt_pd(val[ 8]),
            _mm_sqrt_pd(val[ 9]),
            _mm_sqrt_pd(val[10]),
            _mm_sqrt_pd(val[11]),
            _mm_sqrt_pd(val[12]),
            _mm_sqrt_pd(val[13]),
            _mm_sqrt_pd(val[14]),
            _mm_sqrt_pd(val[15]));
    }

    inline
    void load(const double *data)
    {
        val[ 0] = _mm_loadu_pd(data + 0);
        val[ 1] = _mm_loadu_pd(data + 2);
        val[ 2] = _mm_loadu_pd(data + 4);
        val[ 3] = _mm_loadu_pd(data + 6);
        val[ 4] = _mm_loadu_pd(data + 8);
        val[ 5] = _mm_loadu_pd(data + 10);
        val[ 6] = _mm_loadu_pd(data + 12);
        val[ 7] = _mm_loadu_pd(data + 14);
        val[ 8] = _mm_loadu_pd(data + 16);
        val[ 9] = _mm_loadu_pd(data + 18);
        val[10] = _mm_loadu_pd(data + 20);
        val[11] = _mm_loadu_pd(data + 22);
        val[12] = _mm_loadu_pd(data + 24);
        val[13] = _mm_loadu_pd(data + 26);
        val[14] = _mm_loadu_pd(data + 28);
        val[15] = _mm_loadu_pd(data + 30);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val[ 0] = _mm_load_pd(data +  0);
        val[ 1] = _mm_load_pd(data +  2);
        val[ 2] = _mm_load_pd(data +  4);
        val[ 3] = _mm_load_pd(data +  6);
        val[ 4] = _mm_load_pd(data +  8);
        val[ 5] = _mm_load_pd(data + 10);
        val[ 6] = _mm_load_pd(data + 12);
        val[ 7] = _mm_load_pd(data + 14);
        val[ 8] = _mm_load_pd(data + 16);
        val[ 9] = _mm_load_pd(data + 18);
        val[10] = _mm_load_pd(data + 20);
        val[11] = _mm_load_pd(data + 22);
        val[12] = _mm_load_pd(data + 24);
        val[13] = _mm_load_pd(data + 26);
        val[14] = _mm_load_pd(data + 28);
        val[15] = _mm_load_pd(data + 30);
    }

    inline
    void store(double *data) const
    {
        _mm_storeu_pd(data +  0, val[ 0]);
        _mm_storeu_pd(data +  2, val[ 1]);
        _mm_storeu_pd(data +  4, val[ 2]);
        _mm_storeu_pd(data +  6, val[ 3]);
        _mm_storeu_pd(data +  8, val[ 4]);
        _mm_storeu_pd(data + 10, val[ 5]);
        _mm_storeu_pd(data + 12, val[ 6]);
        _mm_storeu_pd(data + 14, val[ 7]);
        _mm_storeu_pd(data + 16, val[ 8]);
        _mm_storeu_pd(data + 18, val[ 9]);
        _mm_storeu_pd(data + 20, val[10]);
        _mm_storeu_pd(data + 22, val[11]);
        _mm_storeu_pd(data + 24, val[12]);
        _mm_storeu_pd(data + 26, val[13]);
        _mm_storeu_pd(data + 28, val[14]);
        _mm_storeu_pd(data + 30, val[15]);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_pd(data +  0, val[ 0]);
        _mm_store_pd(data +  2, val[ 1]);
        _mm_store_pd(data +  4, val[ 2]);
        _mm_store_pd(data +  6, val[ 3]);
        _mm_store_pd(data +  8, val[ 4]);
        _mm_store_pd(data + 10, val[ 5]);
        _mm_store_pd(data + 12, val[ 6]);
        _mm_store_pd(data + 14, val[ 7]);
        _mm_store_pd(data + 16, val[ 8]);
        _mm_store_pd(data + 18, val[ 9]);
        _mm_store_pd(data + 20, val[10]);
        _mm_store_pd(data + 22, val[11]);
        _mm_store_pd(data + 24, val[12]);
        _mm_store_pd(data + 26, val[13]);
        _mm_store_pd(data + 28, val[14]);
        _mm_store_pd(data + 30, val[15]);
    }

    inline
    void store_nt(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_pd(data +  0, val[ 0]);
        _mm_stream_pd(data +  2, val[ 1]);
        _mm_stream_pd(data +  4, val[ 2]);
        _mm_stream_pd(data +  6, val[ 3]);
        _mm_stream_pd(data +  8, val[ 4]);
        _mm_stream_pd(data + 10, val[ 5]);
        _mm_stream_pd(data + 12, val[ 6]);
        _mm_stream_pd(data + 14, val[ 7]);
        _mm_stream_pd(data + 16, val[ 8]);
        _mm_stream_pd(data + 18, val[ 9]);
        _mm_stream_pd(data + 20, val[10]);
        _mm_stream_pd(data + 22, val[11]);
        _mm_stream_pd(data + 24, val[12]);
        _mm_stream_pd(data + 26, val[13]);
        _mm_stream_pd(data + 28, val[14]);
        _mm_stream_pd(data + 30, val[15]);
    }

    inline
    void gather(const double *ptr, const int *offsets)
    {
        val[ 0] = _mm_loadl_pd(val[ 0], ptr + offsets[ 0]);
        val[ 0] = _mm_loadh_pd(val[ 0], ptr + offsets[ 1]);
        val[ 1] = _mm_loadl_pd(val[ 1], ptr + offsets[ 2]);
        val[ 1] = _mm_loadh_pd(val[ 1], ptr + offsets[ 3]);
        val[ 2] = _mm_loadl_pd(val[ 2], ptr + offsets[ 4]);
        val[ 2] = _mm_loadh_pd(val[ 2], ptr + offsets[ 5]);
        val[ 3] = _mm_loadl_pd(val[ 3], ptr + offsets[ 6]);
        val[ 3] = _mm_loadh_pd(val[ 3], ptr + offsets[ 7]);
        val[ 4] = _mm_loadl_pd(val[ 4], ptr + offsets[ 8]);
        val[ 4] = _mm_loadh_pd(val[ 4], ptr + offsets[ 9]);
        val[ 5] = _mm_loadl_pd(val[ 5], ptr + offsets[10]);
        val[ 5] = _mm_loadh_pd(val[ 5], ptr + offsets[11]);
        val[ 6] = _mm_loadl_pd(val[ 6], ptr + offsets[12]);
        val[ 6] = _mm_loadh_pd(val[ 6], ptr + offsets[13]);
        val[ 7] = _mm_loadl_pd(val[ 7], ptr + offsets[14]);
        val[ 7] = _mm_loadh_pd(val[ 7], ptr + offsets[15]);
        val[ 8] = _mm_loadl_pd(val[ 8], ptr + offsets[16]);
        val[ 8] = _mm_loadh_pd(val[ 8], ptr + offsets[17]);
        val[ 9] = _mm_loadl_pd(val[ 9], ptr + offsets[18]);
        val[ 9] = _mm_loadh_pd(val[ 9], ptr + offsets[19]);
        val[10] = _mm_loadl_pd(val[10], ptr + offsets[20]);
        val[10] = _mm_loadh_pd(val[10], ptr + offsets[21]);
        val[11] = _mm_loadl_pd(val[11], ptr + offsets[22]);
        val[11] = _mm_loadh_pd(val[11], ptr + offsets[23]);
        val[12] = _mm_loadl_pd(val[12], ptr + offsets[24]);
        val[12] = _mm_loadh_pd(val[12], ptr + offsets[25]);
        val[13] = _mm_loadl_pd(val[13], ptr + offsets[26]);
        val[13] = _mm_loadh_pd(val[13], ptr + offsets[27]);
        val[14] = _mm_loadl_pd(val[14], ptr + offsets[28]);
        val[14] = _mm_loadh_pd(val[14], ptr + offsets[29]);
        val[15] = _mm_loadl_pd(val[15], ptr + offsets[30]);
        val[15] = _mm_loadh_pd(val[15], ptr + offsets[31]);
    }

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        _mm_storel_pd(ptr + offsets[ 0], val[ 0]);
        _mm_storeh_pd(ptr + offsets[ 1], val[ 0]);
        _mm_storel_pd(ptr + offsets[ 2], val[ 1]);
        _mm_storeh_pd(ptr + offsets[ 3], val[ 1]);
        _mm_storel_pd(ptr + offsets[ 4], val[ 2]);
        _mm_storeh_pd(ptr + offsets[ 5], val[ 2]);
        _mm_storel_pd(ptr + offsets[ 6], val[ 3]);
        _mm_storeh_pd(ptr + offsets[ 7], val[ 3]);
        _mm_storel_pd(ptr + offsets[ 8], val[ 4]);
        _mm_storeh_pd(ptr + offsets[ 9], val[ 4]);
        _mm_storel_pd(ptr + offsets[10], val[ 5]);
        _mm_storeh_pd(ptr + offsets[11], val[ 5]);
        _mm_storel_pd(ptr + offsets[12], val[ 6]);
        _mm_storeh_pd(ptr + offsets[13], val[ 6]);
        _mm_storel_pd(ptr + offsets[14], val[ 7]);
        _mm_storeh_pd(ptr + offsets[15], val[ 7]);
        _mm_storel_pd(ptr + offsets[16], val[ 8]);
        _mm_storeh_pd(ptr + offsets[17], val[ 8]);
        _mm_storel_pd(ptr + offsets[18], val[ 9]);
        _mm_storeh_pd(ptr + offsets[19], val[ 9]);
        _mm_storel_pd(ptr + offsets[20], val[10]);
        _mm_storeh_pd(ptr + offsets[21], val[10]);
        _mm_storel_pd(ptr + offsets[22], val[11]);
        _mm_storeh_pd(ptr + offsets[23], val[11]);
        _mm_storel_pd(ptr + offsets[24], val[12]);
        _mm_storeh_pd(ptr + offsets[25], val[12]);
        _mm_storel_pd(ptr + offsets[26], val[13]);
        _mm_storeh_pd(ptr + offsets[27], val[13]);
        _mm_storel_pd(ptr + offsets[28], val[14]);
        _mm_storeh_pd(ptr + offsets[29], val[14]);
        _mm_storel_pd(ptr + offsets[30], val[15]);
        _mm_storeh_pd(ptr + offsets[31], val[15]);
    }

    inline
    void blend(const mask_type& mask, const short_vec<double, 32>& other)
    {
#ifdef __SSE4_1__
        val[ 0] = _mm_blendv_pd(val[ 0], other.val[ 0], mask.val[ 0]);
        val[ 1] = _mm_blendv_pd(val[ 1], other.val[ 1], mask.val[ 1]);
        val[ 2] = _mm_blendv_pd(val[ 2], other.val[ 2], mask.val[ 2]);
        val[ 3] = _mm_blendv_pd(val[ 3], other.val[ 3], mask.val[ 3]);
        val[ 4] = _mm_blendv_pd(val[ 4], other.val[ 4], mask.val[ 4]);
        val[ 5] = _mm_blendv_pd(val[ 5], other.val[ 5], mask.val[ 5]);
        val[ 6] = _mm_blendv_pd(val[ 6], other.val[ 6], mask.val[ 6]);
        val[ 7] = _mm_blendv_pd(val[ 7], other.val[ 7], mask.val[ 7]);
        val[ 8] = _mm_blendv_pd(val[ 8], other.val[ 8], mask.val[ 8]);
        val[ 9] = _mm_blendv_pd(val[ 9], other.val[ 9], mask.val[ 9]);
        val[10] = _mm_blendv_pd(val[10], other.val[10], mask.val[10]);
        val[11] = _mm_blendv_pd(val[11], other.val[11], mask.val[11]);
        val[12] = _mm_blendv_pd(val[12], other.val[12], mask.val[12]);
        val[13] = _mm_blendv_pd(val[13], other.val[13], mask.val[13]);
        val[14] = _mm_blendv_pd(val[14], other.val[14], mask.val[14]);
        val[15] = _mm_blendv_pd(val[15], other.val[15], mask.val[15]);
#else
        val[ 0] = _mm_or_pd(
            _mm_and_pd(mask.val[ 0], other.val[ 0]),
            _mm_andnot_pd(mask.val[ 0], val[ 0]));
        val[ 1] = _mm_or_pd(
            _mm_and_pd(mask.val[ 1], other.val[ 1]),
            _mm_andnot_pd(mask.val[ 1], val[ 1]));
        val[ 2] = _mm_or_pd(
            _mm_and_pd(mask.val[ 2], other.val[ 2]),
            _mm_andnot_pd(mask.val[ 2], val[ 2]));
        val[ 3] = _mm_or_pd(
            _mm_and_pd(mask.val[ 3], other.val[ 3]),
            _mm_andnot_pd(mask.val[ 3], val[ 3]));
        val[ 4] = _mm_or_pd(
            _mm_and_pd(mask.val[ 4], other.val[ 4]),
            _mm_andnot_pd(mask.val[ 4], val[ 4]));
        val[ 5] = _mm_or_pd(
            _mm_and_pd(mask.val[ 5], other.val[ 5]),
            _mm_andnot_pd(mask.val[ 5], val[ 5]));
        val[ 6] = _mm_or_pd(
            _mm_and_pd(mask.val[ 6], other.val[ 6]),
            _mm_andnot_pd(mask.val[ 6], val[ 6]));
        val[ 7] = _mm_or_pd(
            _mm_and_pd(mask.val[ 7], other.val[ 7]),
            _mm_andnot_pd(mask.val[ 7], val[ 7]));
        val[ 8] = _mm_or_pd(
            _mm_and_pd(mask.val[ 8], other.val[ 8]),
            _mm_andnot_pd(mask.val[ 8], val[ 8]));
        val[ 9] = _mm_or_pd(
            _mm_and_pd(mask.val[ 9], other.val[ 9]),
            _mm_andnot_pd(mask.val[ 9], val[ 9]));
        val[10] = _mm_or_pd(
            _mm_and_pd(mask.val[10], other.val[10]),
            _mm_andnot_pd(mask.val[10], val[10]));
        val[11] = _mm_or_pd(
            _mm_and_pd(mask.val[11], other.val[11]),
            _mm_andnot_pd(mask.val[11], val[11]));
        val[12] = _mm_or_pd(
            _mm_and_pd(mask.val[12], other.val[12]),
            _mm_andnot_pd(mask.val[12], val[12]));
        val[13] = _mm_or_pd(
            _mm_and_pd(mask.val[13], other.val[13]),
            _mm_andnot_pd(mask.val[13], val[13]));
        val[14] = _mm_or_pd(
            _mm_and_pd(mask.val[14], other.val[14]),
            _mm_andnot_pd(mask.val[14], val[14]));
        val[15] = _mm_or_pd(
            _mm_and_pd(mask.val[15], other.val[15]),
            _mm_andnot_pd(mask.val[15], val[15]));
#endif
    }

private:
    __m128d val[16];
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(double *data, const short_vec<double, 32>& vec)
{
    vec.store(data);
}

inline
short_vec<double, 32> sqrt(const short_vec<double, 32>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 32>& vec)
{
    const double *data1  = reinterpret_cast<const double *>(&vec.val[ 0]);
    const double *data2  = reinterpret_cast<const double *>(&vec.val[ 1]);
    const double *data3  = reinterpret_cast<const double *>(&vec.val[ 2]);
    const double *data4  = reinterpret_cast<const double *>(&vec.val[ 3]);
    const double *data5  = reinterpret_cast<const double *>(&vec.val[ 4]);
    const double *data6  = reinterpret_cast<const double *>(&vec.val[ 5]);
    const double *data7  = reinterpret_cast<const double *>(&vec.val[ 6]);
    const double *data8  = reinterpret_cast<const double *>(&vec.val[ 7]);
    const double *data9  = reinterpret_cast<const double *>(&vec.val[ 8]);
    const double *data10 = reinterpret_cast<const double *>(&vec.val[ 9]);
    const double *data11 = reinterpret_cast<const double *>(&vec.val[10]);
    const double *data12 = reinterpret_cast<const double *>(&vec.val[11]);
    const double *data13 = reinterpret_cast<const double *>(&vec.val[12]);
    const double *data14 = reinterpret_cast<const double *>(&vec.val[13]);
    const double *data15 = reinterpret_cast<const double *>(&vec.val[14]);
    const double *data16 = reinterpret_cast<const double *>(&vec.val[15]);
    __os << "["
         << data1[0]  << ", " << data1[1]  << ", "
         << data2[0]  << ", " << data2[1]  << ", "
         << data3[0]  << ", " << data3[1]  << ", "
         << data4[0]  << ", " << data4[1]  << ", "
         << data5[0]  << ", " << data5[1]  << ", "
         << data6[0]  << ", " << data6[1]  << ", "
         << data7[0]  << ", " << data7[1]  << ", "
         << data8[0]  << ", " << data8[1]  << ", "
         << data9[0]  << ", " << data9[1]  << ", "
         << data10[0] << ", " << data10[1] << ", "
         << data11[0] << ", " << data11[1] << ", "
         << data12[0] << ", " << data12[1] << ", "
         << data13[0] << ", " << data13[1] << ", "
         << data14[0] << ", " << data14[1] << ", "
         << data15[0] << ", " << data15[1] << ", "
         << data16[0] << ", " << data16[1] << "]";
    return __os;
}

}

#endif

#endif
