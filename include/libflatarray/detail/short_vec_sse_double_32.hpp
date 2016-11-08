/**
 * Copyright 2014-2016 Andreas Schäfer
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
        val1( _mm_set1_pd(data)),
        val2( _mm_set1_pd(data)),
        val3( _mm_set1_pd(data)),
        val4( _mm_set1_pd(data)),
        val5( _mm_set1_pd(data)),
        val6( _mm_set1_pd(data)),
        val7( _mm_set1_pd(data)),
        val8( _mm_set1_pd(data)),
        val9( _mm_set1_pd(data)),
        val10(_mm_set1_pd(data)),
        val11(_mm_set1_pd(data)),
        val12(_mm_set1_pd(data)),
        val13(_mm_set1_pd(data)),
        val14(_mm_set1_pd(data)),
        val15(_mm_set1_pd(data)),
        val16(_mm_set1_pd(data))
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
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4),
        val5(val5),
        val6(val6),
        val7(val7),
        val8(val8),
        val9(val9),
        val10(val10),
        val11(val11),
        val12(val12),
        val13(val13),
        val14(val14),
        val15(val15),
        val16(val16)
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
                    _mm_or_pd(val1, val2),
                    _mm_or_pd(val3, val4)),
                _mm_or_pd(
                    _mm_or_pd(val5, val6),
                    _mm_or_pd(val7, val8))),
            _mm_or_pd(
                _mm_or_pd(
                    _mm_or_pd(val9,  val10),
                    _mm_or_pd(val11, val12)),
                _mm_or_pd(
                    _mm_or_pd(val13, val14),
                    _mm_or_pd(val15, val16))));

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
        __m128d buf;
        if (i < 16) {
            if (i < 8) {
                if (i < 4) {
                    if (i < 2) {
                        buf = val1;
                    } else {
                        buf = val2;
                    }
                } else {
                    if (i < 6) {
                        buf = val3;
                    } else {
                        buf = val4;
                    }
                }
            } else {
                if (i < 12) {
                    if (i < 10) {
                        buf = val5;
                    } else {
                        buf = val6;
                    }
                } else {
                    if (i < 14) {
                        buf = val7;
                    } else {
                        buf = val8;
                    }
                }
            }
        } else {
            if (i < 24) {
                if (i < 20) {
                    if (i < 18) {
                        buf = val9;
                    } else {
                        buf = val10;
                    }
                } else {
                    if (i < 22) {
                        buf = val11;
                    } else {
                        buf = val12;
                    }
                }
            } else {
                if (i < 28) {
                    if (i < 26) {
                        buf = val13;
                    } else {
                        buf = val14;
                    }
                } else {
                    if (i < 30) {
                        buf = val15;
                    } else {
                        buf = val16;
                    }
                }
            }
        }

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
        val1  = _mm_sub_pd(val1,  other.val1);
        val2  = _mm_sub_pd(val2,  other.val2);
        val3  = _mm_sub_pd(val3,  other.val3);
        val4  = _mm_sub_pd(val4,  other.val4);
        val5  = _mm_sub_pd(val5,  other.val5);
        val6  = _mm_sub_pd(val6,  other.val6);
        val7  = _mm_sub_pd(val7,  other.val7);
        val8  = _mm_sub_pd(val8,  other.val8);
        val9  = _mm_sub_pd(val9,  other.val9);
        val10 = _mm_sub_pd(val10, other.val10);
        val11 = _mm_sub_pd(val11, other.val11);
        val12 = _mm_sub_pd(val12, other.val12);
        val13 = _mm_sub_pd(val13, other.val13);
        val14 = _mm_sub_pd(val14, other.val14);
        val15 = _mm_sub_pd(val15, other.val15);
        val16 = _mm_sub_pd(val16, other.val16);
    }

    inline
    short_vec<double, 32> operator-(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_sub_pd(val1,  other.val1),
            _mm_sub_pd(val2,  other.val2),
            _mm_sub_pd(val3,  other.val3),
            _mm_sub_pd(val4,  other.val4),
            _mm_sub_pd(val5,  other.val5),
            _mm_sub_pd(val6,  other.val6),
            _mm_sub_pd(val7,  other.val7),
            _mm_sub_pd(val8,  other.val8),
            _mm_sub_pd(val9,  other.val9),
            _mm_sub_pd(val10, other.val10),
            _mm_sub_pd(val11, other.val11),
            _mm_sub_pd(val12, other.val12),
            _mm_sub_pd(val13, other.val13),
            _mm_sub_pd(val14, other.val14),
            _mm_sub_pd(val15, other.val15),
            _mm_sub_pd(val16, other.val16));
    }

    inline
    void operator+=(const short_vec<double, 32>& other)
    {
        val1  = _mm_add_pd(val1,  other.val1);
        val2  = _mm_add_pd(val2,  other.val2);
        val3  = _mm_add_pd(val3,  other.val3);
        val4  = _mm_add_pd(val4,  other.val4);
        val5  = _mm_add_pd(val5,  other.val5);
        val6  = _mm_add_pd(val6,  other.val6);
        val7  = _mm_add_pd(val7,  other.val7);
        val8  = _mm_add_pd(val8,  other.val8);
        val9  = _mm_add_pd(val9,  other.val9);
        val10 = _mm_add_pd(val10, other.val10);
        val11 = _mm_add_pd(val11, other.val11);
        val12 = _mm_add_pd(val12, other.val12);
        val13 = _mm_add_pd(val13, other.val13);
        val14 = _mm_add_pd(val14, other.val14);
        val15 = _mm_add_pd(val15, other.val15);
        val16 = _mm_add_pd(val16, other.val16);
    }

    inline
    short_vec<double, 32> operator+(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_add_pd(val1,  other.val1),
            _mm_add_pd(val2,  other.val2),
            _mm_add_pd(val3,  other.val3),
            _mm_add_pd(val4,  other.val4),
            _mm_add_pd(val5,  other.val5),
            _mm_add_pd(val6,  other.val6),
            _mm_add_pd(val7,  other.val7),
            _mm_add_pd(val8,  other.val8),
            _mm_add_pd(val9,  other.val9),
            _mm_add_pd(val10, other.val10),
            _mm_add_pd(val11, other.val11),
            _mm_add_pd(val12, other.val12),
            _mm_add_pd(val13, other.val13),
            _mm_add_pd(val14, other.val14),
            _mm_add_pd(val15, other.val15),
            _mm_add_pd(val16, other.val16));
    }

    inline
    void operator*=(const short_vec<double, 32>& other)
    {
        val1  = _mm_mul_pd(val1,  other.val1);
        val2  = _mm_mul_pd(val2,  other.val2);
        val3  = _mm_mul_pd(val3,  other.val3);
        val4  = _mm_mul_pd(val4,  other.val4);
        val5  = _mm_mul_pd(val5,  other.val5);
        val6  = _mm_mul_pd(val6,  other.val6);
        val7  = _mm_mul_pd(val7,  other.val7);
        val8  = _mm_mul_pd(val8,  other.val8);
        val9  = _mm_mul_pd(val9,  other.val9);
        val10 = _mm_mul_pd(val10, other.val10);
        val11 = _mm_mul_pd(val11, other.val11);
        val12 = _mm_mul_pd(val12, other.val12);
        val13 = _mm_mul_pd(val13, other.val13);
        val14 = _mm_mul_pd(val14, other.val14);
        val15 = _mm_mul_pd(val15, other.val15);
        val16 = _mm_mul_pd(val16, other.val16);
    }

    inline
    short_vec<double, 32> operator*(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_mul_pd(val1,  other.val1),
            _mm_mul_pd(val2,  other.val2),
            _mm_mul_pd(val3,  other.val3),
            _mm_mul_pd(val4,  other.val4),
            _mm_mul_pd(val5,  other.val5),
            _mm_mul_pd(val6,  other.val6),
            _mm_mul_pd(val7,  other.val7),
            _mm_mul_pd(val8,  other.val8),
            _mm_mul_pd(val9,  other.val9),
            _mm_mul_pd(val10, other.val10),
            _mm_mul_pd(val11, other.val11),
            _mm_mul_pd(val12, other.val12),
            _mm_mul_pd(val13, other.val13),
            _mm_mul_pd(val14, other.val14),
            _mm_mul_pd(val15, other.val15),
            _mm_mul_pd(val16, other.val16));
    }

    inline
    void operator/=(const short_vec<double, 32>& other)
    {
        val1  = _mm_div_pd(val1,  other.val1);
        val2  = _mm_div_pd(val2,  other.val2);
        val3  = _mm_div_pd(val3,  other.val3);
        val4  = _mm_div_pd(val4,  other.val4);
        val5  = _mm_div_pd(val5,  other.val5);
        val6  = _mm_div_pd(val6,  other.val6);
        val7  = _mm_div_pd(val7,  other.val7);
        val8  = _mm_div_pd(val8,  other.val8);
        val9  = _mm_div_pd(val9,  other.val9);
        val10 = _mm_div_pd(val10, other.val10);
        val11 = _mm_div_pd(val11, other.val11);
        val12 = _mm_div_pd(val12, other.val12);
        val13 = _mm_div_pd(val13, other.val13);
        val14 = _mm_div_pd(val14, other.val14);
        val15 = _mm_div_pd(val15, other.val15);
        val16 = _mm_div_pd(val16, other.val16);
    }

    inline
    short_vec<double, 32> operator/(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_div_pd(val1,  other.val1),
            _mm_div_pd(val2,  other.val2),
            _mm_div_pd(val3,  other.val3),
            _mm_div_pd(val4,  other.val4),
            _mm_div_pd(val5,  other.val5),
            _mm_div_pd(val6,  other.val6),
            _mm_div_pd(val7,  other.val7),
            _mm_div_pd(val8,  other.val8),
            _mm_div_pd(val9,  other.val9),
            _mm_div_pd(val10, other.val10),
            _mm_div_pd(val11, other.val11),
            _mm_div_pd(val12, other.val12),
            _mm_div_pd(val13, other.val13),
            _mm_div_pd(val14, other.val14),
            _mm_div_pd(val15, other.val15),
            _mm_div_pd(val16, other.val16));
    }

    inline
    short_vec<double, 32> operator/(const sqrt_reference<double, 32>& other) const;

    inline
    short_vec<double, 32> operator<(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_cmplt_pd(val1,  other.val1),
            _mm_cmplt_pd(val2,  other.val2),
            _mm_cmplt_pd(val3,  other.val3),
            _mm_cmplt_pd(val4,  other.val4),
            _mm_cmplt_pd(val5,  other.val5),
            _mm_cmplt_pd(val6,  other.val6),
            _mm_cmplt_pd(val7,  other.val7),
            _mm_cmplt_pd(val8,  other.val8),
            _mm_cmplt_pd(val9,  other.val9),
            _mm_cmplt_pd(val10, other.val10),
            _mm_cmplt_pd(val11, other.val11),
            _mm_cmplt_pd(val12, other.val12),
            _mm_cmplt_pd(val13, other.val13),
            _mm_cmplt_pd(val14, other.val14),
            _mm_cmplt_pd(val15, other.val15),
            _mm_cmplt_pd(val16, other.val16));
    }

    inline
    short_vec<double, 32> operator<=(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_cmple_pd(val1,  other.val1),
            _mm_cmple_pd(val2,  other.val2),
            _mm_cmple_pd(val3,  other.val3),
            _mm_cmple_pd(val4,  other.val4),
            _mm_cmple_pd(val5,  other.val5),
            _mm_cmple_pd(val6,  other.val6),
            _mm_cmple_pd(val7,  other.val7),
            _mm_cmple_pd(val8,  other.val8),
            _mm_cmple_pd(val9,  other.val9),
            _mm_cmple_pd(val10, other.val10),
            _mm_cmple_pd(val11, other.val11),
            _mm_cmple_pd(val12, other.val12),
            _mm_cmple_pd(val13, other.val13),
            _mm_cmple_pd(val14, other.val14),
            _mm_cmple_pd(val15, other.val15),
            _mm_cmple_pd(val16, other.val16));
    }

    inline
    short_vec<double, 32> operator==(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_cmpeq_pd(val1,  other.val1),
            _mm_cmpeq_pd(val2,  other.val2),
            _mm_cmpeq_pd(val3,  other.val3),
            _mm_cmpeq_pd(val4,  other.val4),
            _mm_cmpeq_pd(val5,  other.val5),
            _mm_cmpeq_pd(val6,  other.val6),
            _mm_cmpeq_pd(val7,  other.val7),
            _mm_cmpeq_pd(val8,  other.val8),
            _mm_cmpeq_pd(val9,  other.val9),
            _mm_cmpeq_pd(val10, other.val10),
            _mm_cmpeq_pd(val11, other.val11),
            _mm_cmpeq_pd(val12, other.val12),
            _mm_cmpeq_pd(val13, other.val13),
            _mm_cmpeq_pd(val14, other.val14),
            _mm_cmpeq_pd(val15, other.val15),
            _mm_cmpeq_pd(val16, other.val16));
    }

    inline
    short_vec<double, 32> operator>(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_cmpgt_pd(val1,  other.val1),
            _mm_cmpgt_pd(val2,  other.val2),
            _mm_cmpgt_pd(val3,  other.val3),
            _mm_cmpgt_pd(val4,  other.val4),
            _mm_cmpgt_pd(val5,  other.val5),
            _mm_cmpgt_pd(val6,  other.val6),
            _mm_cmpgt_pd(val7,  other.val7),
            _mm_cmpgt_pd(val8,  other.val8),
            _mm_cmpgt_pd(val9,  other.val9),
            _mm_cmpgt_pd(val10, other.val10),
            _mm_cmpgt_pd(val11, other.val11),
            _mm_cmpgt_pd(val12, other.val12),
            _mm_cmpgt_pd(val13, other.val13),
            _mm_cmpgt_pd(val14, other.val14),
            _mm_cmpgt_pd(val15, other.val15),
            _mm_cmpgt_pd(val16, other.val16));
    }

    inline
    short_vec<double, 32> operator>=(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm_cmpge_pd(val1,  other.val1),
            _mm_cmpge_pd(val2,  other.val2),
            _mm_cmpge_pd(val3,  other.val3),
            _mm_cmpge_pd(val4,  other.val4),
            _mm_cmpge_pd(val5,  other.val5),
            _mm_cmpge_pd(val6,  other.val6),
            _mm_cmpge_pd(val7,  other.val7),
            _mm_cmpge_pd(val8,  other.val8),
            _mm_cmpge_pd(val9,  other.val9),
            _mm_cmpge_pd(val10, other.val10),
            _mm_cmpge_pd(val11, other.val11),
            _mm_cmpge_pd(val12, other.val12),
            _mm_cmpge_pd(val13, other.val13),
            _mm_cmpge_pd(val14, other.val14),
            _mm_cmpge_pd(val15, other.val15),
            _mm_cmpge_pd(val16, other.val16));
    }

    inline
    short_vec<double, 32> sqrt() const
    {
        return short_vec<double, 32>(
            _mm_sqrt_pd(val1),
            _mm_sqrt_pd(val2),
            _mm_sqrt_pd(val3),
            _mm_sqrt_pd(val4),
            _mm_sqrt_pd(val5),
            _mm_sqrt_pd(val6),
            _mm_sqrt_pd(val7),
            _mm_sqrt_pd(val8),
            _mm_sqrt_pd(val9),
            _mm_sqrt_pd(val10),
            _mm_sqrt_pd(val11),
            _mm_sqrt_pd(val12),
            _mm_sqrt_pd(val13),
            _mm_sqrt_pd(val14),
            _mm_sqrt_pd(val15),
            _mm_sqrt_pd(val16));
    }

    inline
    void load(const double *data)
    {
        val1  = _mm_loadu_pd(data + 0);
        val2  = _mm_loadu_pd(data + 2);
        val3  = _mm_loadu_pd(data + 4);
        val4  = _mm_loadu_pd(data + 6);
        val5  = _mm_loadu_pd(data + 8);
        val6  = _mm_loadu_pd(data + 10);
        val7  = _mm_loadu_pd(data + 12);
        val8  = _mm_loadu_pd(data + 14);
        val9  = _mm_loadu_pd(data + 16);
        val10 = _mm_loadu_pd(data + 18);
        val11 = _mm_loadu_pd(data + 20);
        val12 = _mm_loadu_pd(data + 22);
        val13 = _mm_loadu_pd(data + 24);
        val14 = _mm_loadu_pd(data + 26);
        val15 = _mm_loadu_pd(data + 28);
        val16 = _mm_loadu_pd(data + 30);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val1  = _mm_load_pd(data +  0);
        val2  = _mm_load_pd(data +  2);
        val3  = _mm_load_pd(data +  4);
        val4  = _mm_load_pd(data +  6);
        val5  = _mm_load_pd(data +  8);
        val6  = _mm_load_pd(data + 10);
        val7  = _mm_load_pd(data + 12);
        val8  = _mm_load_pd(data + 14);
        val9  = _mm_load_pd(data + 16);
        val10 = _mm_load_pd(data + 18);
        val11 = _mm_load_pd(data + 20);
        val12 = _mm_load_pd(data + 22);
        val13 = _mm_load_pd(data + 24);
        val14 = _mm_load_pd(data + 26);
        val15 = _mm_load_pd(data + 28);
        val16 = _mm_load_pd(data + 30);
    }

    inline
    void store(double *data) const
    {
        _mm_storeu_pd(data +  0, val1);
        _mm_storeu_pd(data +  2, val2);
        _mm_storeu_pd(data +  4, val3);
        _mm_storeu_pd(data +  6, val4);
        _mm_storeu_pd(data +  8, val5);
        _mm_storeu_pd(data + 10, val6);
        _mm_storeu_pd(data + 12, val7);
        _mm_storeu_pd(data + 14, val8);
        _mm_storeu_pd(data + 16, val9);
        _mm_storeu_pd(data + 18, val10);
        _mm_storeu_pd(data + 20, val11);
        _mm_storeu_pd(data + 22, val12);
        _mm_storeu_pd(data + 24, val13);
        _mm_storeu_pd(data + 26, val14);
        _mm_storeu_pd(data + 28, val15);
        _mm_storeu_pd(data + 30, val16);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_pd(data +  0, val1);
        _mm_store_pd(data +  2, val2);
        _mm_store_pd(data +  4, val3);
        _mm_store_pd(data +  6, val4);
        _mm_store_pd(data +  8, val5);
        _mm_store_pd(data + 10, val6);
        _mm_store_pd(data + 12, val7);
        _mm_store_pd(data + 14, val8);
        _mm_store_pd(data + 16, val9);
        _mm_store_pd(data + 18, val10);
        _mm_store_pd(data + 20, val11);
        _mm_store_pd(data + 22, val12);
        _mm_store_pd(data + 24, val13);
        _mm_store_pd(data + 26, val14);
        _mm_store_pd(data + 28, val15);
        _mm_store_pd(data + 30, val16);
    }

    inline
    void store_nt(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_pd(data +  0, val1);
        _mm_stream_pd(data +  2, val2);
        _mm_stream_pd(data +  4, val3);
        _mm_stream_pd(data +  6, val4);
        _mm_stream_pd(data +  8, val5);
        _mm_stream_pd(data + 10, val6);
        _mm_stream_pd(data + 12, val7);
        _mm_stream_pd(data + 14, val8);
        _mm_stream_pd(data + 16, val9);
        _mm_stream_pd(data + 18, val10);
        _mm_stream_pd(data + 20, val11);
        _mm_stream_pd(data + 22, val12);
        _mm_stream_pd(data + 24, val13);
        _mm_stream_pd(data + 26, val14);
        _mm_stream_pd(data + 28, val15);
        _mm_stream_pd(data + 30, val16);
    }

    inline
    void gather(const double *ptr, const int *offsets)
    {
        val1  = _mm_loadl_pd(val1,  ptr + offsets[ 0]);
        val1  = _mm_loadh_pd(val1,  ptr + offsets[ 1]);
        val2  = _mm_loadl_pd(val2,  ptr + offsets[ 2]);
        val2  = _mm_loadh_pd(val2,  ptr + offsets[ 3]);
        val3  = _mm_loadl_pd(val3,  ptr + offsets[ 4]);
        val3  = _mm_loadh_pd(val3,  ptr + offsets[ 5]);
        val4  = _mm_loadl_pd(val4,  ptr + offsets[ 6]);
        val4  = _mm_loadh_pd(val4,  ptr + offsets[ 7]);
        val5  = _mm_loadl_pd(val5,  ptr + offsets[ 8]);
        val5  = _mm_loadh_pd(val5,  ptr + offsets[ 9]);
        val6  = _mm_loadl_pd(val6,  ptr + offsets[10]);
        val6  = _mm_loadh_pd(val6,  ptr + offsets[11]);
        val7  = _mm_loadl_pd(val7,  ptr + offsets[12]);
        val7  = _mm_loadh_pd(val7,  ptr + offsets[13]);
        val8  = _mm_loadl_pd(val8,  ptr + offsets[14]);
        val8  = _mm_loadh_pd(val8,  ptr + offsets[15]);
        val9  = _mm_loadl_pd(val9,  ptr + offsets[16]);
        val9  = _mm_loadh_pd(val9,  ptr + offsets[17]);
        val10 = _mm_loadl_pd(val10, ptr + offsets[18]);
        val10 = _mm_loadh_pd(val10, ptr + offsets[19]);
        val11 = _mm_loadl_pd(val11, ptr + offsets[20]);
        val11 = _mm_loadh_pd(val11, ptr + offsets[21]);
        val12 = _mm_loadl_pd(val12, ptr + offsets[22]);
        val12 = _mm_loadh_pd(val12, ptr + offsets[23]);
        val13 = _mm_loadl_pd(val13, ptr + offsets[24]);
        val13 = _mm_loadh_pd(val13, ptr + offsets[25]);
        val14 = _mm_loadl_pd(val14, ptr + offsets[26]);
        val14 = _mm_loadh_pd(val14, ptr + offsets[27]);
        val15 = _mm_loadl_pd(val15, ptr + offsets[28]);
        val15 = _mm_loadh_pd(val15, ptr + offsets[29]);
        val16 = _mm_loadl_pd(val16, ptr + offsets[30]);
        val16 = _mm_loadh_pd(val16, ptr + offsets[31]);
    }

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        _mm_storel_pd(ptr + offsets[ 0], val1);
        _mm_storeh_pd(ptr + offsets[ 1], val1);
        _mm_storel_pd(ptr + offsets[ 2], val2);
        _mm_storeh_pd(ptr + offsets[ 3], val2);
        _mm_storel_pd(ptr + offsets[ 4], val3);
        _mm_storeh_pd(ptr + offsets[ 5], val3);
        _mm_storel_pd(ptr + offsets[ 6], val4);
        _mm_storeh_pd(ptr + offsets[ 7], val4);
        _mm_storel_pd(ptr + offsets[ 8], val5);
        _mm_storeh_pd(ptr + offsets[ 9], val5);
        _mm_storel_pd(ptr + offsets[10], val6);
        _mm_storeh_pd(ptr + offsets[11], val6);
        _mm_storel_pd(ptr + offsets[12], val7);
        _mm_storeh_pd(ptr + offsets[13], val7);
        _mm_storel_pd(ptr + offsets[14], val8);
        _mm_storeh_pd(ptr + offsets[15], val8);
        _mm_storel_pd(ptr + offsets[16], val9);
        _mm_storeh_pd(ptr + offsets[17], val9);
        _mm_storel_pd(ptr + offsets[18], val10);
        _mm_storeh_pd(ptr + offsets[19], val10);
        _mm_storel_pd(ptr + offsets[20], val11);
        _mm_storeh_pd(ptr + offsets[21], val11);
        _mm_storel_pd(ptr + offsets[22], val12);
        _mm_storeh_pd(ptr + offsets[23], val12);
        _mm_storel_pd(ptr + offsets[24], val13);
        _mm_storeh_pd(ptr + offsets[25], val13);
        _mm_storel_pd(ptr + offsets[26], val14);
        _mm_storeh_pd(ptr + offsets[27], val14);
        _mm_storel_pd(ptr + offsets[28], val15);
        _mm_storeh_pd(ptr + offsets[29], val15);
        _mm_storel_pd(ptr + offsets[30], val16);
        _mm_storeh_pd(ptr + offsets[31], val16);
    }

    inline
    void blend(const mask_type& mask, const short_vec<double, 32>& other)
    {
#ifdef __SSE4_1__
        val1  = _mm_blendv_pd(val1,  other.val1,  mask.val1);
        val2  = _mm_blendv_pd(val2,  other.val2,  mask.val2);
        val3  = _mm_blendv_pd(val3,  other.val3,  mask.val3);
        val4  = _mm_blendv_pd(val4,  other.val4,  mask.val4);
        val5  = _mm_blendv_pd(val5,  other.val5,  mask.val5);
        val6  = _mm_blendv_pd(val6,  other.val6,  mask.val6);
        val7  = _mm_blendv_pd(val7,  other.val7,  mask.val7);
        val8  = _mm_blendv_pd(val8,  other.val8,  mask.val8);
        val9  = _mm_blendv_pd(val9,  other.val9,  mask.val9);
        val10 = _mm_blendv_pd(val10, other.val10, mask.val10);
        val11 = _mm_blendv_pd(val11, other.val11, mask.val11);
        val12 = _mm_blendv_pd(val12, other.val12, mask.val12);
        val13 = _mm_blendv_pd(val13, other.val13, mask.val13);
        val14 = _mm_blendv_pd(val14, other.val14, mask.val14);
        val15 = _mm_blendv_pd(val15, other.val15, mask.val15);
        val16 = _mm_blendv_pd(val16, other.val16, mask.val16);
#else
        val1 = _mm_or_pd(
            _mm_and_pd(mask.val1, other.val1),
            _mm_andnot_pd(mask.val1, val1));
        val2 = _mm_or_pd(
            _mm_and_pd(mask.val2, other.val2),
            _mm_andnot_pd(mask.val2, val2));
        val3 = _mm_or_pd(
            _mm_and_pd(mask.val3, other.val3),
            _mm_andnot_pd(mask.val3, val3));
        val4 = _mm_or_pd(
            _mm_and_pd(mask.val4, other.val4),
            _mm_andnot_pd(mask.val4, val4));
        val5 = _mm_or_pd(
            _mm_and_pd(mask.val5, other.val5),
            _mm_andnot_pd(mask.val5, val5));
        val6 = _mm_or_pd(
            _mm_and_pd(mask.val6, other.val6),
            _mm_andnot_pd(mask.val6, val6));
        val7 = _mm_or_pd(
            _mm_and_pd(mask.val7, other.val7),
            _mm_andnot_pd(mask.val7, val7));
        val8 = _mm_or_pd(
            _mm_and_pd(mask.val8, other.val8),
            _mm_andnot_pd(mask.val8, val8));
        val9 = _mm_or_pd(
            _mm_and_pd(mask.val9, other.val9),
            _mm_andnot_pd(mask.val9, val9));
        val10 = _mm_or_pd(
            _mm_and_pd(mask.val10, other.val10),
            _mm_andnot_pd(mask.val10, val10));
        val11 = _mm_or_pd(
            _mm_and_pd(mask.val11, other.val11),
            _mm_andnot_pd(mask.val11, val11));
        val12 = _mm_or_pd(
            _mm_and_pd(mask.val12, other.val12),
            _mm_andnot_pd(mask.val12, val12));
        val13 = _mm_or_pd(
            _mm_and_pd(mask.val13, other.val13),
            _mm_andnot_pd(mask.val13, val13));
        val14 = _mm_or_pd(
            _mm_and_pd(mask.val14, other.val14),
            _mm_andnot_pd(mask.val14, val14));
        val15 = _mm_or_pd(
            _mm_and_pd(mask.val15, other.val15),
            _mm_andnot_pd(mask.val15, val15));
        val16 = _mm_or_pd(
            _mm_and_pd(mask.val16, other.val16),
            _mm_andnot_pd(mask.val16, val16));
#endif
    }

private:
    __m128d val1;
    __m128d val2;
    __m128d val3;
    __m128d val4;
    __m128d val5;
    __m128d val6;
    __m128d val7;
    __m128d val8;
    __m128d val9;
    __m128d val10;
    __m128d val11;
    __m128d val12;
    __m128d val13;
    __m128d val14;
    __m128d val15;
    __m128d val16;
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
    const double *data1  = reinterpret_cast<const double *>(&vec.val1);
    const double *data2  = reinterpret_cast<const double *>(&vec.val2);
    const double *data3  = reinterpret_cast<const double *>(&vec.val3);
    const double *data4  = reinterpret_cast<const double *>(&vec.val4);
    const double *data5  = reinterpret_cast<const double *>(&vec.val5);
    const double *data6  = reinterpret_cast<const double *>(&vec.val6);
    const double *data7  = reinterpret_cast<const double *>(&vec.val7);
    const double *data8  = reinterpret_cast<const double *>(&vec.val8);
    const double *data9  = reinterpret_cast<const double *>(&vec.val9);
    const double *data10 = reinterpret_cast<const double *>(&vec.val10);
    const double *data11 = reinterpret_cast<const double *>(&vec.val11);
    const double *data12 = reinterpret_cast<const double *>(&vec.val12);
    const double *data13 = reinterpret_cast<const double *>(&vec.val13);
    const double *data14 = reinterpret_cast<const double *>(&vec.val14);
    const double *data15 = reinterpret_cast<const double *>(&vec.val15);
    const double *data16 = reinterpret_cast<const double *>(&vec.val16);
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
