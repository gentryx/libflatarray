/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_16_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE2) ||            \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE4_1)

#include <emmintrin.h>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

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
    typedef short_vec<double, 16> mask_type;
    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 16>& vec);

    inline
    short_vec(const double data = 0) :
        val1(_mm_set1_pd(data)),
        val2(_mm_set1_pd(data)),
        val3(_mm_set1_pd(data)),
        val4(_mm_set1_pd(data)),
        val5(_mm_set1_pd(data)),
        val6(_mm_set1_pd(data)),
        val7(_mm_set1_pd(data)),
        val8(_mm_set1_pd(data))
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
        const __m128d& val8) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4),
        val5(val5),
        val6(val6),
        val7(val7),
        val8(val8)
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
        __m128d buf1 = _mm_or_pd(
            _mm_or_pd(
                _mm_or_pd(val1, val2),
                _mm_or_pd(val3, val4)),
            _mm_or_pd(
                _mm_or_pd(val5, val6),
                _mm_or_pd(val7, val8)));
        __m128d buf2 = _mm_shuffle_pd(buf1, buf1, 1);

        return _mm_cvtsd_f64(buf1) || _mm_cvtsd_f64(buf2);
    }

    inline
    double get(int i) const
    {
        __m128d buf;
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

        i &= 1;

        if (i == 0) {
            return _mm_cvtsd_f64(buf);
        }

        buf = _mm_shuffle_pd(buf, buf, 1);
        return _mm_cvtsd_f64(buf);
    }

    inline
    void operator-=(const short_vec<double, 16>& other)
    {
        val1 = _mm_sub_pd(val1, other.val1);
        val2 = _mm_sub_pd(val2, other.val2);
        val3 = _mm_sub_pd(val3, other.val3);
        val4 = _mm_sub_pd(val4, other.val4);
        val5 = _mm_sub_pd(val5, other.val5);
        val6 = _mm_sub_pd(val6, other.val6);
        val7 = _mm_sub_pd(val7, other.val7);
        val8 = _mm_sub_pd(val8, other.val8);
    }

    inline
    short_vec<double, 16> operator-(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm_sub_pd(val1, other.val1),
            _mm_sub_pd(val2, other.val2),
            _mm_sub_pd(val3, other.val3),
            _mm_sub_pd(val4, other.val4),
            _mm_sub_pd(val5, other.val5),
            _mm_sub_pd(val6, other.val6),
            _mm_sub_pd(val7, other.val7),
            _mm_sub_pd(val8, other.val8));
    }

    inline
    void operator+=(const short_vec<double, 16>& other)
    {
        val1 = _mm_add_pd(val1, other.val1);
        val2 = _mm_add_pd(val2, other.val2);
        val3 = _mm_add_pd(val3, other.val3);
        val4 = _mm_add_pd(val4, other.val4);
        val5 = _mm_add_pd(val5, other.val5);
        val6 = _mm_add_pd(val6, other.val6);
        val7 = _mm_add_pd(val7, other.val7);
        val8 = _mm_add_pd(val8, other.val8);
    }

    inline
    short_vec<double, 16> operator+(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm_add_pd(val1, other.val1),
            _mm_add_pd(val2, other.val2),
            _mm_add_pd(val3, other.val3),
            _mm_add_pd(val4, other.val4),
            _mm_add_pd(val5, other.val5),
            _mm_add_pd(val6, other.val6),
            _mm_add_pd(val7, other.val7),
            _mm_add_pd(val8, other.val8));
    }

    inline
    void operator*=(const short_vec<double, 16>& other)
    {
        val1 = _mm_mul_pd(val1, other.val1);
        val2 = _mm_mul_pd(val2, other.val2);
        val3 = _mm_mul_pd(val3, other.val3);
        val4 = _mm_mul_pd(val4, other.val4);
        val5 = _mm_mul_pd(val5, other.val5);
        val6 = _mm_mul_pd(val6, other.val6);
        val7 = _mm_mul_pd(val7, other.val7);
        val8 = _mm_mul_pd(val8, other.val8);
    }

    inline
    short_vec<double, 16> operator*(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm_mul_pd(val1, other.val1),
            _mm_mul_pd(val2, other.val2),
            _mm_mul_pd(val3, other.val3),
            _mm_mul_pd(val4, other.val4),
            _mm_mul_pd(val5, other.val5),
            _mm_mul_pd(val6, other.val6),
            _mm_mul_pd(val7, other.val7),
            _mm_mul_pd(val8, other.val8));
    }

    inline
    void operator/=(const short_vec<double, 16>& other)
    {
        val1 = _mm_div_pd(val1, other.val1);
        val2 = _mm_div_pd(val2, other.val2);
        val3 = _mm_div_pd(val3, other.val3);
        val4 = _mm_div_pd(val4, other.val4);
        val5 = _mm_div_pd(val5, other.val5);
        val6 = _mm_div_pd(val6, other.val6);
        val7 = _mm_div_pd(val7, other.val7);
        val8 = _mm_div_pd(val8, other.val8);
    }

    inline
    short_vec<double, 16> operator/(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm_div_pd(val1, other.val1),
            _mm_div_pd(val2, other.val2),
            _mm_div_pd(val3, other.val3),
            _mm_div_pd(val4, other.val4),
            _mm_div_pd(val5, other.val5),
            _mm_div_pd(val6, other.val6),
            _mm_div_pd(val7, other.val7),
            _mm_div_pd(val8, other.val8));
    }

    inline
    short_vec<double, 16> operator/(const sqrt_reference<double, 16>& other) const;

    inline
    short_vec<double, 16> operator<(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm_cmplt_pd(val1, other.val1),
            _mm_cmplt_pd(val2, other.val2),
            _mm_cmplt_pd(val3, other.val3),
            _mm_cmplt_pd(val4, other.val4),
            _mm_cmplt_pd(val5, other.val5),
            _mm_cmplt_pd(val6, other.val6),
            _mm_cmplt_pd(val7, other.val7),
            _mm_cmplt_pd(val8, other.val8));
    }

    inline
    short_vec<double, 16> operator<=(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm_cmple_pd(val1, other.val1),
            _mm_cmple_pd(val2, other.val2),
            _mm_cmple_pd(val3, other.val3),
            _mm_cmple_pd(val4, other.val4),
            _mm_cmple_pd(val5, other.val5),
            _mm_cmple_pd(val6, other.val6),
            _mm_cmple_pd(val7, other.val7),
            _mm_cmple_pd(val8, other.val8));
    }

    inline
    short_vec<double, 16> operator==(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm_cmpeq_pd(val1, other.val1),
            _mm_cmpeq_pd(val2, other.val2),
            _mm_cmpeq_pd(val3, other.val3),
            _mm_cmpeq_pd(val4, other.val4),
            _mm_cmpeq_pd(val5, other.val5),
            _mm_cmpeq_pd(val6, other.val6),
            _mm_cmpeq_pd(val7, other.val7),
            _mm_cmpeq_pd(val8, other.val8));
    }

    inline
    short_vec<double, 16> operator>(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm_cmpgt_pd(val1, other.val1),
            _mm_cmpgt_pd(val2, other.val2),
            _mm_cmpgt_pd(val3, other.val3),
            _mm_cmpgt_pd(val4, other.val4),
            _mm_cmpgt_pd(val5, other.val5),
            _mm_cmpgt_pd(val6, other.val6),
            _mm_cmpgt_pd(val7, other.val7),
            _mm_cmpgt_pd(val8, other.val8));
    }

    inline
    short_vec<double, 16> operator>=(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm_cmpge_pd(val1, other.val1),
            _mm_cmpge_pd(val2, other.val2),
            _mm_cmpge_pd(val3, other.val3),
            _mm_cmpge_pd(val4, other.val4),
            _mm_cmpge_pd(val5, other.val5),
            _mm_cmpge_pd(val6, other.val6),
            _mm_cmpge_pd(val7, other.val7),
            _mm_cmpge_pd(val8, other.val8));
    }

    inline
    short_vec<double, 16> sqrt() const
    {
        return short_vec<double, 16>(
            _mm_sqrt_pd(val1),
            _mm_sqrt_pd(val2),
            _mm_sqrt_pd(val3),
            _mm_sqrt_pd(val4),
            _mm_sqrt_pd(val5),
            _mm_sqrt_pd(val6),
            _mm_sqrt_pd(val7),
            _mm_sqrt_pd(val8));
    }

    inline
    void load(const double *data)
    {
        val1 = _mm_loadu_pd(data + 0);
        val2 = _mm_loadu_pd(data + 2);
        val3 = _mm_loadu_pd(data + 4);
        val4 = _mm_loadu_pd(data + 6);
        val5 = _mm_loadu_pd(data + 8);
        val6 = _mm_loadu_pd(data + 10);
        val7 = _mm_loadu_pd(data + 12);
        val8 = _mm_loadu_pd(data + 14);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val1 = _mm_load_pd(data +  0);
        val2 = _mm_load_pd(data +  2);
        val3 = _mm_load_pd(data +  4);
        val4 = _mm_load_pd(data +  6);
        val5 = _mm_load_pd(data +  8);
        val6 = _mm_load_pd(data + 10);
        val7 = _mm_load_pd(data + 12);
        val8 = _mm_load_pd(data + 14);
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
    }

    inline
    void gather(const double *ptr, const int *offsets)
    {
        val1 = _mm_loadl_pd(val1, ptr + offsets[ 0]);
        val1 = _mm_loadh_pd(val1, ptr + offsets[ 1]);
        val2 = _mm_loadl_pd(val2, ptr + offsets[ 2]);
        val2 = _mm_loadh_pd(val2, ptr + offsets[ 3]);
        val3 = _mm_loadl_pd(val3, ptr + offsets[ 4]);
        val3 = _mm_loadh_pd(val3, ptr + offsets[ 5]);
        val4 = _mm_loadl_pd(val4, ptr + offsets[ 6]);
        val4 = _mm_loadh_pd(val4, ptr + offsets[ 7]);
        val5 = _mm_loadl_pd(val5, ptr + offsets[ 8]);
        val5 = _mm_loadh_pd(val5, ptr + offsets[ 9]);
        val6 = _mm_loadl_pd(val6, ptr + offsets[10]);
        val6 = _mm_loadh_pd(val6, ptr + offsets[11]);
        val7 = _mm_loadl_pd(val7, ptr + offsets[12]);
        val7 = _mm_loadh_pd(val7, ptr + offsets[13]);
        val8 = _mm_loadl_pd(val8, ptr + offsets[14]);
        val8 = _mm_loadh_pd(val8, ptr + offsets[15]);
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
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(double *data, const short_vec<double, 16>& vec)
{
    vec.store(data);
}

inline
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
    const double *data5 = reinterpret_cast<const double *>(&vec.val5);
    const double *data6 = reinterpret_cast<const double *>(&vec.val6);
    const double *data7 = reinterpret_cast<const double *>(&vec.val7);
    const double *data8 = reinterpret_cast<const double *>(&vec.val8);
    __os << "["
         << data1[0] << ", " << data1[1] << ", "
         << data2[0] << ", " << data2[1] << ", "
         << data3[0] << ", " << data3[1] << ", "
         << data4[0] << ", " << data4[1] << ", "
         << data5[0] << ", " << data5[1] << ", "
         << data6[0] << ", " << data6[1] << ", "
         << data7[0] << ", " << data7[1] << ", "
         << data8[0] << ", " << data8[1] << "]";
    return __os;
}

}

#endif

#endif
