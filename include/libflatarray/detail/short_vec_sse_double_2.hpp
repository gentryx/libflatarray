/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_2_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_DOUBLE_2_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE2) ||            \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE4_1) ||          \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX2) ||            \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX512F)

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
class short_vec<double, 2> : public short_vec_base<double, 2>
{
public:
    static const std::size_t ARITY = 2;
    typedef short_vec<double, 2> mask_type;
    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 2>& vec);

    inline
    short_vec(const double data = 0) :
        val(_mm_set1_pd(data))
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(const __m128d& val) :
        val(val)
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
#ifdef __SSE4_1__
        return (0 == _mm_testz_si128(
                    _mm_castpd_si128(val),
                    _mm_castpd_si128(val)));
#else
        __m128d buf0 = _mm_shuffle_pd(val, val, 1);
        return _mm_cvtsd_f64(buf0) || _mm_cvtsd_f64(val);
#endif
    }

    inline
    double operator[](const int i) const
    {
        if (i == 0) {
            return _mm_cvtsd_f64(val);
        }

        __m128d buf = _mm_shuffle_pd(val, val, 1);
        return _mm_cvtsd_f64(buf);
    }

    inline
    void operator-=(const short_vec<double, 2>& other)
    {
        val = _mm_sub_pd(val, other.val);
    }

    inline
    short_vec<double, 2> operator-(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_sub_pd(val, other.val));
    }

    inline
    void operator+=(const short_vec<double, 2>& other)
    {
        val = _mm_add_pd(val, other.val);
    }

    inline
    short_vec<double, 2> operator+(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_add_pd(val, other.val));
    }

    inline
    void operator*=(const short_vec<double, 2>& other)
    {
        val = _mm_mul_pd(val, other.val);
    }

    inline
    short_vec<double, 2> operator*(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_mul_pd(val, other.val));
    }

    inline
    void operator/=(const short_vec<double, 2>& other)
    {
        val = _mm_div_pd(val, other.val);
    }

    inline
    short_vec<double, 2> operator/(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_div_pd(val, other.val));
    }

    inline
    short_vec<double, 2> operator<(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_cmplt_pd(val, other.val));
    }

    inline
    short_vec<double, 2> operator<=(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_cmple_pd(val, other.val));
    }

    inline
    short_vec<double, 2> operator==(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_cmpeq_pd(val, other.val));
    }

    inline
    short_vec<double, 2> operator>(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_cmpgt_pd(val, other.val));
    }

    inline
    short_vec<double, 2> operator>=(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            _mm_cmpge_pd(val, other.val));
    }

    inline
    short_vec<double, 2> sqrt() const
    {
        return short_vec<double, 2>(
            _mm_sqrt_pd(val));
    }

    inline
    void load(const double *data)
    {
        val = _mm_loadu_pd(data);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val = _mm_load_pd(data);
    }

    inline
    void store(double *data) const
    {
        _mm_storeu_pd(data + 0, val);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_pd(data + 0, val);
    }

    inline
    void store_nt(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_pd(data + 0, val);
    }

    inline
    void gather(const double *ptr, const int *offsets)
    {
        val = _mm_loadl_pd(val, ptr + offsets[0]);
        val = _mm_loadh_pd(val, ptr + offsets[1]);
    }

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        _mm_storel_pd(ptr + offsets[0], val);
        _mm_storeh_pd(ptr + offsets[1], val);
    }

    inline
    void blend(const mask_type& mask, const short_vec<double, 2>& other)
    {
#ifdef __SSE4_1__
        val  = _mm_blendv_pd(val,  other.val,  mask.val);
#else
        val = _mm_or_pd(
            _mm_and_pd(mask.val, other.val),
            _mm_andnot_pd(mask.val, val));
#endif
    }

private:
    __m128d val;
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
    const double *data1 = reinterpret_cast<const double *>(&vec.val);
    __os << "[" << data1[0] << ", " << data1[1]  << "]";
    return __os;
}

}

#endif

#endif
