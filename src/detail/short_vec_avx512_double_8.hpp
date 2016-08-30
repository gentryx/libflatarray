/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_DOUBLE_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_DOUBLE_8_HPP

#if LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX512F

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
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
class short_vec<double, 8>
{
public:
    static const int ARITY = 8;
    typedef __mmask8 mask_type;

    typedef short_vec_strategy::avx512f strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 8>& vec);

    inline
    short_vec(const double data = 0) :
        val1(_mm512_set1_pd(data))
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(const __m512d& val1) :
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
    void operator-=(const short_vec<double, 8>& other)
    {
        val1 = _mm512_sub_pd(val1, other.val1);
    }

    inline
    short_vec<double, 8> operator-(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm512_sub_pd(val1, other.val1));
    }

    inline
    void operator+=(const short_vec<double, 8>& other)
    {
        val1 = _mm512_add_pd(val1, other.val1);
    }

    inline
    short_vec<double, 8> operator+(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm512_add_pd(val1, other.val1));
    }

    inline
    void operator*=(const short_vec<double, 8>& other)
    {
        val1 = _mm512_mul_pd(val1, other.val1);
    }

    inline
    short_vec<double, 8> operator*(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm512_mul_pd(val1, other.val1));
    }

    inline
    void operator/=(const short_vec<double, 8>& other)
    {
        val1 = _mm512_div_pd(val1, other.val1);
    }

    inline
    short_vec<double, 8> operator/(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm512_div_pd(val1, other.val1));
    }

    inline
    mask_type operator<(const short_vec<double, 32>& other) const
    {
        return
            (_mm512_cmp_pd_mask(val1, other.val1, _CMP_LT_OS) <<  0);
    }

    inline
    mask_type operator<=(const short_vec<double, 32>& other) const
    {
        return
            (_mm512_cmp_pd_mask(val1, other.val1, _CMP_LE_OS) <<  0);
    }

    inline
    mask_type operator==(const short_vec<double, 32>& other) const
    {
        return
            (_mm512_cmp_pd_mask(val1, other.val1, _CMP_EQ_OQ) <<  0);
    }

    inline
    mask_type operator>(const short_vec<double, 32>& other) const
    {
        return
            (_mm512_cmp_pd_mask(val1, other.val1, _CMP_GT_OS) <<  0);
    }

    inline
    mask_type operator>=(const short_vec<double, 32>& other) const
    {
        return
            (_mm512_cmp_pd_mask(val1, other.val1, _CMP_GE_OS) <<  0);
    }

    inline
    short_vec<double, 8> sqrt() const
    {
        return short_vec<double, 8>(
            _mm512_sqrt_pd(val1));
    }

    inline
    void load(const double *data)
    {
        val1 = _mm512_loadu_pd(data);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        val1 = _mm512_load_pd(data);
    }

    inline
    void store(double *data) const
    {
        _mm512_storeu_pd(data, val1);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_store_pd(data, val1);
    }

    inline
    void store_nt(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_stream_pd(data, val1);
    }

    inline
    void gather(const double *ptr, const int *offsets)
    {
        __m256i indices;
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets));
        val1    = _mm512_i32gather_pd(indices, ptr, 8);
    }

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        __m256i indices;
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets));
        _mm512_i32scatter_pd(ptr, indices, val1, 8);
    }

private:
    __m512d val1;
};

inline
void operator<<(double *data, const short_vec<double, 8>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<double, 8> sqrt(const short_vec<double, 8>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 8>& vec)
{
    const double *data1 = reinterpret_cast<const double *>(&vec.val1);

    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << ", " << data1[4] << ", " << data1[5] << ", " << data1[6] << ", " << data1[7]
         << "]";
    return __os;
}

}

#endif

#endif
