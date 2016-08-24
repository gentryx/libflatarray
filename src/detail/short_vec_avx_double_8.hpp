/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_DOUBLE_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_DOUBLE_8_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX) || (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX2)

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

    typedef short_vec_strategy::avx strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 8>& vec);

    inline
    short_vec(const double data = 0) :
        val1(_mm256_broadcast_sd(&data)),
        val2(_mm256_broadcast_sd(&data))
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(const __m256d& val1, const __m256d& val2) :
        val1(val1),
        val2(val2)
    {}

#ifdef LIBFLATARRAY_WITH_CPP14
    inline
    short_vec(const std::initializer_list<double>& il)
    {
        const double *ptr = reinterpret_cast<const double *>(&(*il.begin()));
        load(ptr);
    }
#endif

    inline
    void operator-=(const short_vec<double, 8>& other)
    {
        val1 = _mm256_sub_pd(val1, other.val1);
        val2 = _mm256_sub_pd(val2, other.val2);
    }

    inline
    short_vec<double, 8> operator-(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm256_sub_pd(val1, other.val1),
            _mm256_sub_pd(val2, other.val2));
    }

    inline
    void operator+=(const short_vec<double, 8>& other)
    {
        val1 = _mm256_add_pd(val1, other.val1);
        val2 = _mm256_add_pd(val2, other.val2);
    }

    inline
    short_vec<double, 8> operator+(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm256_add_pd(val1, other.val1),
            _mm256_add_pd(val2, other.val2));
    }

    inline
    void operator*=(const short_vec<double, 8>& other)
    {
        val1 = _mm256_mul_pd(val1, other.val1);
        val2 = _mm256_mul_pd(val2, other.val2);
    }

    inline
    short_vec<double, 8> operator*(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm256_mul_pd(val1, other.val1),
            _mm256_mul_pd(val2, other.val2));
    }

    inline
    void operator/=(const short_vec<double, 8>& other)
    {
        val1 = _mm256_div_pd(val1, other.val1);
        val2 = _mm256_div_pd(val2, other.val2);
    }

    inline
    short_vec<double, 8> operator/(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm256_div_pd(val1, other.val1),
            _mm256_div_pd(val2, other.val2));
    }

    inline
    short_vec<double, 8> operator<(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm256_cmp_pd(val1, other.val1, _CMP_LT_OS),
            _mm256_cmp_pd(val2, other.val2, _CMP_LT_OS));
    }

    inline
    short_vec<double, 8> operator<=(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm256_cmp_pd(val1, other.val1, _CMP_LE_OS),
            _mm256_cmp_pd(val2, other.val2, _CMP_LE_OS));
    }

    inline
    short_vec<double, 8> operator==(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm256_cmp_pd(val1, other.val1, _CMP_EQ_OQ),
            _mm256_cmp_pd(val2, other.val2, _CMP_EQ_OQ));
    }

    inline
    short_vec<double, 8> operator>(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm256_cmp_pd(val1, other.val1, _CMP_GT_OS),
            _mm256_cmp_pd(val2, other.val2, _CMP_GT_OS));
    }

    inline
    short_vec<double, 8> operator>=(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            _mm256_cmp_pd(val1, other.val1, _CMP_GE_OS),
            _mm256_cmp_pd(val2, other.val2, _CMP_GE_OS));
    }

    inline
    short_vec<double, 8> sqrt() const
    {
        return short_vec<double, 8>(
            _mm256_sqrt_pd(val1),
            _mm256_sqrt_pd(val2));
    }

    inline
    void load(const double *data)
    {
        val1 = _mm256_loadu_pd(data + 0);
        val2 = _mm256_loadu_pd(data + 4);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        val1 = _mm256_load_pd(data + 0);
        val2 = _mm256_load_pd(data + 4);
    }

    inline
    void store(double *data) const
    {
        _mm256_storeu_pd(data +  0, val1);
        _mm256_storeu_pd(data +  4, val2);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_store_pd(data + 0, val1);
        _mm256_store_pd(data + 4, val2);
    }

    inline
    void store_nt(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_stream_pd(data + 0, val1);
        _mm256_stream_pd(data + 4, val2);
    }

#ifdef __AVX2__
    inline
    void gather(const double *ptr, const int *offsets)
    {
        __m128i indices;
        indices = _mm_loadu_si128(reinterpret_cast<const __m128i *>(offsets));
        val1    = _mm256_i32gather_pd(ptr, indices, 8);
        indices = _mm_loadu_si128(reinterpret_cast<const __m128i *>(offsets + 4));
        val2    = _mm256_i32gather_pd(ptr, indices, 8);
    }
#else
    inline
    void gather(const double *ptr, const int *offsets)
    {
        val1 = _mm256_set_pd(
            *(ptr + offsets[3]),
            *(ptr + offsets[2]),
            *(ptr + offsets[1]),
            *(ptr + offsets[0]));

        val2 = _mm256_set_pd(
            *(ptr + offsets[7]),
            *(ptr + offsets[6]),
            *(ptr + offsets[5]),
            *(ptr + offsets[4]));
    }
#endif

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        __m128d tmp;
        tmp = _mm256_extractf128_pd(val1, 0);
        _mm_storel_pd(ptr + offsets[0], tmp);
        _mm_storeh_pd(ptr + offsets[1], tmp);
        tmp = _mm256_extractf128_pd(val1, 1);
        _mm_storel_pd(ptr + offsets[2], tmp);
        _mm_storeh_pd(ptr + offsets[3], tmp);
        tmp = _mm256_extractf128_pd(val2, 0);
        _mm_storel_pd(ptr + offsets[4], tmp);
        _mm_storeh_pd(ptr + offsets[5], tmp);
        tmp = _mm256_extractf128_pd(val2, 1);
        _mm_storel_pd(ptr + offsets[6], tmp);
        _mm_storeh_pd(ptr + offsets[7], tmp);
    }

private:
    __m256d val1;
    __m256d val2;
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
    const double *data2 = reinterpret_cast<const double *>(&vec.val2);

    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << ", " << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
         << "]";
    return __os;
}

}

#endif

#endif
