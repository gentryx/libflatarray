/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_DOUBLE_4_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_DOUBLE_4_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX) ||     \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX2) ||    \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX512F)

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
class short_vec<double, 4>
{
public:
    static const int ARITY = 4;
    typedef short_vec<double, 4> mask_type;
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
        // merge both 128-bit lanes of AVX register:
        __m128d buf1 = _mm_or_pd(
            _mm256_extractf128_pd(val1, 0),
            _mm256_extractf128_pd(val1, 1));
        // shuffle upper 64-bit half down to first 64 bits so we can
        // "or" both together:
        __m128d buf2 = _mm_shuffle_pd(buf1, buf1, 1 << 0);
        buf2 = _mm_or_pd(buf1, buf2);
        // another shuffle to extract the upper 64-bit half:
        buf1 = _mm_shuffle_pd(buf2, buf2, 1 << 0);
        return _mm_cvtsd_f64(buf1) || _mm_cvtsd_f64(buf2);
    }

    inline
    double get(int i) const
    {
        __m128d buf;
        if (i < 2) {
            buf = _mm256_extractf128_pd(val1, 0);
        } else {
            buf = _mm256_extractf128_pd(val1, 1);
        }

        i &= 1;

        if (i == 0) {
            return _mm_cvtsd_f64(buf);
        }

        buf = _mm_shuffle_pd(buf, buf, 1);
        return _mm_cvtsd_f64(buf);
    }

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
        return short_vec<double, 4>(_mm256_add_pd(val1, other.val1));
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
    short_vec<double, 4> operator<(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm256_cmp_pd(val1, other.val1, _CMP_LT_OS));
    }

    inline
    short_vec<double, 4> operator<=(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm256_cmp_pd(val1, other.val1, _CMP_LE_OS));
    }

    inline
    short_vec<double, 4> operator==(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm256_cmp_pd(val1, other.val1, _CMP_EQ_OQ));
    }

    inline
    short_vec<double, 4> operator>(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm256_cmp_pd(val1, other.val1, _CMP_GT_OS));
    }

    inline
    short_vec<double, 4> operator>=(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            _mm256_cmp_pd(val1, other.val1, _CMP_GE_OS));
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
    void gather(const double *ptr, const int *offsets)
    {
        __m128i indices;
        indices = _mm_loadu_si128(reinterpret_cast<const __m128i *>(offsets));
        val1    = _mm256_i32gather_pd(ptr, indices, 8);
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
    }

    inline
    void blend(const mask_type& mask, const short_vec<double, 4>& other)
    {
        val1  = _mm256_blendv_pd(val1,  other.val1,  mask.val1);
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
