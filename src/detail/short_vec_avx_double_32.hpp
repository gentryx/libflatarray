/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_DOUBLE_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_DOUBLE_32_HPP

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
class short_vec<double, 32>
{
public:
    static const int ARITY = 32;

    typedef short_vec_strategy::avx strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 32>& vec);

    inline
    short_vec(const double data = 0) :
        val1(_mm256_broadcast_sd(&data)),
        val2(_mm256_broadcast_sd(&data)),
        val3(_mm256_broadcast_sd(&data)),
        val4(_mm256_broadcast_sd(&data)),
        val5(_mm256_broadcast_sd(&data)),
        val6(_mm256_broadcast_sd(&data)),
        val7(_mm256_broadcast_sd(&data)),
        val8(_mm256_broadcast_sd(&data))
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(
        const __m256d& val1,
        const __m256d& val2,
        const __m256d& val3,
        const __m256d& val4,
        const __m256d& val5,
        const __m256d& val6,
        const __m256d& val7,
        const __m256d& val8) :
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
        __m256d buf0 = _mm256_or_pd(
            _mm256_or_pd(
                _mm256_or_pd(val1, val2),
                _mm256_or_pd(val3, val4)),
            _mm256_or_pd(
                _mm256_or_pd(val1, val2),
                _mm256_or_pd(val3, val4)));

        // merge both 128-bit lanes of AVX register:
        __m128d buf1 = _mm_or_pd(
            _mm256_extractf128_pd(buf0, 0),
            _mm256_extractf128_pd(buf0, 1));
        // shuffle upper 64-bit half down to first 64 bits so we can
        // "or" both together:
        __m128d buf2 = _mm_shuffle_pd(buf1, buf1, 1 << 0);
        buf2 = _mm_or_pd(buf1, buf2);
        // another shuffle to extract the upper 64-bit half:
        buf1 = _mm_shuffle_pd(buf2, buf2, 1 << 0);
        return _mm_cvtsd_f64(buf1) || _mm_cvtsd_f64(buf2);
    }

    inline
    void operator-=(const short_vec<double, 32>& other)
    {
        val1 = _mm256_sub_pd(val1, other.val1);
        val2 = _mm256_sub_pd(val2, other.val2);
        val3 = _mm256_sub_pd(val3, other.val3);
        val4 = _mm256_sub_pd(val4, other.val4);
        val5 = _mm256_sub_pd(val5, other.val5);
        val6 = _mm256_sub_pd(val6, other.val6);
        val7 = _mm256_sub_pd(val7, other.val7);
        val8 = _mm256_sub_pd(val8, other.val8);
    }

    inline
    short_vec<double, 32> operator-(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm256_sub_pd(val1, other.val1),
            _mm256_sub_pd(val2, other.val2),
            _mm256_sub_pd(val3, other.val3),
            _mm256_sub_pd(val4, other.val4),
            _mm256_sub_pd(val5, other.val5),
            _mm256_sub_pd(val6, other.val6),
            _mm256_sub_pd(val7, other.val7),
            _mm256_sub_pd(val8, other.val8));
    }

    inline
    void operator+=(const short_vec<double, 32>& other)
    {
        val1 = _mm256_add_pd(val1, other.val1);
        val2 = _mm256_add_pd(val2, other.val2);
        val3 = _mm256_add_pd(val3, other.val3);
        val4 = _mm256_add_pd(val4, other.val4);
        val5 = _mm256_add_pd(val5, other.val5);
        val6 = _mm256_add_pd(val6, other.val6);
        val7 = _mm256_add_pd(val7, other.val7);
        val8 = _mm256_add_pd(val8, other.val8);
    }

    inline
    short_vec<double, 32> operator+(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm256_add_pd(val1, other.val1),
            _mm256_add_pd(val2, other.val2),
            _mm256_add_pd(val3, other.val3),
            _mm256_add_pd(val4, other.val4),
            _mm256_add_pd(val5, other.val5),
            _mm256_add_pd(val6, other.val6),
            _mm256_add_pd(val7, other.val7),
            _mm256_add_pd(val8, other.val8));
    }

    inline
    void operator*=(const short_vec<double, 32>& other)
    {
        val1 = _mm256_mul_pd(val1, other.val1);
        val2 = _mm256_mul_pd(val2, other.val2);
        val3 = _mm256_mul_pd(val3, other.val3);
        val4 = _mm256_mul_pd(val4, other.val4);
        val5 = _mm256_mul_pd(val5, other.val5);
        val6 = _mm256_mul_pd(val6, other.val6);
        val7 = _mm256_mul_pd(val7, other.val7);
        val8 = _mm256_mul_pd(val8, other.val8);
    }

    inline
    short_vec<double, 32> operator*(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm256_mul_pd(val1, other.val1),
            _mm256_mul_pd(val2, other.val2),
            _mm256_mul_pd(val3, other.val3),
            _mm256_mul_pd(val4, other.val4),
            _mm256_mul_pd(val5, other.val5),
            _mm256_mul_pd(val6, other.val6),
            _mm256_mul_pd(val7, other.val7),
            _mm256_mul_pd(val8, other.val8));
    }

    inline
    void operator/=(const short_vec<double, 32>& other)
    {
        val1 = _mm256_div_pd(val1, other.val1);
        val2 = _mm256_div_pd(val2, other.val2);
        val3 = _mm256_div_pd(val3, other.val3);
        val4 = _mm256_div_pd(val4, other.val4);
        val5 = _mm256_div_pd(val5, other.val5);
        val6 = _mm256_div_pd(val6, other.val6);
        val7 = _mm256_div_pd(val7, other.val7);
        val8 = _mm256_div_pd(val8, other.val8);
    }

    inline
    short_vec<double, 32> operator/(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm256_div_pd(val1, other.val1),
            _mm256_div_pd(val2, other.val2),
            _mm256_div_pd(val3, other.val3),
            _mm256_div_pd(val4, other.val4),
            _mm256_div_pd(val5, other.val5),
            _mm256_div_pd(val6, other.val6),
            _mm256_div_pd(val7, other.val7),
            _mm256_div_pd(val8, other.val8));
    }

    inline
    short_vec<double, 32> operator<(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm256_cmp_pd(val1, other.val1, _CMP_LT_OS),
            _mm256_cmp_pd(val2, other.val2, _CMP_LT_OS),
            _mm256_cmp_pd(val3, other.val3, _CMP_LT_OS),
            _mm256_cmp_pd(val4, other.val4, _CMP_LT_OS),
            _mm256_cmp_pd(val5, other.val5, _CMP_LT_OS),
            _mm256_cmp_pd(val6, other.val6, _CMP_LT_OS),
            _mm256_cmp_pd(val7, other.val7, _CMP_LT_OS),
            _mm256_cmp_pd(val8, other.val8, _CMP_LT_OS));
    }

    inline
    short_vec<double, 32> operator<=(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm256_cmp_pd(val1, other.val1, _CMP_LE_OS),
            _mm256_cmp_pd(val2, other.val2, _CMP_LE_OS),
            _mm256_cmp_pd(val3, other.val3, _CMP_LE_OS),
            _mm256_cmp_pd(val4, other.val4, _CMP_LE_OS),
            _mm256_cmp_pd(val5, other.val5, _CMP_LE_OS),
            _mm256_cmp_pd(val6, other.val6, _CMP_LE_OS),
            _mm256_cmp_pd(val7, other.val7, _CMP_LE_OS),
            _mm256_cmp_pd(val8, other.val8, _CMP_LE_OS));
    }

    inline
    short_vec<double, 32> operator==(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm256_cmp_pd(val1, other.val1, _CMP_EQ_OQ),
            _mm256_cmp_pd(val2, other.val2, _CMP_EQ_OQ),
            _mm256_cmp_pd(val3, other.val3, _CMP_EQ_OQ),
            _mm256_cmp_pd(val4, other.val4, _CMP_EQ_OQ),
            _mm256_cmp_pd(val5, other.val5, _CMP_EQ_OQ),
            _mm256_cmp_pd(val6, other.val6, _CMP_EQ_OQ),
            _mm256_cmp_pd(val7, other.val7, _CMP_EQ_OQ),
            _mm256_cmp_pd(val8, other.val8, _CMP_EQ_OQ));
    }

    inline
    short_vec<double, 32> operator>(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm256_cmp_pd(val1, other.val1, _CMP_GT_OS),
            _mm256_cmp_pd(val2, other.val2, _CMP_GT_OS),
            _mm256_cmp_pd(val3, other.val3, _CMP_GT_OS),
            _mm256_cmp_pd(val4, other.val4, _CMP_GT_OS),
            _mm256_cmp_pd(val5, other.val5, _CMP_GT_OS),
            _mm256_cmp_pd(val6, other.val6, _CMP_GT_OS),
            _mm256_cmp_pd(val7, other.val7, _CMP_GT_OS),
            _mm256_cmp_pd(val8, other.val8, _CMP_GT_OS));
    }

    inline
    short_vec<double, 32> operator>=(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm256_cmp_pd(val1, other.val1, _CMP_GE_OS),
            _mm256_cmp_pd(val2, other.val2, _CMP_GE_OS),
            _mm256_cmp_pd(val3, other.val3, _CMP_GE_OS),
            _mm256_cmp_pd(val4, other.val4, _CMP_GE_OS),
            _mm256_cmp_pd(val5, other.val5, _CMP_GE_OS),
            _mm256_cmp_pd(val6, other.val6, _CMP_GE_OS),
            _mm256_cmp_pd(val7, other.val7, _CMP_GE_OS),
            _mm256_cmp_pd(val8, other.val8, _CMP_GE_OS));
    }

    inline
    short_vec<double, 32> sqrt() const
    {
        return short_vec<double, 32>(
            _mm256_sqrt_pd(val1),
            _mm256_sqrt_pd(val2),
            _mm256_sqrt_pd(val3),
            _mm256_sqrt_pd(val4),
            _mm256_sqrt_pd(val5),
            _mm256_sqrt_pd(val6),
            _mm256_sqrt_pd(val7),
            _mm256_sqrt_pd(val8));
    }

    inline
    void load(const double *data)
    {
        val1 = _mm256_loadu_pd(data +  0);
        val2 = _mm256_loadu_pd(data +  4);
        val3 = _mm256_loadu_pd(data +  8);
        val4 = _mm256_loadu_pd(data + 12);
        val5 = _mm256_loadu_pd(data + 16);
        val6 = _mm256_loadu_pd(data + 20);
        val7 = _mm256_loadu_pd(data + 24);
        val8 = _mm256_loadu_pd(data + 28);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        val1 = _mm256_load_pd(data +  0);
        val2 = _mm256_load_pd(data +  4);
        val3 = _mm256_load_pd(data +  8);
        val4 = _mm256_load_pd(data + 12);
        val5 = _mm256_load_pd(data + 16);
        val6 = _mm256_load_pd(data + 20);
        val7 = _mm256_load_pd(data + 24);
        val8 = _mm256_load_pd(data + 28);
    }

    inline
    void store(double *data) const
    {
        _mm256_storeu_pd(data +  0, val1);
        _mm256_storeu_pd(data +  4, val2);
        _mm256_storeu_pd(data +  8, val3);
        _mm256_storeu_pd(data + 12, val4);
        _mm256_storeu_pd(data + 16, val5);
        _mm256_storeu_pd(data + 20, val6);
        _mm256_storeu_pd(data + 24, val7);
        _mm256_storeu_pd(data + 28, val8);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_store_pd(data +  0, val1);
        _mm256_store_pd(data +  4, val2);
        _mm256_store_pd(data +  8, val3);
        _mm256_store_pd(data + 12, val4);
        _mm256_store_pd(data + 16, val5);
        _mm256_store_pd(data + 20, val6);
        _mm256_store_pd(data + 24, val7);
        _mm256_store_pd(data + 28, val8);
    }

    inline
    void store_nt(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_stream_pd(data +  0, val1);
        _mm256_stream_pd(data +  4, val2);
        _mm256_stream_pd(data +  8, val3);
        _mm256_stream_pd(data + 12, val4);
        _mm256_stream_pd(data + 16, val5);
        _mm256_stream_pd(data + 20, val6);
        _mm256_stream_pd(data + 24, val7);
        _mm256_stream_pd(data + 28, val8);
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
        indices = _mm_loadu_si128(reinterpret_cast<const __m128i *>(offsets + 8));
        val3    = _mm256_i32gather_pd(ptr, indices, 8);
        indices = _mm_loadu_si128(reinterpret_cast<const __m128i *>(offsets + 12));
        val4    = _mm256_i32gather_pd(ptr, indices, 8);
        indices = _mm_loadu_si128(reinterpret_cast<const __m128i *>(offsets + 16));
        val5    = _mm256_i32gather_pd(ptr, indices, 8);
        indices = _mm_loadu_si128(reinterpret_cast<const __m128i *>(offsets + 20));
        val6    = _mm256_i32gather_pd(ptr, indices, 8);
        indices = _mm_loadu_si128(reinterpret_cast<const __m128i *>(offsets + 24));
        val7    = _mm256_i32gather_pd(ptr, indices, 8);
        indices = _mm_loadu_si128(reinterpret_cast<const __m128i *>(offsets + 28));
        val8    = _mm256_i32gather_pd(ptr, indices, 8);
    }
#else
    inline
    void gather(const double *ptr, const int *offsets)
    {
        val1 = _mm256_set_pd(
            *(ptr + offsets[ 3]),
            *(ptr + offsets[ 2]),
            *(ptr + offsets[ 1]),
            *(ptr + offsets[ 0]));

        val2 = _mm256_set_pd(
            *(ptr + offsets[ 7]),
            *(ptr + offsets[ 6]),
            *(ptr + offsets[ 5]),
            *(ptr + offsets[ 4]));

        val3 = _mm256_set_pd(
            *(ptr + offsets[11]),
            *(ptr + offsets[10]),
            *(ptr + offsets[ 9]),
            *(ptr + offsets[ 8]));

        val4 = _mm256_set_pd(
            *(ptr + offsets[15]),
            *(ptr + offsets[14]),
            *(ptr + offsets[13]),
            *(ptr + offsets[12]));

        val5 = _mm256_set_pd(
            *(ptr + offsets[19]),
            *(ptr + offsets[18]),
            *(ptr + offsets[17]),
            *(ptr + offsets[16]));

        val6 = _mm256_set_pd(
            *(ptr + offsets[23]),
            *(ptr + offsets[22]),
            *(ptr + offsets[21]),
            *(ptr + offsets[20]));

        val7 = _mm256_set_pd(
            *(ptr + offsets[27]),
            *(ptr + offsets[26]),
            *(ptr + offsets[25]),
            *(ptr + offsets[24]));

        val8 = _mm256_set_pd(
            *(ptr + offsets[31]),
            *(ptr + offsets[30]),
            *(ptr + offsets[29]),
            *(ptr + offsets[28]));
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

        tmp = _mm256_extractf128_pd(val3, 0);
        _mm_storel_pd(ptr + offsets[8], tmp);
        _mm_storeh_pd(ptr + offsets[9], tmp);

        tmp = _mm256_extractf128_pd(val3, 1);
        _mm_storel_pd(ptr + offsets[10], tmp);
        _mm_storeh_pd(ptr + offsets[11], tmp);

        tmp = _mm256_extractf128_pd(val4, 0);
        _mm_storel_pd(ptr + offsets[12], tmp);
        _mm_storeh_pd(ptr + offsets[13], tmp);

        tmp = _mm256_extractf128_pd(val4, 1);
        _mm_storel_pd(ptr + offsets[14], tmp);
        _mm_storeh_pd(ptr + offsets[15], tmp);

        tmp = _mm256_extractf128_pd(val5, 0);
        _mm_storel_pd(ptr + offsets[16], tmp);
        _mm_storeh_pd(ptr + offsets[17], tmp);

        tmp = _mm256_extractf128_pd(val5, 1);
        _mm_storel_pd(ptr + offsets[18], tmp);
        _mm_storeh_pd(ptr + offsets[19], tmp);

        tmp = _mm256_extractf128_pd(val6, 0);
        _mm_storel_pd(ptr + offsets[20], tmp);
        _mm_storeh_pd(ptr + offsets[21], tmp);

        tmp = _mm256_extractf128_pd(val6, 1);
        _mm_storel_pd(ptr + offsets[22], tmp);
        _mm_storeh_pd(ptr + offsets[23], tmp);

        tmp = _mm256_extractf128_pd(val7, 0);
        _mm_storel_pd(ptr + offsets[24], tmp);
        _mm_storeh_pd(ptr + offsets[25], tmp);

        tmp = _mm256_extractf128_pd(val7, 1);
        _mm_storel_pd(ptr + offsets[26], tmp);
        _mm_storeh_pd(ptr + offsets[27], tmp);

        tmp = _mm256_extractf128_pd(val8, 0);
        _mm_storel_pd(ptr + offsets[28], tmp);
        _mm_storeh_pd(ptr + offsets[29], tmp);

        tmp = _mm256_extractf128_pd(val8, 1);
        _mm_storel_pd(ptr + offsets[30], tmp);
        _mm_storeh_pd(ptr + offsets[31], tmp);
    }

private:
    __m256d val1;
    __m256d val2;
    __m256d val3;
    __m256d val4;
    __m256d val5;
    __m256d val6;
    __m256d val7;
    __m256d val8;
};

inline
void operator<<(double *data, const short_vec<double, 32>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

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
    const double *data1 = reinterpret_cast<const double *>(&vec.val1);
    const double *data2 = reinterpret_cast<const double *>(&vec.val2);
    const double *data3 = reinterpret_cast<const double *>(&vec.val3);
    const double *data4 = reinterpret_cast<const double *>(&vec.val4);
    const double *data5 = reinterpret_cast<const double *>(&vec.val5);
    const double *data6 = reinterpret_cast<const double *>(&vec.val6);
    const double *data7 = reinterpret_cast<const double *>(&vec.val7);
    const double *data8 = reinterpret_cast<const double *>(&vec.val8);

    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << ", " << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
         << ", " << data3[0] << ", " << data3[1] << ", " << data3[2] << ", " << data3[3]
         << ", " << data4[0] << ", " << data4[1] << ", " << data4[2] << ", " << data4[3]
         << ", " << data5[0] << ", " << data5[1] << ", " << data5[2] << ", " << data5[3]
         << ", " << data6[0] << ", " << data6[1] << ", " << data6[2] << ", " << data6[3]
         << ", " << data7[0] << ", " << data7[1] << ", " << data7[2] << ", " << data7[3]
         << ", " << data8[0] << ", " << data8[1] << ", " << data8[2] << ", " << data8[3]
         << "]";
    return __os;
}

}

#endif

#endif
