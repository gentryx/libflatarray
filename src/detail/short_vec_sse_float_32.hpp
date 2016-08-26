/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_32_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE2) ||            \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE4_1)

#include <emmintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

template<typename CARGO, int ARITY>
class sqrt_reference;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<float, 32>
{
public:
    static const int ARITY = 32;

    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 32>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm_set1_ps(data)),
        val2(_mm_set1_ps(data)),
        val3(_mm_set1_ps(data)),
        val4(_mm_set1_ps(data)),
        val5(_mm_set1_ps(data)),
        val6(_mm_set1_ps(data)),
        val7(_mm_set1_ps(data)),
        val8(_mm_set1_ps(data))
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(const __m128& val1, const __m128& val2, const __m128& val3, const __m128& val4, const __m128& val5, const __m128& val6, const __m128& val7, const __m128& val8) :
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
    short_vec(const std::initializer_list<float>& il)
    {
        const float *ptr = static_cast<const float *>(&(*il.begin()));
        load(ptr);
    }
#endif

    inline
    short_vec(const sqrt_reference<float, 32>& other);

    inline
    bool any() const
    {
        __m128 buf1 = _mm_or_ps(
            _mm_or_ps(_mm_or_ps(val1, val2),
                      _mm_or_ps(val3, val4)),
            _mm_or_ps(_mm_or_ps(val5, val6),
                      _mm_or_ps(val7, val8)));
        __m128 buf2 = _mm_shuffle_ps(buf1, buf1, (3 << 2) | (2 << 0));
        buf1 = _mm_or_ps(buf1, buf2);
        buf2 = _mm_shuffle_ps(buf1, buf1, (1 << 0));
        return _mm_cvtss_f32(buf1) | _mm_cvtss_f32(buf2);
    }

    inline
    void operator-=(const short_vec<float, 32>& other)
    {
        val1 = _mm_sub_ps(val1, other.val1);
        val2 = _mm_sub_ps(val2, other.val2);
        val3 = _mm_sub_ps(val3, other.val3);
        val4 = _mm_sub_ps(val4, other.val4);
        val5 = _mm_sub_ps(val5, other.val5);
        val6 = _mm_sub_ps(val6, other.val6);
        val7 = _mm_sub_ps(val7, other.val7);
        val8 = _mm_sub_ps(val8, other.val8);
    }

    inline
    short_vec<float, 32> operator-(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_sub_ps(val1, other.val1),
            _mm_sub_ps(val2, other.val2),
            _mm_sub_ps(val3, other.val3),
            _mm_sub_ps(val4, other.val4),
            _mm_sub_ps(val5, other.val5),
            _mm_sub_ps(val6, other.val6),
            _mm_sub_ps(val7, other.val7),
            _mm_sub_ps(val8, other.val8));
    }

    inline
    void operator+=(const short_vec<float, 32>& other)
    {
        val1 = _mm_add_ps(val1, other.val1);
        val2 = _mm_add_ps(val2, other.val2);
        val3 = _mm_add_ps(val3, other.val3);
        val4 = _mm_add_ps(val4, other.val4);
        val5 = _mm_add_ps(val5, other.val5);
        val6 = _mm_add_ps(val6, other.val6);
        val7 = _mm_add_ps(val7, other.val7);
        val8 = _mm_add_ps(val8, other.val8);
    }

    inline
    short_vec<float, 32> operator+(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_add_ps(val1, other.val1),
            _mm_add_ps(val2, other.val2),
            _mm_add_ps(val3, other.val3),
            _mm_add_ps(val4, other.val4),
            _mm_add_ps(val5, other.val5),
            _mm_add_ps(val6, other.val6),
            _mm_add_ps(val7, other.val7),
            _mm_add_ps(val8, other.val8));
    }

    inline
    void operator*=(const short_vec<float, 32>& other)
    {
        val1 = _mm_mul_ps(val1, other.val1);
        val2 = _mm_mul_ps(val2, other.val2);
        val3 = _mm_mul_ps(val3, other.val3);
        val4 = _mm_mul_ps(val4, other.val4);
        val5 = _mm_mul_ps(val5, other.val5);
        val6 = _mm_mul_ps(val6, other.val6);
        val7 = _mm_mul_ps(val7, other.val7);
        val8 = _mm_mul_ps(val8, other.val8);
    }

    inline
    short_vec<float, 32> operator*(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_mul_ps(val1, other.val1),
            _mm_mul_ps(val2, other.val2),
            _mm_mul_ps(val3, other.val3),
            _mm_mul_ps(val4, other.val4),
            _mm_mul_ps(val5, other.val5),
            _mm_mul_ps(val6, other.val6),
            _mm_mul_ps(val7, other.val7),
            _mm_mul_ps(val8, other.val8));
    }

    inline
    void operator/=(const short_vec<float, 32>& other)
    {
        val1 = _mm_div_ps(val1, other.val1);
        val2 = _mm_div_ps(val2, other.val2);
        val3 = _mm_div_ps(val3, other.val3);
        val4 = _mm_div_ps(val4, other.val4);
        val5 = _mm_div_ps(val5, other.val5);
        val6 = _mm_div_ps(val6, other.val6);
        val7 = _mm_div_ps(val7, other.val7);
        val8 = _mm_div_ps(val8, other.val8);
    }

    inline
    void operator/=(const sqrt_reference<float, 32>& other);

    inline
    short_vec<float, 32> operator/(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_div_ps(val1, other.val1),
            _mm_div_ps(val2, other.val2),
            _mm_div_ps(val3, other.val3),
            _mm_div_ps(val4, other.val4),
            _mm_div_ps(val5, other.val5),
            _mm_div_ps(val6, other.val6),
            _mm_div_ps(val7, other.val7),
            _mm_div_ps(val8, other.val8));
    }

    inline
    short_vec<float, 32> operator/(const sqrt_reference<float, 32>& other) const;

    inline
    short_vec<float, 32> operator<(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_cmplt_ps(val1, other.val1),
            _mm_cmplt_ps(val2, other.val2),
            _mm_cmplt_ps(val3, other.val3),
            _mm_cmplt_ps(val4, other.val4),
            _mm_cmplt_ps(val5, other.val5),
            _mm_cmplt_ps(val6, other.val6),
            _mm_cmplt_ps(val7, other.val7),
            _mm_cmplt_ps(val8, other.val8));
    }

    inline
    short_vec<float, 32> operator<=(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_cmple_ps(val1, other.val1),
            _mm_cmple_ps(val2, other.val2),
            _mm_cmple_ps(val3, other.val3),
            _mm_cmple_ps(val4, other.val4),
            _mm_cmple_ps(val5, other.val5),
            _mm_cmple_ps(val6, other.val6),
            _mm_cmple_ps(val7, other.val7),
            _mm_cmple_ps(val8, other.val8));
    }

    inline
    short_vec<float, 32> operator==(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_cmpeq_ps(val1, other.val1),
            _mm_cmpeq_ps(val2, other.val2),
            _mm_cmpeq_ps(val3, other.val3),
            _mm_cmpeq_ps(val4, other.val4),
            _mm_cmpeq_ps(val5, other.val5),
            _mm_cmpeq_ps(val6, other.val6),
            _mm_cmpeq_ps(val7, other.val7),
            _mm_cmpeq_ps(val8, other.val8));
    }

    inline
    short_vec<float, 32> operator>(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_cmpgt_ps(val1, other.val1),
            _mm_cmpgt_ps(val2, other.val2),
            _mm_cmpgt_ps(val3, other.val3),
            _mm_cmpgt_ps(val4, other.val4),
            _mm_cmpgt_ps(val5, other.val5),
            _mm_cmpgt_ps(val6, other.val6),
            _mm_cmpgt_ps(val7, other.val7),
            _mm_cmpgt_ps(val8, other.val8));
    }

    inline
    short_vec<float, 32> operator>=(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_cmpge_ps(val1, other.val1),
            _mm_cmpge_ps(val2, other.val2),
            _mm_cmpge_ps(val3, other.val3),
            _mm_cmpge_ps(val4, other.val4),
            _mm_cmpge_ps(val5, other.val5),
            _mm_cmpge_ps(val6, other.val6),
            _mm_cmpge_ps(val7, other.val7),
            _mm_cmpge_ps(val8, other.val8));
    }

    inline
    short_vec<float, 32> sqrt() const
    {
        return short_vec<float, 32>(
            _mm_sqrt_ps(val1),
            _mm_sqrt_ps(val2),
            _mm_sqrt_ps(val3),
            _mm_sqrt_ps(val4),
            _mm_sqrt_ps(val5),
            _mm_sqrt_ps(val6),
            _mm_sqrt_ps(val7),
            _mm_sqrt_ps(val8));
    }

    inline
    void load(const float *data)
    {
        val1 = _mm_loadu_ps(data +  0);
        val2 = _mm_loadu_ps(data +  4);
        val3 = _mm_loadu_ps(data +  8);
        val4 = _mm_loadu_ps(data + 12);
        val5 = _mm_loadu_ps(data + 16);
        val6 = _mm_loadu_ps(data + 20);
        val7 = _mm_loadu_ps(data + 24);
        val8 = _mm_loadu_ps(data + 28);
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val1 = _mm_load_ps(data +  0);
        val2 = _mm_load_ps(data +  4);
        val3 = _mm_load_ps(data +  8);
        val4 = _mm_load_ps(data + 12);
        val5 = _mm_load_ps(data + 16);
        val6 = _mm_load_ps(data + 20);
        val7 = _mm_load_ps(data + 24);
        val8 = _mm_load_ps(data + 28);
    }

    inline
    void store(float *data) const
    {
        _mm_storeu_ps(data +  0, val1);
        _mm_storeu_ps(data +  4, val2);
        _mm_storeu_ps(data +  8, val3);
        _mm_storeu_ps(data + 12, val4);
        _mm_storeu_ps(data + 16, val5);
        _mm_storeu_ps(data + 20, val6);
        _mm_storeu_ps(data + 24, val7);
        _mm_storeu_ps(data + 28, val8);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_ps(data +  0, val1);
        _mm_store_ps(data +  4, val2);
        _mm_store_ps(data +  8, val3);
        _mm_store_ps(data + 12, val4);
        _mm_store_ps(data + 16, val5);
        _mm_store_ps(data + 20, val6);
        _mm_store_ps(data + 24, val7);
        _mm_store_ps(data + 28, val8);
    }

    inline
    void store_nt(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_ps(data +  0, val1);
        _mm_stream_ps(data +  4, val2);
        _mm_stream_ps(data +  8, val3);
        _mm_stream_ps(data + 12, val4);
        _mm_stream_ps(data + 16, val5);
        _mm_stream_ps(data + 20, val6);
        _mm_stream_ps(data + 24, val7);
        _mm_stream_ps(data + 28, val8);
    }

#ifdef __SSE4_1__
    inline
    void gather(const float *ptr, const int *offsets)
    {
        val1 = _mm_load_ss(ptr + offsets[0]);
        SHORTVEC_INSERT_PS(val1, ptr, offsets[ 1], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val1, ptr, offsets[ 2], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val1, ptr, offsets[ 3], _MM_MK_INSERTPS_NDX(0,3,0));

        val2 = _mm_load_ss(ptr + offsets[4]);
        SHORTVEC_INSERT_PS(val2, ptr, offsets[ 5], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val2, ptr, offsets[ 6], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val2, ptr, offsets[ 7], _MM_MK_INSERTPS_NDX(0,3,0));

        val3 = _mm_load_ss(ptr + offsets[8]);
        SHORTVEC_INSERT_PS(val3, ptr, offsets[ 9], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val3, ptr, offsets[10], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val3, ptr, offsets[11], _MM_MK_INSERTPS_NDX(0,3,0));

        val4 = _mm_load_ss(ptr + offsets[12]);
        SHORTVEC_INSERT_PS(val4, ptr, offsets[13], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val4, ptr, offsets[14], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val4, ptr, offsets[15], _MM_MK_INSERTPS_NDX(0,3,0));

        val5 = _mm_load_ss(ptr + offsets[16]);
        SHORTVEC_INSERT_PS(val1, ptr, offsets[17], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val1, ptr, offsets[18], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val1, ptr, offsets[19], _MM_MK_INSERTPS_NDX(0,3,0));

        val6 = _mm_load_ss(ptr + offsets[20]);
        SHORTVEC_INSERT_PS(val2, ptr, offsets[21], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val2, ptr, offsets[22], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val2, ptr, offsets[23], _MM_MK_INSERTPS_NDX(0,3,0));

        val7 = _mm_load_ss(ptr + offsets[24]);
        SHORTVEC_INSERT_PS(val3, ptr, offsets[25], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val3, ptr, offsets[26], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val3, ptr, offsets[27], _MM_MK_INSERTPS_NDX(0,3,0));

        val8 = _mm_load_ss(ptr + offsets[28]);
        SHORTVEC_INSERT_PS(val4, ptr, offsets[29], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val4, ptr, offsets[30], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val4, ptr, offsets[31], _MM_MK_INSERTPS_NDX(0,3,0));
    }

    inline
    void scatter(float *ptr, const int *offsets) const
    {
        ShortVecHelpers::ExtractResult r1, r2, r3, r4;
        r1.i = _mm_extract_ps(val1, 0);
        r2.i = _mm_extract_ps(val1, 1);
        r3.i = _mm_extract_ps(val1, 2);
        r4.i = _mm_extract_ps(val1, 3);
        ptr[offsets[0]] = r1.f;
        ptr[offsets[1]] = r2.f;
        ptr[offsets[2]] = r3.f;
        ptr[offsets[3]] = r4.f;

        r1.i = _mm_extract_ps(val2, 0);
        r2.i = _mm_extract_ps(val2, 1);
        r3.i = _mm_extract_ps(val2, 2);
        r4.i = _mm_extract_ps(val2, 3);
        ptr[offsets[4]] = r1.f;
        ptr[offsets[5]] = r2.f;
        ptr[offsets[6]] = r3.f;
        ptr[offsets[7]] = r4.f;

        r1.i = _mm_extract_ps(val3, 0);
        r2.i = _mm_extract_ps(val3, 1);
        r3.i = _mm_extract_ps(val3, 2);
        r4.i = _mm_extract_ps(val3, 3);
        ptr[offsets[ 8]] = r1.f;
        ptr[offsets[ 9]] = r2.f;
        ptr[offsets[10]] = r3.f;
        ptr[offsets[11]] = r4.f;

        r1.i = _mm_extract_ps(val4, 0);
        r2.i = _mm_extract_ps(val4, 1);
        r3.i = _mm_extract_ps(val4, 2);
        r4.i = _mm_extract_ps(val4, 3);
        ptr[offsets[12]] = r1.f;
        ptr[offsets[13]] = r2.f;
        ptr[offsets[14]] = r3.f;
        ptr[offsets[15]] = r4.f;

        r1.i = _mm_extract_ps(val5, 0);
        r2.i = _mm_extract_ps(val5, 1);
        r3.i = _mm_extract_ps(val5, 2);
        r4.i = _mm_extract_ps(val5, 3);
        ptr[offsets[16]] = r1.f;
        ptr[offsets[17]] = r2.f;
        ptr[offsets[18]] = r3.f;
        ptr[offsets[19]] = r4.f;

        r1.i = _mm_extract_ps(val6, 0);
        r2.i = _mm_extract_ps(val6, 1);
        r3.i = _mm_extract_ps(val6, 2);
        r4.i = _mm_extract_ps(val6, 3);
        ptr[offsets[20]] = r1.f;
        ptr[offsets[21]] = r2.f;
        ptr[offsets[22]] = r3.f;
        ptr[offsets[23]] = r4.f;

        r1.i = _mm_extract_ps(val7, 0);
        r2.i = _mm_extract_ps(val7, 1);
        r3.i = _mm_extract_ps(val7, 2);
        r4.i = _mm_extract_ps(val7, 3);
        ptr[offsets[24]] = r1.f;
        ptr[offsets[25]] = r2.f;
        ptr[offsets[26]] = r3.f;
        ptr[offsets[27]] = r4.f;

        r1.i = _mm_extract_ps(val8, 0);
        r2.i = _mm_extract_ps(val8, 1);
        r3.i = _mm_extract_ps(val8, 2);
        r4.i = _mm_extract_ps(val8, 3);
        ptr[offsets[28]] = r1.f;
        ptr[offsets[29]] = r2.f;
        ptr[offsets[30]] = r3.f;
        ptr[offsets[31]] = r4.f;
    }
#else
    inline
    void gather(const float *ptr, const int *offsets)
    {
        __m128 f1, f2, f3, f4;
        f1   = _mm_load_ss(ptr + offsets[0]);
        f2   = _mm_load_ss(ptr + offsets[2]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[1]);
        f4   = _mm_load_ss(ptr + offsets[3]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val1 = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[4]);
        f2   = _mm_load_ss(ptr + offsets[6]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[5]);
        f4   = _mm_load_ss(ptr + offsets[7]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val2 = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[ 8]);
        f2   = _mm_load_ss(ptr + offsets[10]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[ 9]);
        f4   = _mm_load_ss(ptr + offsets[11]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val3 = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[12]);
        f2   = _mm_load_ss(ptr + offsets[14]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[13]);
        f4   = _mm_load_ss(ptr + offsets[15]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val4 = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[16]);
        f2   = _mm_load_ss(ptr + offsets[17]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[18]);
        f4   = _mm_load_ss(ptr + offsets[19]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val5 = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[20]);
        f2   = _mm_load_ss(ptr + offsets[21]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[22]);
        f4   = _mm_load_ss(ptr + offsets[23]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val6 = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[24]);
        f2   = _mm_load_ss(ptr + offsets[25]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[26]);
        f4   = _mm_load_ss(ptr + offsets[27]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val7 = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[28]);
        f2   = _mm_load_ss(ptr + offsets[29]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[30]);
        f4   = _mm_load_ss(ptr + offsets[31]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val8 = _mm_unpacklo_ps(f1, f3);

    }

    inline
    void scatter(float *ptr, const int *offsets) const
    {
        __m128 tmp = val1;
        _mm_store_ss(ptr + offsets[0], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[1], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[2], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[3], tmp);

        tmp = val2;
        _mm_store_ss(ptr + offsets[4], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[5], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[6], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[7], tmp);

        tmp = val3;
        _mm_store_ss(ptr + offsets[8], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[9], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[10], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[11], tmp);

        tmp = val4;
        _mm_store_ss(ptr + offsets[12], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[13], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[14], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[15], tmp);

        tmp = val5;
        _mm_store_ss(ptr + offsets[16], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[17], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[18], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[19], tmp);

        tmp = val6;
        _mm_store_ss(ptr + offsets[20], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[21], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[22], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[23], tmp);

        tmp = val7;
        _mm_store_ss(ptr + offsets[24], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[25], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[26], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[27], tmp);

        tmp = val8;
        _mm_store_ss(ptr + offsets[28], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[29], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[30], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[31], tmp);
   }
#endif

private:
    __m128 val1;
    __m128 val2;
    __m128 val3;
    __m128 val4;
    __m128 val5;
    __m128 val6;
    __m128 val7;
    __m128 val8;
};

inline
void operator<<(float *data, const short_vec<float, 32>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<float, 32>
{
public:
    template<typename OTHER_CARGO, int OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<float, 32>& vec) :
        vec(vec)
    {}

private:
    short_vec<float, 32> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<float, 32>::short_vec(const sqrt_reference<float, 32>& other) :
    val1(_mm_sqrt_ps(other.vec.val1)),
    val2(_mm_sqrt_ps(other.vec.val2)),
    val3(_mm_sqrt_ps(other.vec.val3)),
    val4(_mm_sqrt_ps(other.vec.val4)),
    val5(_mm_sqrt_ps(other.vec.val5)),
    val6(_mm_sqrt_ps(other.vec.val6)),
    val7(_mm_sqrt_ps(other.vec.val7)),
    val8(_mm_sqrt_ps(other.vec.val8))
{}

inline
void short_vec<float, 32>::operator/=(const sqrt_reference<float, 32>& other)
{
    val1 = _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1));
    val2 = _mm_mul_ps(val2, _mm_rsqrt_ps(other.vec.val2));
    val3 = _mm_mul_ps(val3, _mm_rsqrt_ps(other.vec.val3));
    val4 = _mm_mul_ps(val4, _mm_rsqrt_ps(other.vec.val4));
    val5 = _mm_mul_ps(val5, _mm_rsqrt_ps(other.vec.val5));
    val6 = _mm_mul_ps(val6, _mm_rsqrt_ps(other.vec.val6));
    val7 = _mm_mul_ps(val7, _mm_rsqrt_ps(other.vec.val7));
    val8 = _mm_mul_ps(val8, _mm_rsqrt_ps(other.vec.val8));
}

inline
short_vec<float, 32> short_vec<float, 32>::operator/(const sqrt_reference<float, 32>& other) const
{
    return short_vec<float, 32>(
        _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1)),
        _mm_mul_ps(val2, _mm_rsqrt_ps(other.vec.val2)),
        _mm_mul_ps(val3, _mm_rsqrt_ps(other.vec.val3)),
        _mm_mul_ps(val4, _mm_rsqrt_ps(other.vec.val4)),
        _mm_mul_ps(val5, _mm_rsqrt_ps(other.vec.val5)),
        _mm_mul_ps(val6, _mm_rsqrt_ps(other.vec.val6)),
        _mm_mul_ps(val7, _mm_rsqrt_ps(other.vec.val7)),
        _mm_mul_ps(val8, _mm_rsqrt_ps(other.vec.val8)));
}

inline
sqrt_reference<float, 32> sqrt(const short_vec<float, 32>& vec)
{
    return sqrt_reference<float, 32>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 32>& vec)
{
    const float *data1 = reinterpret_cast<const float *>(&vec.val1);
    const float *data2 = reinterpret_cast<const float *>(&vec.val2);
    const float *data3 = reinterpret_cast<const float *>(&vec.val3);
    const float *data4 = reinterpret_cast<const float *>(&vec.val4);
    __os << "["
         << data1[0] << ", " << data1[1]  << ", " << data1[2] << ", " << data1[3] << ", "
         << data2[0] << ", " << data2[1]  << ", " << data2[2] << ", " << data2[3] << ", "
         << data3[0] << ", " << data3[1]  << ", " << data3[2] << ", " << data3[3] << ", "
         << data4[0] << ", " << data4[1]  << ", " << data4[2] << ", " << data4[3] << ", "
         << data5[0] << ", " << data5[1]  << ", " << data5[2] << ", " << data5[3] << ", "
         << data6[0] << ", " << data6[1]  << ", " << data6[2] << ", " << data6[3] << ", "
         << data7[0] << ", " << data7[1]  << ", " << data7[2] << ", " << data7[3] << ", "
         << data8[0] << ", " << data8[1]  << ", " << data8[2] << ", " << data8[3] << "]";
    return __os;
}

}

#endif

#endif
