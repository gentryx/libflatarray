/**
 * Copyright 2014-2017 Andreas Schäfer
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

#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>
#include <libflatarray/short_vec_base.hpp>

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <emmintrin.h>
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibFlatArray {

template<typename CARGO, std::size_t ARITY>
class short_vec;

template<typename CARGO, std::size_t ARITY>
class sqrt_reference;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<float, 32> : public short_vec_base<float, 32>
{
public:
    static const std::size_t ARITY = 32;
    typedef short_vec<float, 32> mask_type;
    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 32>& vec);

    inline
    short_vec(const float data = 0) :
        val{_mm_set1_ps(data),
            _mm_set1_ps(data),
            _mm_set1_ps(data),
            _mm_set1_ps(data),
            _mm_set1_ps(data),
            _mm_set1_ps(data),
            _mm_set1_ps(data),
            _mm_set1_ps(data)}
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(
        const __m128& val1,
        const __m128& val2,
        const __m128& val3,
        const __m128& val4,
        const __m128& val5,
        const __m128& val6,
        const __m128& val7,
        const __m128& val8) :
        val{val1,
            val2,
            val3,
            val4,
            val5,
            val6,
            val7,
            val8}
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
            _mm_or_ps(_mm_or_ps(val[ 0], val[ 1]),
                      _mm_or_ps(val[ 2], val[ 3])),
            _mm_or_ps(_mm_or_ps(val[ 4], val[ 5]),
                      _mm_or_ps(val[ 6], val[ 7])));

#ifdef __SSE4_1__
        return (0 == _mm_testz_si128(
                    _mm_castps_si128(buf1),
                    _mm_castps_si128(buf1)));
#else
        __m128 buf2 = _mm_shuffle_ps(buf1, buf1, (3 << 2) | (2 << 0));
        buf1 = _mm_or_ps(buf1, buf2);
        buf2 = _mm_shuffle_ps(buf1, buf1, (1 << 0));
        return _mm_cvtss_f32(buf1) || _mm_cvtss_f32(buf2);
#endif
    }

    inline
    float operator[](int i) const
    {
        __m128 buf = val[i >> 2];
        i &= 3;

        if (i == 3) {
            return _mm_cvtss_f32(_mm_shuffle_ps(buf, buf, 3));
        }
        if (i == 2) {
            return _mm_cvtss_f32(_mm_shuffle_ps(buf, buf, 2));
        }
        if (i == 1) {
            return _mm_cvtss_f32(_mm_shuffle_ps(buf, buf, 1));
        }

        return _mm_cvtss_f32(buf);
    }

    inline
    void operator-=(const short_vec<float, 32>& other)
    {
        val[ 0] = _mm_sub_ps(val[ 0], other.val[ 0]);
        val[ 1] = _mm_sub_ps(val[ 1], other.val[ 1]);
        val[ 2] = _mm_sub_ps(val[ 2], other.val[ 2]);
        val[ 3] = _mm_sub_ps(val[ 3], other.val[ 3]);
        val[ 4] = _mm_sub_ps(val[ 4], other.val[ 4]);
        val[ 5] = _mm_sub_ps(val[ 5], other.val[ 5]);
        val[ 6] = _mm_sub_ps(val[ 6], other.val[ 6]);
        val[ 7] = _mm_sub_ps(val[ 7], other.val[ 7]);
    }

    inline
    short_vec<float, 32> operator-(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_sub_ps(val[ 0], other.val[ 0]),
            _mm_sub_ps(val[ 1], other.val[ 1]),
            _mm_sub_ps(val[ 2], other.val[ 2]),
            _mm_sub_ps(val[ 3], other.val[ 3]),
            _mm_sub_ps(val[ 4], other.val[ 4]),
            _mm_sub_ps(val[ 5], other.val[ 5]),
            _mm_sub_ps(val[ 6], other.val[ 6]),
            _mm_sub_ps(val[ 7], other.val[ 7]));
    }

    inline
    void operator+=(const short_vec<float, 32>& other)
    {
        val[ 0] = _mm_add_ps(val[ 0], other.val[ 0]);
        val[ 1] = _mm_add_ps(val[ 1], other.val[ 1]);
        val[ 2] = _mm_add_ps(val[ 2], other.val[ 2]);
        val[ 3] = _mm_add_ps(val[ 3], other.val[ 3]);
        val[ 4] = _mm_add_ps(val[ 4], other.val[ 4]);
        val[ 5] = _mm_add_ps(val[ 5], other.val[ 5]);
        val[ 6] = _mm_add_ps(val[ 6], other.val[ 6]);
        val[ 7] = _mm_add_ps(val[ 7], other.val[ 7]);
    }

    inline
    short_vec<float, 32> operator+(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_add_ps(val[ 0], other.val[ 0]),
            _mm_add_ps(val[ 1], other.val[ 1]),
            _mm_add_ps(val[ 2], other.val[ 2]),
            _mm_add_ps(val[ 3], other.val[ 3]),
            _mm_add_ps(val[ 4], other.val[ 4]),
            _mm_add_ps(val[ 5], other.val[ 5]),
            _mm_add_ps(val[ 6], other.val[ 6]),
            _mm_add_ps(val[ 7], other.val[ 7]));
    }

    inline
    void operator*=(const short_vec<float, 32>& other)
    {
        val[ 0] = _mm_mul_ps(val[ 0], other.val[ 0]);
        val[ 1] = _mm_mul_ps(val[ 1], other.val[ 1]);
        val[ 2] = _mm_mul_ps(val[ 2], other.val[ 2]);
        val[ 3] = _mm_mul_ps(val[ 3], other.val[ 3]);
        val[ 4] = _mm_mul_ps(val[ 4], other.val[ 4]);
        val[ 5] = _mm_mul_ps(val[ 5], other.val[ 5]);
        val[ 6] = _mm_mul_ps(val[ 6], other.val[ 6]);
        val[ 7] = _mm_mul_ps(val[ 7], other.val[ 7]);
    }

    inline
    short_vec<float, 32> operator*(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_mul_ps(val[ 0], other.val[ 0]),
            _mm_mul_ps(val[ 1], other.val[ 1]),
            _mm_mul_ps(val[ 2], other.val[ 2]),
            _mm_mul_ps(val[ 3], other.val[ 3]),
            _mm_mul_ps(val[ 4], other.val[ 4]),
            _mm_mul_ps(val[ 5], other.val[ 5]),
            _mm_mul_ps(val[ 6], other.val[ 6]),
            _mm_mul_ps(val[ 7], other.val[ 7]));
    }

    inline
    void operator/=(const short_vec<float, 32>& other)
    {
        val[ 0] = _mm_div_ps(val[ 0], other.val[ 0]);
        val[ 1] = _mm_div_ps(val[ 1], other.val[ 1]);
        val[ 2] = _mm_div_ps(val[ 2], other.val[ 2]);
        val[ 3] = _mm_div_ps(val[ 3], other.val[ 3]);
        val[ 4] = _mm_div_ps(val[ 4], other.val[ 4]);
        val[ 5] = _mm_div_ps(val[ 5], other.val[ 5]);
        val[ 6] = _mm_div_ps(val[ 6], other.val[ 6]);
        val[ 7] = _mm_div_ps(val[ 7], other.val[ 7]);
    }

    inline
    void operator/=(const sqrt_reference<float, 32>& other);

    inline
    short_vec<float, 32> operator/(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_div_ps(val[ 0], other.val[ 0]),
            _mm_div_ps(val[ 1], other.val[ 1]),
            _mm_div_ps(val[ 2], other.val[ 2]),
            _mm_div_ps(val[ 3], other.val[ 3]),
            _mm_div_ps(val[ 4], other.val[ 4]),
            _mm_div_ps(val[ 5], other.val[ 5]),
            _mm_div_ps(val[ 6], other.val[ 6]),
            _mm_div_ps(val[ 7], other.val[ 7]));
    }

    inline
    short_vec<float, 32> operator/(const sqrt_reference<float, 32>& other) const;

    inline
    short_vec<float, 32> operator<(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_cmplt_ps(val[ 0], other.val[ 0]),
            _mm_cmplt_ps(val[ 1], other.val[ 1]),
            _mm_cmplt_ps(val[ 2], other.val[ 2]),
            _mm_cmplt_ps(val[ 3], other.val[ 3]),
            _mm_cmplt_ps(val[ 4], other.val[ 4]),
            _mm_cmplt_ps(val[ 5], other.val[ 5]),
            _mm_cmplt_ps(val[ 6], other.val[ 6]),
            _mm_cmplt_ps(val[ 7], other.val[ 7]));
    }

    inline
    short_vec<float, 32> operator<=(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_cmple_ps(val[ 0], other.val[ 0]),
            _mm_cmple_ps(val[ 1], other.val[ 1]),
            _mm_cmple_ps(val[ 2], other.val[ 2]),
            _mm_cmple_ps(val[ 3], other.val[ 3]),
            _mm_cmple_ps(val[ 4], other.val[ 4]),
            _mm_cmple_ps(val[ 5], other.val[ 5]),
            _mm_cmple_ps(val[ 6], other.val[ 6]),
            _mm_cmple_ps(val[ 7], other.val[ 7]));
    }

    inline
    short_vec<float, 32> operator==(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_cmpeq_ps(val[ 0], other.val[ 0]),
            _mm_cmpeq_ps(val[ 1], other.val[ 1]),
            _mm_cmpeq_ps(val[ 2], other.val[ 2]),
            _mm_cmpeq_ps(val[ 3], other.val[ 3]),
            _mm_cmpeq_ps(val[ 4], other.val[ 4]),
            _mm_cmpeq_ps(val[ 5], other.val[ 5]),
            _mm_cmpeq_ps(val[ 6], other.val[ 6]),
            _mm_cmpeq_ps(val[ 7], other.val[ 7]));
    }

    inline
    short_vec<float, 32> operator>(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_cmpgt_ps(val[ 0], other.val[ 0]),
            _mm_cmpgt_ps(val[ 1], other.val[ 1]),
            _mm_cmpgt_ps(val[ 2], other.val[ 2]),
            _mm_cmpgt_ps(val[ 3], other.val[ 3]),
            _mm_cmpgt_ps(val[ 4], other.val[ 4]),
            _mm_cmpgt_ps(val[ 5], other.val[ 5]),
            _mm_cmpgt_ps(val[ 6], other.val[ 6]),
            _mm_cmpgt_ps(val[ 7], other.val[ 7]));
    }

    inline
    short_vec<float, 32> operator>=(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm_cmpge_ps(val[ 0], other.val[ 0]),
            _mm_cmpge_ps(val[ 1], other.val[ 1]),
            _mm_cmpge_ps(val[ 2], other.val[ 2]),
            _mm_cmpge_ps(val[ 3], other.val[ 3]),
            _mm_cmpge_ps(val[ 4], other.val[ 4]),
            _mm_cmpge_ps(val[ 5], other.val[ 5]),
            _mm_cmpge_ps(val[ 6], other.val[ 6]),
            _mm_cmpge_ps(val[ 7], other.val[ 7]));
    }

    inline
    short_vec<float, 32> sqrt() const
    {
        return short_vec<float, 32>(
            _mm_sqrt_ps(val[ 0]),
            _mm_sqrt_ps(val[ 1]),
            _mm_sqrt_ps(val[ 2]),
            _mm_sqrt_ps(val[ 3]),
            _mm_sqrt_ps(val[ 4]),
            _mm_sqrt_ps(val[ 5]),
            _mm_sqrt_ps(val[ 6]),
            _mm_sqrt_ps(val[ 7]));
    }

    inline
    void load(const float *data)
    {
        val[ 0] = _mm_loadu_ps(data +  0);
        val[ 1] = _mm_loadu_ps(data +  4);
        val[ 2] = _mm_loadu_ps(data +  8);
        val[ 3] = _mm_loadu_ps(data + 12);
        val[ 4] = _mm_loadu_ps(data + 16);
        val[ 5] = _mm_loadu_ps(data + 20);
        val[ 6] = _mm_loadu_ps(data + 24);
        val[ 7] = _mm_loadu_ps(data + 28);
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val[ 0] = _mm_load_ps(data +  0);
        val[ 1] = _mm_load_ps(data +  4);
        val[ 2] = _mm_load_ps(data +  8);
        val[ 3] = _mm_load_ps(data + 12);
        val[ 4] = _mm_load_ps(data + 16);
        val[ 5] = _mm_load_ps(data + 20);
        val[ 6] = _mm_load_ps(data + 24);
        val[ 7] = _mm_load_ps(data + 28);
    }

    inline
    void store(float *data) const
    {
        _mm_storeu_ps(data +  0, val[ 0]);
        _mm_storeu_ps(data +  4, val[ 1]);
        _mm_storeu_ps(data +  8, val[ 2]);
        _mm_storeu_ps(data + 12, val[ 3]);
        _mm_storeu_ps(data + 16, val[ 4]);
        _mm_storeu_ps(data + 20, val[ 5]);
        _mm_storeu_ps(data + 24, val[ 6]);
        _mm_storeu_ps(data + 28, val[ 7]);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_ps(data +  0, val[ 0]);
        _mm_store_ps(data +  4, val[ 1]);
        _mm_store_ps(data +  8, val[ 2]);
        _mm_store_ps(data + 12, val[ 3]);
        _mm_store_ps(data + 16, val[ 4]);
        _mm_store_ps(data + 20, val[ 5]);
        _mm_store_ps(data + 24, val[ 6]);
        _mm_store_ps(data + 28, val[ 7]);
    }

    inline
    void store_nt(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_ps(data +  0, val[ 0]);
        _mm_stream_ps(data +  4, val[ 1]);
        _mm_stream_ps(data +  8, val[ 2]);
        _mm_stream_ps(data + 12, val[ 3]);
        _mm_stream_ps(data + 16, val[ 4]);
        _mm_stream_ps(data + 20, val[ 5]);
        _mm_stream_ps(data + 24, val[ 6]);
        _mm_stream_ps(data + 28, val[ 7]);
    }

#ifdef __SSE4_1__
    inline
    void gather(const float *ptr, const int *offsets)
    {
        val[ 0] = _mm_load_ss(ptr + offsets[0]);
        SHORTVEC_INSERT_PS(val[ 0], ptr, offsets[ 1], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val[ 0], ptr, offsets[ 2], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val[ 0], ptr, offsets[ 3], _MM_MK_INSERTPS_NDX(0,3,0));

        val[ 1] = _mm_load_ss(ptr + offsets[4]);
        SHORTVEC_INSERT_PS(val[ 1], ptr, offsets[ 5], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val[ 1], ptr, offsets[ 6], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val[ 1], ptr, offsets[ 7], _MM_MK_INSERTPS_NDX(0,3,0));

        val[ 2] = _mm_load_ss(ptr + offsets[8]);
        SHORTVEC_INSERT_PS(val[ 2], ptr, offsets[ 9], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val[ 2], ptr, offsets[10], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val[ 2], ptr, offsets[11], _MM_MK_INSERTPS_NDX(0,3,0));

        val[ 3] = _mm_load_ss(ptr + offsets[12]);
        SHORTVEC_INSERT_PS(val[ 3], ptr, offsets[13], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val[ 3], ptr, offsets[14], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val[ 3], ptr, offsets[15], _MM_MK_INSERTPS_NDX(0,3,0));

        val[ 4] = _mm_load_ss(ptr + offsets[16]);
        SHORTVEC_INSERT_PS(val[ 4], ptr, offsets[17], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val[ 4], ptr, offsets[18], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val[ 4], ptr, offsets[19], _MM_MK_INSERTPS_NDX(0,3,0));

        val[ 5] = _mm_load_ss(ptr + offsets[20]);
        SHORTVEC_INSERT_PS(val[ 5], ptr, offsets[21], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val[ 5], ptr, offsets[22], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val[ 5], ptr, offsets[23], _MM_MK_INSERTPS_NDX(0,3,0));

        val[ 6] = _mm_load_ss(ptr + offsets[24]);
        SHORTVEC_INSERT_PS(val[ 6], ptr, offsets[25], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val[ 6], ptr, offsets[26], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val[ 6], ptr, offsets[27], _MM_MK_INSERTPS_NDX(0,3,0));

        val[ 7] = _mm_load_ss(ptr + offsets[28]);
        SHORTVEC_INSERT_PS(val[ 7], ptr, offsets[29], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val[ 7], ptr, offsets[30], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val[ 7], ptr, offsets[31], _MM_MK_INSERTPS_NDX(0,3,0));
    }

    inline
    void scatter(float *ptr, const int *offsets) const
    {
        ShortVecHelpers::ExtractResult r1, r2, r3, r4;
        r1.i = _mm_extract_ps(val[ 0], 0);
        r2.i = _mm_extract_ps(val[ 0], 1);
        r3.i = _mm_extract_ps(val[ 0], 2);
        r4.i = _mm_extract_ps(val[ 0], 3);
        ptr[offsets[0]] = r1.f;
        ptr[offsets[1]] = r2.f;
        ptr[offsets[2]] = r3.f;
        ptr[offsets[3]] = r4.f;

        r1.i = _mm_extract_ps(val[ 1], 0);
        r2.i = _mm_extract_ps(val[ 1], 1);
        r3.i = _mm_extract_ps(val[ 1], 2);
        r4.i = _mm_extract_ps(val[ 1], 3);
        ptr[offsets[4]] = r1.f;
        ptr[offsets[5]] = r2.f;
        ptr[offsets[6]] = r3.f;
        ptr[offsets[7]] = r4.f;

        r1.i = _mm_extract_ps(val[ 2], 0);
        r2.i = _mm_extract_ps(val[ 2], 1);
        r3.i = _mm_extract_ps(val[ 2], 2);
        r4.i = _mm_extract_ps(val[ 2], 3);
        ptr[offsets[ 8]] = r1.f;
        ptr[offsets[ 9]] = r2.f;
        ptr[offsets[10]] = r3.f;
        ptr[offsets[11]] = r4.f;

        r1.i = _mm_extract_ps(val[ 3], 0);
        r2.i = _mm_extract_ps(val[ 3], 1);
        r3.i = _mm_extract_ps(val[ 3], 2);
        r4.i = _mm_extract_ps(val[ 3], 3);
        ptr[offsets[12]] = r1.f;
        ptr[offsets[13]] = r2.f;
        ptr[offsets[14]] = r3.f;
        ptr[offsets[15]] = r4.f;

        r1.i = _mm_extract_ps(val[ 4], 0);
        r2.i = _mm_extract_ps(val[ 4], 1);
        r3.i = _mm_extract_ps(val[ 4], 2);
        r4.i = _mm_extract_ps(val[ 4], 3);
        ptr[offsets[16]] = r1.f;
        ptr[offsets[17]] = r2.f;
        ptr[offsets[18]] = r3.f;
        ptr[offsets[19]] = r4.f;

        r1.i = _mm_extract_ps(val[ 5], 0);
        r2.i = _mm_extract_ps(val[ 5], 1);
        r3.i = _mm_extract_ps(val[ 5], 2);
        r4.i = _mm_extract_ps(val[ 5], 3);
        ptr[offsets[20]] = r1.f;
        ptr[offsets[21]] = r2.f;
        ptr[offsets[22]] = r3.f;
        ptr[offsets[23]] = r4.f;

        r1.i = _mm_extract_ps(val[ 6], 0);
        r2.i = _mm_extract_ps(val[ 6], 1);
        r3.i = _mm_extract_ps(val[ 6], 2);
        r4.i = _mm_extract_ps(val[ 6], 3);
        ptr[offsets[24]] = r1.f;
        ptr[offsets[25]] = r2.f;
        ptr[offsets[26]] = r3.f;
        ptr[offsets[27]] = r4.f;

        r1.i = _mm_extract_ps(val[ 7], 0);
        r2.i = _mm_extract_ps(val[ 7], 1);
        r3.i = _mm_extract_ps(val[ 7], 2);
        r4.i = _mm_extract_ps(val[ 7], 3);
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
        val[ 0] = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[4]);
        f2   = _mm_load_ss(ptr + offsets[6]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[5]);
        f4   = _mm_load_ss(ptr + offsets[7]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val[ 1] = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[ 8]);
        f2   = _mm_load_ss(ptr + offsets[10]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[ 9]);
        f4   = _mm_load_ss(ptr + offsets[11]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val[ 2] = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[12]);
        f2   = _mm_load_ss(ptr + offsets[14]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[13]);
        f4   = _mm_load_ss(ptr + offsets[15]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val[ 3] = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[16]);
        f2   = _mm_load_ss(ptr + offsets[18]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[17]);
        f4   = _mm_load_ss(ptr + offsets[19]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val[ 4] = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[20]);
        f2   = _mm_load_ss(ptr + offsets[22]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[21]);
        f4   = _mm_load_ss(ptr + offsets[23]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val[ 5] = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[24]);
        f2   = _mm_load_ss(ptr + offsets[26]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[25]);
        f4   = _mm_load_ss(ptr + offsets[27]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val[ 6] = _mm_unpacklo_ps(f1, f3);

        f1   = _mm_load_ss(ptr + offsets[28]);
        f2   = _mm_load_ss(ptr + offsets[30]);
        f1   = _mm_unpacklo_ps(f1, f2);
        f3   = _mm_load_ss(ptr + offsets[29]);
        f4   = _mm_load_ss(ptr + offsets[31]);
        f3   = _mm_unpacklo_ps(f3, f4);
        val[ 7] = _mm_unpacklo_ps(f1, f3);

    }

    inline
    void scatter(float *ptr, const int *offsets) const
    {
        __m128 tmp = val[ 0];
        _mm_store_ss(ptr + offsets[0], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[1], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[2], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[3], tmp);

        tmp = val[ 1];
        _mm_store_ss(ptr + offsets[4], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[5], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[6], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[7], tmp);

        tmp = val[ 2];
        _mm_store_ss(ptr + offsets[8], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[9], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[10], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[11], tmp);

        tmp = val[ 3];
        _mm_store_ss(ptr + offsets[12], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[13], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[14], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[15], tmp);

        tmp = val[ 4];
        _mm_store_ss(ptr + offsets[16], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[17], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[18], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[19], tmp);

        tmp = val[ 5];
        _mm_store_ss(ptr + offsets[20], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[21], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[22], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[23], tmp);

        tmp = val[ 6];
        _mm_store_ss(ptr + offsets[24], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[25], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[26], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[27], tmp);

        tmp = val[ 7];
        _mm_store_ss(ptr + offsets[28], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[29], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[30], tmp);
        tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
        _mm_store_ss(ptr + offsets[31], tmp);
   }
#endif

    inline
    void blend(const mask_type& mask, const short_vec<float, 32>& other)
    {
#ifdef __SSE4_1__
        val[ 0] = _mm_blendv_ps(val[ 0], other.val[ 0], mask.val[ 0]);
        val[ 1] = _mm_blendv_ps(val[ 1], other.val[ 1], mask.val[ 1]);
        val[ 2] = _mm_blendv_ps(val[ 2], other.val[ 2], mask.val[ 2]);
        val[ 3] = _mm_blendv_ps(val[ 3], other.val[ 3], mask.val[ 3]);
        val[ 4] = _mm_blendv_ps(val[ 4], other.val[ 4], mask.val[ 4]);
        val[ 5] = _mm_blendv_ps(val[ 5], other.val[ 5], mask.val[ 5]);
        val[ 6] = _mm_blendv_ps(val[ 6], other.val[ 6], mask.val[ 6]);
        val[ 7] = _mm_blendv_ps(val[ 7], other.val[ 7], mask.val[ 7]);
#else
        val[ 0] = _mm_or_ps(
            _mm_and_ps(mask.val[ 0], other.val[ 0]),
            _mm_andnot_ps(mask.val[ 0], val[ 0]));
        val[ 1] = _mm_or_ps(
            _mm_and_ps(mask.val[ 1], other.val[ 1]),
            _mm_andnot_ps(mask.val[ 1], val[ 1]));
        val[ 2] = _mm_or_ps(
            _mm_and_ps(mask.val[ 2], other.val[ 2]),
            _mm_andnot_ps(mask.val[ 2], val[ 2]));
        val[ 3] = _mm_or_ps(
            _mm_and_ps(mask.val[ 3], other.val[ 3]),
            _mm_andnot_ps(mask.val[ 3], val[ 3]));
        val[ 4] = _mm_or_ps(
            _mm_and_ps(mask.val[ 4], other.val[ 4]),
            _mm_andnot_ps(mask.val[ 4], val[ 4]));
        val[ 5] = _mm_or_ps(
            _mm_and_ps(mask.val[ 5], other.val[ 5]),
            _mm_andnot_ps(mask.val[ 5], val[ 5]));
        val[ 6] = _mm_or_ps(
            _mm_and_ps(mask.val[ 6], other.val[ 6]),
            _mm_andnot_ps(mask.val[ 6], val[ 6]));
        val[ 7] = _mm_or_ps(
            _mm_and_ps(mask.val[ 7], other.val[ 7]),
            _mm_andnot_ps(mask.val[ 7], val[ 7]));
#endif
    }

private:
    __m128 val[8];
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
    template<typename OTHER_CARGO, std::size_t OTHER_ARITY>
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
    val{_mm_sqrt_ps(other.vec.val[ 0]),
        _mm_sqrt_ps(other.vec.val[ 1]),
        _mm_sqrt_ps(other.vec.val[ 2]),
        _mm_sqrt_ps(other.vec.val[ 3]),
        _mm_sqrt_ps(other.vec.val[ 4]),
        _mm_sqrt_ps(other.vec.val[ 5]),
        _mm_sqrt_ps(other.vec.val[ 6]),
        _mm_sqrt_ps(other.vec.val[ 7])}
{}

inline
void short_vec<float, 32>::operator/=(const sqrt_reference<float, 32>& other)
{
    val[ 0] = _mm_mul_ps(val[ 0], _mm_rsqrt_ps(other.vec.val[ 0]));
    val[ 1] = _mm_mul_ps(val[ 1], _mm_rsqrt_ps(other.vec.val[ 1]));
    val[ 2] = _mm_mul_ps(val[ 2], _mm_rsqrt_ps(other.vec.val[ 2]));
    val[ 3] = _mm_mul_ps(val[ 3], _mm_rsqrt_ps(other.vec.val[ 3]));
    val[ 4] = _mm_mul_ps(val[ 4], _mm_rsqrt_ps(other.vec.val[ 4]));
    val[ 5] = _mm_mul_ps(val[ 5], _mm_rsqrt_ps(other.vec.val[ 5]));
    val[ 6] = _mm_mul_ps(val[ 6], _mm_rsqrt_ps(other.vec.val[ 6]));
    val[ 7] = _mm_mul_ps(val[ 7], _mm_rsqrt_ps(other.vec.val[ 7]));
}

inline
short_vec<float, 32> short_vec<float, 32>::operator/(const sqrt_reference<float, 32>& other) const
{
    return short_vec<float, 32>(
        _mm_mul_ps(val[ 0], _mm_rsqrt_ps(other.vec.val[ 0])),
        _mm_mul_ps(val[ 1], _mm_rsqrt_ps(other.vec.val[ 1])),
        _mm_mul_ps(val[ 2], _mm_rsqrt_ps(other.vec.val[ 2])),
        _mm_mul_ps(val[ 3], _mm_rsqrt_ps(other.vec.val[ 3])),
        _mm_mul_ps(val[ 4], _mm_rsqrt_ps(other.vec.val[ 4])),
        _mm_mul_ps(val[ 5], _mm_rsqrt_ps(other.vec.val[ 5])),
        _mm_mul_ps(val[ 6], _mm_rsqrt_ps(other.vec.val[ 6])),
        _mm_mul_ps(val[ 7], _mm_rsqrt_ps(other.vec.val[ 7])));
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
    const float *data1 = reinterpret_cast<const float *>(&vec.val[ 0]);
    const float *data2 = reinterpret_cast<const float *>(&vec.val[ 1]);
    const float *data3 = reinterpret_cast<const float *>(&vec.val[ 2]);
    const float *data4 = reinterpret_cast<const float *>(&vec.val[ 3]);
    const float *data5 = reinterpret_cast<const float *>(&vec.val[ 4]);
    const float *data6 = reinterpret_cast<const float *>(&vec.val[ 5]);
    const float *data7 = reinterpret_cast<const float *>(&vec.val[ 6]);
    const float *data8 = reinterpret_cast<const float *>(&vec.val[ 7]);
    __os << "["
         << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3] << ", "
         << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3] << ", "
         << data3[0] << ", " << data3[1] << ", " << data3[2] << ", " << data3[3] << ", "
         << data4[0] << ", " << data4[1] << ", " << data4[2] << ", " << data4[3] << ", "
         << data5[0] << ", " << data5[1] << ", " << data5[2] << ", " << data5[3] << ", "
         << data6[0] << ", " << data6[1] << ", " << data6[2] << ", " << data6[3] << ", "
         << data7[0] << ", " << data7[1] << ", " << data7[2] << ", " << data7[3] << ", "
         << data8[0] << ", " << data8[1] << ", " << data8[2] << ", " << data8[3] << "]";
    return __os;
}

}

#endif

#endif
