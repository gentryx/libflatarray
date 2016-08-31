/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_FLOAT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_FLOAT_32_HPP

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
    typedef unsigned mask_type;
    typedef short_vec_strategy::avx512f strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 32>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm512_set1_ps(data)),
        val2(_mm512_set1_ps(data))
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(const __m512& val1, const __m512& val2) :
        val1(val1), val2(val2)
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
        __m512 buf0 = _mm512_or_ps(val1, val2);
        __m128 buf1 = _mm_or_ps(
            _mm_or_ps(
                _mm512_extractf32x4_ps(buf0, 0),
                _mm512_extractf32x4_ps(buf0, 1)),
            _mm_or_ps(
                _mm512_extractf32x4_ps(buf0, 2),
                _mm512_extractf32x4_ps(buf0, 3)));
        // shuffle upper 64-bit half down to first 64 bits so we can
        // "or" both together:
        __m128 buf2 = _mm_shuffle_ps(buf1, buf1, (3 << 2) | (2 << 0));
        buf2 = _mm_or_ps(buf1, buf2);
        // another shuffle to extract 2nd least significant float
        // member and or it together with least significant float
        // member:
        buf1 = _mm_shuffle_ps(buf2, buf2, (1 << 0));
        return _mm_cvtss_f32(buf1) || _mm_cvtss_f32(buf2);
    }

    inline
    float get(int i) const
    {
        __m512 buf0;
        if (i < 16) {
            buf0 = val1;
        } else {
            buf0 = val2;
        }

        i &= 15;

        __m128 buf1;
        if (i < 8) {
            if (i < 4) {
                buf1 =  _mm512_extractf32x4_ps(buf0, 0);
            } else {
                buf1 =  _mm512_extractf32x4_ps(buf0, 1);
            }
        } else {
            if (i < 12)  {
                buf1 =  _mm512_extractf32x4_ps(buf0, 2);
            } else {
                buf1 =  _mm512_extractf32x4_ps(buf0, 3);
            }
        }

        i &= 3;

        if (i == 3) {
            return _mm_cvtss_f32(_mm_shuffle_ps(buf1, buf1, 3));
        }
        if (i == 2) {
            return _mm_cvtss_f32(_mm_shuffle_ps(buf1, buf1, 2));
        }
        if (i == 1) {
            return _mm_cvtss_f32(_mm_shuffle_ps(buf1, buf1, 1));
        }

        return _mm_cvtss_f32(buf1);
    }

    inline
    void operator-=(const short_vec<float, 32>& other)
    {
        val1 = _mm512_sub_ps(val1, other.val1);
        val2 = _mm512_sub_ps(val2, other.val2);
    }

    inline
    short_vec<float, 32> operator-(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm512_sub_ps(val1, other.val1),
            _mm512_sub_ps(val2, other.val2));
    }

    inline
    void operator+=(const short_vec<float, 32>& other)
    {
        val1 = _mm512_add_ps(val1, other.val1);
        val2 = _mm512_add_ps(val2, other.val2);
    }

    inline
    short_vec<float, 32> operator+(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm512_add_ps(val1, other.val1),
            _mm512_add_ps(val2, other.val2));
    }

    inline
    void operator*=(const short_vec<float, 32>& other)
    {
        val1 = _mm512_mul_ps(val1, other.val1);
        val2 = _mm512_mul_ps(val2, other.val2);
    }

    inline
    short_vec<float, 32> operator*(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm512_mul_ps(val1, other.val1),
            _mm512_mul_ps(val2, other.val2));
    }

    inline
    void operator/=(const short_vec<float, 32>& other)
    {
        val1 = _mm512_mul_ps(val1, _mm512_rcp14_ps(other.val1));
        val2 = _mm512_mul_ps(val2, _mm512_rcp14_ps(other.val2));
    }

    inline
    void operator/=(const sqrt_reference<float, 32>& other);

    inline
    short_vec<float, 32> operator/(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm512_mul_ps(val1, _mm512_rcp14_ps(other.val1)),
            _mm512_mul_ps(val2, _mm512_rcp14_ps(other.val2)));
    }

    inline
    short_vec<float, 32> operator/(const sqrt_reference<float, 32>& other) const;

    inline
    mask_type operator<(const short_vec<float, 32>& other) const
    {
        return
            (_mm512_cmp_ps_mask(val1, other.val1, _CMP_LT_OS) <<  0) +
            (_mm512_cmp_ps_mask(val2, other.val2, _CMP_LT_OS) << 16);
    }

    inline
    mask_type operator<=(const short_vec<float, 32>& other) const
    {
        return
            (_mm512_cmp_ps_mask(val1, other.val1, _CMP_LE_OS) <<  0) +
            (_mm512_cmp_ps_mask(val2, other.val2, _CMP_LE_OS) << 16);
    }

    inline
    mask_type operator==(const short_vec<float, 32>& other) const
    {
        return
            (_mm512_cmp_ps_mask(val1, other.val1, _CMP_EQ_OQ) <<  0) +
            (_mm512_cmp_ps_mask(val2, other.val2, _CMP_EQ_OQ) << 16);
    }

    inline
    mask_type operator>(const short_vec<float, 32>& other) const
    {
        return
            (_mm512_cmp_ps_mask(val1, other.val1, _CMP_GT_OS) <<  0) +
            (_mm512_cmp_ps_mask(val2, other.val2, _CMP_GT_OS) << 16);
    }

    inline
    mask_type operator>=(const short_vec<float, 32>& other) const
    {
        return
            (_mm512_cmp_ps_mask(val1, other.val1, _CMP_GE_OS) <<  0) +
            (_mm512_cmp_ps_mask(val2, other.val2, _CMP_GE_OS) << 16);
    }

    inline
    short_vec<float, 32> sqrt() const
    {
        return short_vec<float, 32>(
            _mm512_sqrt_ps(val1),
            _mm512_sqrt_ps(val2));
    }

    inline
    void load(const float *data)
    {
        val1 = _mm512_loadu_ps(data +  0);
        val2 = _mm512_loadu_ps(data + 16);
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        val1 = _mm512_load_ps(data +  0);
        val2 = _mm512_load_ps(data + 16);
    }

    inline
    void store(float *data) const
    {
        _mm512_storeu_ps(data +  0, val1);
        _mm512_storeu_ps(data + 16, val2);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_store_ps(data +  0, val1);
        _mm512_store_ps(data + 16, val2);
    }

    inline
    void store_nt(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_stream_ps(data +  0, val1);
        _mm512_stream_ps(data + 16, val2);
    }

    inline
    void gather(const float *ptr, const int *offsets)
    {
        __m512i indices;
        SHORTVEC_ASSERT_ALIGNED(offsets, 64);
        indices = _mm512_load_epi32(offsets);
        val1    = _mm512_i32gather_ps(indices, ptr, 4);
        indices = _mm512_load_epi32(offsets + 16);
        val2    = _mm512_i32gather_ps(indices, ptr, 4);
    }

    inline
    void scatter(float *ptr, const int *offsets) const
    {
        __m512i indices;
        SHORTVEC_ASSERT_ALIGNED(offsets, 64);
        indices = _mm512_load_epi32(offsets);
        _mm512_i32scatter_ps(ptr, indices, val1, 4);
        indices = _mm512_load_epi32(offsets + 16);
        _mm512_i32scatter_ps(ptr, indices, val2, 4);
    }

private:
    __m512 val1;
    __m512 val2;
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
    val1(_mm512_sqrt_ps(other.vec.val1)),
    val2(_mm512_sqrt_ps(other.vec.val2))
{}

inline
void short_vec<float, 32>::operator/=(const sqrt_reference<float, 32>& other)
{
    val1 = _mm512_mul_ps(val1, _mm512_rsqrt14_ps(other.vec.val1));
    val2 = _mm512_mul_ps(val2, _mm512_rsqrt14_ps(other.vec.val2));
}

inline
short_vec<float, 32> short_vec<float, 32>::operator/(const sqrt_reference<float, 32>& other) const
{
    return short_vec<float, 32>(
        _mm512_mul_ps(val1, _mm512_rsqrt14_ps(other.vec.val1)),
        _mm512_mul_ps(val2, _mm512_rsqrt14_ps(other.vec.val2)));
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
    __os << "["  << data1[ 0] << ", " << data1[ 1] << ", " << data1[ 2] << ", " << data1[ 3]
         << ", " << data1[ 4] << ", " << data1[ 5] << ", " << data1[ 6] << ", " << data1[ 7]
         << ", " << data1[ 8] << ", " << data1[ 9] << ", " << data1[10] << ", " << data1[11]
         << ", " << data1[12] << ", " << data1[13] << ", " << data1[14] << ", " << data1[15]
         << ", " << data2[ 0] << ", " << data2[ 1] << ", " << data2[ 2] << ", " << data2[ 3]
         << ", " << data2[ 4] << ", " << data2[ 5] << ", " << data2[ 6] << ", " << data2[ 7]
         << ", " << data2[ 8] << ", " << data2[ 9] << ", " << data2[10] << ", " << data2[11]
         << ", " << data2[12] << ", " << data2[13] << ", " << data2[14] << ", " << data2[15]
         << "]";
    return __os;
}

}

#endif

#endif
