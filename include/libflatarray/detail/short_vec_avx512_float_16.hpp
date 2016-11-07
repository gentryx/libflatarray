/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_FLOAT_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_FLOAT_16_HPP

#if LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX512F

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
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
class short_vec<float, 16>
{
public:
    static const std::size_t ARITY = 16;
    typedef __mmask16 mask_type;
    typedef short_vec_strategy::avx512f strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 16>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm512_set1_ps(data))
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(const __m512& val1) :
        val1(val1)
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
    short_vec(const sqrt_reference<float, 16>& other);

    inline
    bool any() const
    {
        return _mm512_test_epi64_mask(
            _mm512_castps_si512(val1),
            _mm512_castps_si512(val1));
    }

    inline
    float get(int i) const
    {
        __m128 buf0;
        if (i < 8) {
            if (i < 4) {
                buf0 =  _mm512_extractf32x4_ps(val1, 0);
            } else {
                buf0 =  _mm512_extractf32x4_ps(val1, 1);
            }
        } else {
            if (i < 12)  {
                buf0 =  _mm512_extractf32x4_ps(val1, 2);
            } else {
                buf0 =  _mm512_extractf32x4_ps(val1, 3);
            }
        }

        i &= 3;

        if (i == 3) {
            return _mm_cvtss_f32(_mm_shuffle_ps(buf0, buf0, 3));
        }
        if (i == 2) {
            return _mm_cvtss_f32(_mm_shuffle_ps(buf0, buf0, 2));
        }
        if (i == 1) {
            return _mm_cvtss_f32(_mm_shuffle_ps(buf0, buf0, 1));
        }

        return _mm_cvtss_f32(buf0);
    }

    inline
    void operator-=(const short_vec<float, 16>& other)
    {
        val1 = _mm512_sub_ps(val1, other.val1);
    }

    inline
    short_vec<float, 16> operator-(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm512_sub_ps(val1, other.val1));
    }

    inline
    void operator+=(const short_vec<float, 16>& other)
    {
        val1 = _mm512_add_ps(val1, other.val1);
    }

    inline
    short_vec<float, 16> operator+(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm512_add_ps(val1, other.val1));
    }

    inline
    void operator*=(const short_vec<float, 16>& other)
    {
        val1 = _mm512_mul_ps(val1, other.val1);
    }

    inline
    short_vec<float, 16> operator*(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm512_mul_ps(val1, other.val1));
    }

    inline
    void operator/=(const short_vec<float, 16>& other)
    {
        val1 = _mm512_mul_ps(val1, _mm512_rcp14_ps(other.val1));
    }

    inline
    void operator/=(const sqrt_reference<float, 16>& other);

    inline
    short_vec<float, 16> operator/(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm512_mul_ps(val1, _mm512_rcp14_ps(other.val1)));
    }

    inline
    short_vec<float, 16> operator/(const sqrt_reference<float, 16>& other) const;

    inline
    mask_type operator<(const short_vec<float, 16>& other) const
    {
        return
            (_mm512_cmp_ps_mask(val1, other.val1, _CMP_LT_OS) <<  0);
    }

    inline
    mask_type operator<=(const short_vec<float, 16>& other) const
    {
        return
            (_mm512_cmp_ps_mask(val1, other.val1, _CMP_LE_OS) <<  0);
    }

    inline
    mask_type operator==(const short_vec<float, 16>& other) const
    {
        return
            (_mm512_cmp_ps_mask(val1, other.val1, _CMP_EQ_OQ) <<  0);
    }

    inline
    mask_type operator>(const short_vec<float, 16>& other) const
    {
        return
            (_mm512_cmp_ps_mask(val1, other.val1, _CMP_GT_OS) <<  0);
    }

    inline
    mask_type operator>=(const short_vec<float, 16>& other) const
    {
        return
            (_mm512_cmp_ps_mask(val1, other.val1, _CMP_GE_OS) <<  0);
    }

    inline
    short_vec<float, 16> sqrt() const
    {
        return short_vec<float, 16>(
            _mm512_sqrt_ps(val1));
    }

    inline
    void load(const float *data)
    {
        val1 = _mm512_loadu_ps(data);
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        val1 = _mm512_load_ps(data);
    }

    inline
    void store(float *data) const
    {
        _mm512_storeu_ps(data, val1);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_store_ps(data, val1);
    }

    inline
    void store_nt(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_stream_ps(data, val1);
    }

    inline
    void gather(const float *ptr, const int *offsets)
    {
        __m512i indices;
        SHORTVEC_ASSERT_ALIGNED(offsets, 64);
        indices = _mm512_load_epi32(offsets);
        val1    = _mm512_i32gather_ps(indices, ptr, 4);
    }

    inline
    void scatter(float *ptr, const int *offsets) const
    {
        __m512i indices;
        SHORTVEC_ASSERT_ALIGNED(offsets, 64);
        indices = _mm512_load_epi32(offsets);
        _mm512_i32scatter_ps(ptr, indices, val1, 4);
    }

    inline
    void blend(const mask_type& mask, const short_vec<float, 16>& other)
    {
        val1 = _mm512_mask_blend_ps((mask >>  0)        , val1, other.val1);
    }

private:
    __m512 val1;
};

inline
void operator<<(float *data, const short_vec<float, 16>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<float, 16>
{
public:
    template<typename OTHER_CARGO, std::size_t OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<float, 16>& vec) :
        vec(vec)
    {}

private:
    short_vec<float, 16> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<float, 16>::short_vec(const sqrt_reference<float, 16>& other) :
    val1(_mm512_sqrt_ps(other.vec.val1))
{}

inline
void short_vec<float, 16>::operator/=(const sqrt_reference<float, 16>& other)
{
    val1 = _mm512_mul_ps(val1, _mm512_rsqrt14_ps(other.vec.val1));
}

inline
short_vec<float, 16> short_vec<float, 16>::operator/(const sqrt_reference<float, 16>& other) const
{
    return short_vec<float, 16>(
        _mm512_mul_ps(val1, _mm512_rsqrt14_ps(other.vec.val1)));
}

inline
sqrt_reference<float, 16> sqrt(const short_vec<float, 16>& vec)
{
    return sqrt_reference<float, 16>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 16>& vec)
{
    const float *data1 = reinterpret_cast<const float *>(&vec.val1);
    __os << "["  << data1[ 0] << ", " << data1[ 1] << ", " << data1[ 2] << ", " << data1[ 3]
         << ", " << data1[ 4] << ", " << data1[ 5] << ", " << data1[ 6] << ", " << data1[ 7]
         << ", " << data1[ 8] << ", " << data1[ 9] << ", " << data1[10] << ", " << data1[11]
         << ", " << data1[12] << ", " << data1[13] << ", " << data1[14] << ", " << data1[15]
         << "]";
    return __os;
}

}

#endif

#endif
