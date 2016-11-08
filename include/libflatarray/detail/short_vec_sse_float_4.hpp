/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_4_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_4_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE2) ||            \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE4_1) ||          \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX2) ||            \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_AVX512F)

#include <emmintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>
#include <libflatarray/short_vec_base.hpp>

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

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
class short_vec<float, 4> : public short_vec_base<float, 4>
{
public:
    static const std::size_t ARITY = 4;
    typedef short_vec<float, 4> mask_type;
    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 4>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm_set1_ps(data))
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(const __m128& val1) :
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
    short_vec(const sqrt_reference<float, 4>& other);

    inline
    bool any() const
    {
#ifdef __SSE4_1__
        return (0 == _mm_testz_si128(
                    _mm_castps_si128(val1),
                    _mm_castps_si128(val1)));
#else
        __m128 buf1 = _mm_shuffle_ps(val1, val1, (3 << 2) | (2 << 0));
        buf1 = _mm_or_ps(val1, buf1);
        __m128 buf2 = _mm_shuffle_ps(buf1, buf1, (1 << 0));
        return _mm_cvtss_f32(buf1) || _mm_cvtss_f32(buf2);
#endif
    }

    inline
    float operator[](int i) const
    {
        if (i == 3) {
            return _mm_cvtss_f32(_mm_shuffle_ps(val1, val1, 3));
        }
        if (i == 2) {
            return _mm_cvtss_f32(_mm_shuffle_ps(val1, val1, 2));
        }
        if (i == 1) {
            return _mm_cvtss_f32(_mm_shuffle_ps(val1, val1, 1));
        }

        return _mm_cvtss_f32(val1);
    }

    inline
    void operator-=(const short_vec<float, 4>& other)
    {
        val1 = _mm_sub_ps(val1, other.val1);
    }

    inline
    short_vec<float, 4> operator-(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(
            _mm_sub_ps(val1, other.val1));
    }

    inline
    void operator+=(const short_vec<float, 4>& other)
    {
        val1 = _mm_add_ps(val1, other.val1);
    }

    inline
    short_vec<float, 4> operator+(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(
            _mm_add_ps(val1, other.val1));
    }

    inline
    void operator*=(const short_vec<float, 4>& other)
    {
        val1 = _mm_mul_ps(val1, other.val1);
    }

    inline
    short_vec<float, 4> operator*(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(
            _mm_mul_ps(val1, other.val1));
    }

    inline
    void operator/=(const short_vec<float, 4>& other)
    {
        val1 = _mm_div_ps(val1, other.val1);
    }

    inline
    void operator/=(const sqrt_reference<float, 4>& other);

    inline
    short_vec<float, 4> operator/(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(
            _mm_div_ps(val1, other.val1));
    }

    inline
    short_vec<float, 4> operator/(const sqrt_reference<float, 4>& other) const;

    inline
    short_vec<float, 4> operator<(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(
            _mm_cmplt_ps(val1, other.val1));
    }

    inline
    short_vec<float, 4> operator<=(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(
            _mm_cmple_ps(val1, other.val1));
    }

    inline
    short_vec<float, 4> operator==(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(
            _mm_cmpeq_ps(val1, other.val1));
    }

    inline
    short_vec<float, 4> operator>(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(
            _mm_cmpgt_ps(val1, other.val1));
    }

    inline
    short_vec<float, 4> operator>=(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(
            _mm_cmpge_ps(val1, other.val1));
    }

    inline
    short_vec<float, 4> sqrt() const
    {
        return short_vec<float, 4>(
            _mm_sqrt_ps(val1));
    }

    inline
    void load(const float *data)
    {
        val1 = _mm_loadu_ps(data);
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val1 = _mm_load_ps(data);
    }

    inline
    void store(float *data) const
    {
        _mm_storeu_ps(data + 0, val1);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_ps(data + 0, val1);
    }

    inline
    void store_nt(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_ps(data + 0, val1);
    }

#ifdef __SSE4_1__
    inline
    void gather(const float *ptr, const int *offsets)
    {
        val1 = _mm_load_ss(ptr + offsets[0]);
        SHORTVEC_INSERT_PS(val1, ptr, offsets[1], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val1, ptr, offsets[2], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val1, ptr, offsets[3], _MM_MK_INSERTPS_NDX(0,3,0));
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
    }
#endif

    inline
    void blend(const mask_type& mask, const short_vec<float, 4>& other)
    {
#ifdef __SSE4_1__
        val1 = _mm_blendv_ps(val1, other.val1, mask.val1);
#else
        val1 = _mm_or_ps(
            _mm_and_ps(mask.val1, other.val1),
            _mm_andnot_ps(mask.val1, val1));
#endif
    }

private:
    __m128 val1;
};

inline
void operator<<(float *data, const short_vec<float, 4>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<float, 4>
{
public:
    template<typename OTHER_CARGO, std::size_t OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<float, 4>& vec) :
        vec(vec)
    {}

private:
    short_vec<float, 4> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<float, 4>::short_vec(const sqrt_reference<float, 4>& other) :
    val1(_mm_sqrt_ps(other.vec.val1))
{}

inline
void short_vec<float, 4>::operator/=(const sqrt_reference<float, 4>& other)
{
    val1 = _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1));
}

inline
short_vec<float, 4> short_vec<float, 4>::operator/(const sqrt_reference<float, 4>& other) const
{
    return short_vec<float, 4>(
        _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1)));
}

inline
sqrt_reference<float, 4> sqrt(const short_vec<float, 4>& vec)
{
    return sqrt_reference<float, 4>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 4>& vec)
{
    const float *data1 = reinterpret_cast<const float *>(&vec.val1);
    __os << "[" << data1[0] << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3] << "]";
    return __os;
}

}

#endif

#endif
