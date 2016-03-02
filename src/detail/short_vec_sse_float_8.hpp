/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_8_HPP

#ifdef __SSE__

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

#ifndef __AVX__
#ifndef __CUDA_ARCH__

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
class short_vec<float, 8>
{
public:
    static const int ARITY = 8;

    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 8>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm_set1_ps(data)),
        val2(_mm_set1_ps(data))
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(const __m128& val1, const __m128& val2) :
        val1(val1),
        val2(val2)
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
    short_vec(const sqrt_reference<float, 8>& other);

    inline
    void operator-=(const short_vec<float, 8>& other)
    {
        val1 = _mm_sub_ps(val1, other.val1);
        val2 = _mm_sub_ps(val2, other.val2);
    }

    inline
    short_vec<float, 8> operator-(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm_sub_ps(val1, other.val1),
            _mm_sub_ps(val2, other.val2));
    }

    inline
    void operator+=(const short_vec<float, 8>& other)
    {
        val1 = _mm_add_ps(val1, other.val1);
        val2 = _mm_add_ps(val2, other.val2);
    }

    inline
    short_vec<float, 8> operator+(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm_add_ps(val1, other.val1),
            _mm_add_ps(val2, other.val2));
    }

    inline
    void operator*=(const short_vec<float, 8>& other)
    {
        val1 = _mm_mul_ps(val1, other.val1);
        val2 = _mm_mul_ps(val2, other.val2);
    }

    inline
    short_vec<float, 8> operator*(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm_mul_ps(val1, other.val1),
            _mm_mul_ps(val2, other.val2));
    }

    inline
    void operator/=(const short_vec<float, 8>& other)
    {
        val1 = _mm_div_ps(val1, other.val1);
        val2 = _mm_div_ps(val2, other.val2);
    }

    inline
    void operator/=(const sqrt_reference<float, 8>& other);

    inline
    short_vec<float, 8> operator/(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm_div_ps(val1, other.val1),
            _mm_div_ps(val2, other.val2));
    }

    inline
    short_vec<float, 8> operator/(const sqrt_reference<float, 8>& other) const;

    inline
    short_vec<float, 8> sqrt() const
    {
        return short_vec<float, 8>(
            _mm_sqrt_ps(val1),
            _mm_sqrt_ps(val2));
    }

    inline
    void load(const float *data)
    {
        val1 = _mm_loadu_ps(data +  0);
        val2 = _mm_loadu_ps(data +  4);
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val1 = _mm_load_ps(data +  0);
        val2 = _mm_load_ps(data +  4);
    }

    inline
    void store(float *data) const
    {
        _mm_storeu_ps(data + 0, val1);
        _mm_storeu_ps(data + 4, val2);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_ps(data + 0, val1);
        _mm_store_ps(data + 4, val2);
    }

    inline
    void store_nt(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_ps(data + 0, val1);
        _mm_stream_ps(data + 4, val2);
    }

#ifdef __SSE4_1__
    inline
    void gather(const float *ptr, const int *offsets)
    {
        val1 = _mm_load_ss(ptr + offsets[0]);
        SHORTVEC_INSERT_PS(val1, ptr, offsets[1], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val1, ptr, offsets[2], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val1, ptr, offsets[3], _MM_MK_INSERTPS_NDX(0,3,0));
        val2 = _mm_load_ss(ptr + offsets[4]);
        SHORTVEC_INSERT_PS(val2, ptr, offsets[5], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS(val2, ptr, offsets[6], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS(val2, ptr, offsets[7], _MM_MK_INSERTPS_NDX(0,3,0));
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
   }
#endif

private:
    __m128 val1;
    __m128 val2;
};

inline
void operator<<(float *data, const short_vec<float, 8>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<float, 8>
{
public:
    template<typename OTHER_CARGO, int OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<float, 8>& vec) :
        vec(vec)
    {}

private:
    short_vec<float, 8> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<float, 8>::short_vec(const sqrt_reference<float, 8>& other) :
    val1(_mm_sqrt_ps(other.vec.val1)),
    val2(_mm_sqrt_ps(other.vec.val2))
{}

inline
void short_vec<float, 8>::operator/=(const sqrt_reference<float, 8>& other)
{
    val1 = _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1));
    val2 = _mm_mul_ps(val2, _mm_rsqrt_ps(other.vec.val2));
}

inline
short_vec<float, 8> short_vec<float, 8>::operator/(const sqrt_reference<float, 8>& other) const
{
    return short_vec<float, 8>(
        _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1)),
        _mm_mul_ps(val2, _mm_rsqrt_ps(other.vec.val2)));
}

inline
sqrt_reference<float, 8> sqrt(const short_vec<float, 8>& vec)
{
    return sqrt_reference<float, 8>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 8>& vec)
{
    const float *data1 = reinterpret_cast<const float *>(&vec.val1);
    const float *data2 = reinterpret_cast<const float *>(&vec.val2);
    __os << "[" << data1[0] << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3]  << ", " << data2[0]  << ", " << data2[1]  << ", " << data2[2]  << ", " << data2[3] << "]";
    return __os;
}

}

#endif
#endif
#endif

#endif
