/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_16_HPP

#ifdef __SSE__

#include <emmintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#ifndef __AVX512F__
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
class short_vec<float, 16>
{
public:
    static const int ARITY = 16;

    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 16>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm_set1_ps(data)),
        val2(_mm_set1_ps(data)),
        val3(_mm_set1_ps(data)),
        val4(_mm_set1_ps(data))
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(const __m128& val1, const __m128& val2, const __m128& val3, const __m128& val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
    {}

    inline
    short_vec(const sqrt_reference<float, 16>& other);

    inline
    void operator-=(const short_vec<float, 16>& other)
    {
        val1 = _mm_sub_ps(val1, other.val1);
        val2 = _mm_sub_ps(val2, other.val2);
        val3 = _mm_sub_ps(val3, other.val3);
        val4 = _mm_sub_ps(val4, other.val4);
    }

    inline
    short_vec<float, 16> operator-(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_sub_ps(val1, other.val1),
            _mm_sub_ps(val2, other.val2),
            _mm_sub_ps(val3, other.val3),
            _mm_sub_ps(val4, other.val4));
    }

    inline
    void operator+=(const short_vec<float, 16>& other)
    {
        val1 = _mm_add_ps(val1, other.val1);
        val2 = _mm_add_ps(val2, other.val2);
        val3 = _mm_add_ps(val3, other.val3);
        val4 = _mm_add_ps(val4, other.val4);
    }

    inline
    short_vec<float, 16> operator+(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_add_ps(val1, other.val1),
            _mm_add_ps(val2, other.val2),
            _mm_add_ps(val3, other.val3),
            _mm_add_ps(val4, other.val4));
    }

    inline
    void operator*=(const short_vec<float, 16>& other)
    {
        val1 = _mm_mul_ps(val1, other.val1);
        val2 = _mm_mul_ps(val2, other.val2);
        val3 = _mm_mul_ps(val3, other.val3);
        val4 = _mm_mul_ps(val4, other.val4);
    }

    inline
    short_vec<float, 16> operator*(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_mul_ps(val1, other.val1),
            _mm_mul_ps(val2, other.val2),
            _mm_mul_ps(val3, other.val3),
            _mm_mul_ps(val4, other.val4));
    }

    inline
    void operator/=(const short_vec<float, 16>& other)
    {
        val1 = _mm_div_ps(val1, other.val1);
        val2 = _mm_div_ps(val2, other.val2);
        val3 = _mm_div_ps(val3, other.val3);
        val4 = _mm_div_ps(val4, other.val4);
    }

    inline
    void operator/=(const sqrt_reference<float, 16>& other);

    inline
    short_vec<float, 16> operator/(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_div_ps(val1, other.val1),
            _mm_div_ps(val2, other.val2),
            _mm_div_ps(val3, other.val3),
            _mm_div_ps(val4, other.val4));
    }

    inline
    short_vec<float, 16> operator/(const sqrt_reference<float, 16>& other) const;

    inline
    short_vec<float, 16> sqrt() const
    {
        return short_vec<float, 16>(
            _mm_sqrt_ps(val1),
            _mm_sqrt_ps(val2),
            _mm_sqrt_ps(val3),
            _mm_sqrt_ps(val4));
    }

    inline
    void load(const float *data)
    {
        val1 = _mm_loadu_ps(data +  0);
        val2 = _mm_loadu_ps(data +  4);
        val3 = _mm_loadu_ps(data +  8);
        val4 = _mm_loadu_ps(data + 12);
    }

    inline
    void loadAligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val1 = _mm_load_ps(data +  0);
        val2 = _mm_load_ps(data +  4);
        val3 = _mm_load_ps(data +  8);
        val4 = _mm_load_ps(data + 12);
    }

    inline
    void store(float *data) const
    {
        _mm_storeu_ps(data +  0, val1);
        _mm_storeu_ps(data +  4, val2);
        _mm_storeu_ps(data +  8, val3);
        _mm_storeu_ps(data + 12, val4);
    }

    inline
    void storeAligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_ps(data +  0, val1);
        _mm_store_ps(data +  4, val2);
        _mm_store_ps(data +  8, val3);
        _mm_store_ps(data + 12, val4);
    }

    inline
    void storeNT(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_ps(data +  0, val1);
        _mm_stream_ps(data +  4, val2);
        _mm_stream_ps(data +  8, val3);
        _mm_stream_ps(data + 12, val4);
    }

#ifdef __SSE4_1__
    inline
    void gather(const float *ptr, const unsigned *offsets)
    {
        val1 = _mm_load_ss(ptr + offsets[0]);
        ShortVecHelpers::_mm_insert_ps2(val1, ptr, offsets[ 1], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2(val1, ptr, offsets[ 2], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2(val1, ptr, offsets[ 3], _MM_MK_INSERTPS_NDX(0,3,0));
        val2 = _mm_load_ss(ptr + offsets[4]);
        ShortVecHelpers::_mm_insert_ps2(val2, ptr, offsets[ 5], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2(val2, ptr, offsets[ 6], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2(val2, ptr, offsets[ 7], _MM_MK_INSERTPS_NDX(0,3,0));
        val3 = _mm_load_ss(ptr + offsets[8]);
        ShortVecHelpers::_mm_insert_ps2(val3, ptr, offsets[ 9], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2(val3, ptr, offsets[10], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2(val3, ptr, offsets[11], _MM_MK_INSERTPS_NDX(0,3,0));
        val4 = _mm_load_ss(ptr + offsets[12]);
        ShortVecHelpers::_mm_insert_ps2(val4, ptr, offsets[13], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2(val4, ptr, offsets[14], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2(val4, ptr, offsets[15], _MM_MK_INSERTPS_NDX(0,3,0));
    }

    inline
    void scatter(float *ptr, const unsigned *offsets) const
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
    }
#else
    inline
    void gather(const float *ptr, const unsigned *offsets)
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
    }

    inline
    void scatter(float *ptr, const unsigned *offsets) const
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
   }
#endif

private:
    __m128 val1;
    __m128 val2;
    __m128 val3;
    __m128 val4;
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
    template<typename OTHER_CARGO, int OTHER_ARITY>
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
    val1(_mm_sqrt_ps(other.vec.val1)),
    val2(_mm_sqrt_ps(other.vec.val2)),
    val3(_mm_sqrt_ps(other.vec.val3)),
    val4(_mm_sqrt_ps(other.vec.val4))
{}

inline
void short_vec<float, 16>::operator/=(const sqrt_reference<float, 16>& other)
{
    val1 = _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1));
    val2 = _mm_mul_ps(val2, _mm_rsqrt_ps(other.vec.val2));
    val3 = _mm_mul_ps(val3, _mm_rsqrt_ps(other.vec.val3));
    val4 = _mm_mul_ps(val4, _mm_rsqrt_ps(other.vec.val4));
}

inline
short_vec<float, 16> short_vec<float, 16>::operator/(const sqrt_reference<float, 16>& other) const
{
    return short_vec<float, 16>(
        _mm_mul_ps(val1, _mm_rsqrt_ps(other.vec.val1)),
        _mm_mul_ps(val2, _mm_rsqrt_ps(other.vec.val2)),
        _mm_mul_ps(val3, _mm_rsqrt_ps(other.vec.val3)),
        _mm_mul_ps(val4, _mm_rsqrt_ps(other.vec.val4)));
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
    const float *data2 = reinterpret_cast<const float *>(&vec.val2);
    const float *data3 = reinterpret_cast<const float *>(&vec.val3);
    const float *data4 = reinterpret_cast<const float *>(&vec.val4);
    __os << "["
         << data1[0] << ", " << data1[1]  << ", " << data1[2] << ", " << data1[3] << ", "
         << data2[0] << ", " << data2[1]  << ", " << data2[2] << ", " << data2[3] << ", "
         << data3[0] << ", " << data3[1]  << ", " << data3[2] << ", " << data3[3] << ", "
         << data4[0] << ", " << data4[1]  << ", " << data4[2] << ", " << data4[3] << "]";
    return __os;
}

}

#endif
#endif
#endif
#endif

#endif
