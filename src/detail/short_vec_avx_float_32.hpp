/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_FLOAT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_FLOAT_32_HPP

#ifdef __AVX__

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>

#ifndef __AVX512F__
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
class short_vec<float, 32>
{
public:
    static const int ARITY = 32;

    typedef short_vec_strategy::avx strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 32>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm256_broadcast_ss(&data)),
        val2(_mm256_broadcast_ss(&data)),
        val3(_mm256_broadcast_ss(&data)),
        val4(_mm256_broadcast_ss(&data))
    {}

    inline
    short_vec(const float *data) :
        val1(_mm256_loadu_ps(data + 0)),
        val2(_mm256_loadu_ps(data + 8)),
        val3(_mm256_loadu_ps(data + 16)),
        val4(_mm256_loadu_ps(data + 24))
    {}

    inline
    short_vec(const __m256& val1, const __m256& val2, const __m256& val3, const __m256& val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
    {}

    inline
    short_vec(const sqrt_reference<float, 32>& other);

    inline
    void operator-=(const short_vec<float, 32>& other)
    {
        val1 = _mm256_sub_ps(val1, other.val1);
        val2 = _mm256_sub_ps(val2, other.val2);
        val3 = _mm256_sub_ps(val3, other.val3);
        val4 = _mm256_sub_ps(val4, other.val4);
    }

    inline
    short_vec<float, 32> operator-(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm256_sub_ps(val1, other.val1),
            _mm256_sub_ps(val2, other.val2),
            _mm256_sub_ps(val3, other.val3),
            _mm256_sub_ps(val4, other.val4));
    }

    inline
    void operator+=(const short_vec<float, 32>& other)
    {
        val1 = _mm256_add_ps(val1, other.val1);
        val2 = _mm256_add_ps(val2, other.val2);
        val3 = _mm256_add_ps(val3, other.val3);
        val4 = _mm256_add_ps(val4, other.val4);
    }

    inline
    short_vec<float, 32> operator+(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm256_add_ps(val1, other.val1),
            _mm256_add_ps(val2, other.val2),
            _mm256_add_ps(val3, other.val3),
            _mm256_add_ps(val4, other.val4));
    }

    inline
    void operator*=(const short_vec<float, 32>& other)
    {
        val1 = _mm256_mul_ps(val1, other.val1);
        val2 = _mm256_mul_ps(val2, other.val2);
        val3 = _mm256_mul_ps(val3, other.val3);
        val4 = _mm256_mul_ps(val4, other.val4);
    }

    inline
    short_vec<float, 32> operator*(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm256_mul_ps(val1, other.val1),
            _mm256_mul_ps(val2, other.val2),
            _mm256_mul_ps(val3, other.val3),
            _mm256_mul_ps(val4, other.val4));
    }

    inline
    void operator/=(const short_vec<float, 32>& other)
    {
        val1 = _mm256_mul_ps(val1, _mm256_rcp_ps(other.val1));
        val2 = _mm256_mul_ps(val2, _mm256_rcp_ps(other.val2));
        val3 = _mm256_mul_ps(val3, _mm256_rcp_ps(other.val3));
        val4 = _mm256_mul_ps(val4, _mm256_rcp_ps(other.val4));
    }

    inline
    void operator/=(const sqrt_reference<float, 32>& other);

    inline
    short_vec<float, 32> operator/(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm256_mul_ps(val1, _mm256_rcp_ps(other.val1)),
            _mm256_mul_ps(val2, _mm256_rcp_ps(other.val2)),
            _mm256_mul_ps(val3, _mm256_rcp_ps(other.val3)),
            _mm256_mul_ps(val4, _mm256_rcp_ps(other.val4)));
    }

    inline
    short_vec<float, 32> operator/(const sqrt_reference<float, 32>& other) const;

    inline
    short_vec<float, 32> sqrt() const
    {
        return short_vec<float, 32>(
            _mm256_sqrt_ps(val1),
            _mm256_sqrt_ps(val2),
            _mm256_sqrt_ps(val3),
            _mm256_sqrt_ps(val4));
    }

    inline
    void store(float *data) const
    {
        _mm256_storeu_ps(data +  0, val1);
        _mm256_storeu_ps(data +  8, val2);
        _mm256_storeu_ps(data + 16, val3);
        _mm256_storeu_ps(data + 24, val4);
    }

#ifdef __AVX2__
    inline
    void gather(const float *ptr, const unsigned *offsets)
    {
        __m256i indices;
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets));
        val1    = _mm256_i32gather_ps(ptr, indices, 4);
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets + 8));
        val2    = _mm256_i32gather_ps(ptr, indices, 4);
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets + 16));
        val3    = _mm256_i32gather_ps(ptr, indices, 4);
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets + 24));
        val4    = _mm256_i32gather_ps(ptr, indices, 4);
    }
#else
    inline
    void gather(const float *ptr, const unsigned *offsets)
    {
        __m128 tmp;
        tmp  = _mm_load_ss(ptr + offsets[0]);
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[1], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[2], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[3], _MM_MK_INSERTPS_NDX(0,3,0));
        val1 = _mm256_insertf128_ps(val1, tmp, 0);
        tmp  = _mm_load_ss(ptr + offsets[4]);
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[5], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[6], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[7], _MM_MK_INSERTPS_NDX(0,3,0));
        val1 = _mm256_insertf128_ps(val1, tmp, 1);
        tmp  = _mm_load_ss(ptr + offsets[8]);
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[ 9], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[10], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[11], _MM_MK_INSERTPS_NDX(0,3,0));
        val2 = _mm256_insertf128_ps(val2, tmp, 0);
        tmp  = _mm_load_ss(ptr + offsets[12]);
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[13], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[14], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[15], _MM_MK_INSERTPS_NDX(0,3,0));
        val2 = _mm256_insertf128_ps(val2, tmp, 1);
        tmp  = _mm_load_ss(ptr + offsets[16]);
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[17], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[18], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[19], _MM_MK_INSERTPS_NDX(0,3,0));
        val3 = _mm256_insertf128_ps(val3, tmp, 0);
        tmp  = _mm_load_ss(ptr + offsets[20]);
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[21], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[22], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[23], _MM_MK_INSERTPS_NDX(0,3,0));
        val3 = _mm256_insertf128_ps(val3, tmp, 1);
        tmp  = _mm_load_ss(ptr + offsets[24]);
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[25], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[26], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[27], _MM_MK_INSERTPS_NDX(0,3,0));
        val4 = _mm256_insertf128_ps(val4, tmp, 0);
        tmp  = _mm_load_ss(ptr + offsets[28]);
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[29], _MM_MK_INSERTPS_NDX(0,1,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[30], _MM_MK_INSERTPS_NDX(0,2,0));
        ShortVecHelpers::_mm_insert_ps2_avx(tmp, ptr, offsets[31], _MM_MK_INSERTPS_NDX(0,3,0));
        val4 = _mm256_insertf128_ps(val4, tmp, 1);
    }
#endif

    inline
    void scatter(float *ptr, const unsigned *offsets) const
    {
        __m128 tmp;
        tmp = _mm256_extractf128_ps(val1, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[ 0]], tmp, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[ 1]], tmp, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[ 2]], tmp, 2);
        _MM_EXTRACT_FLOAT(ptr[offsets[ 3]], tmp, 3);
        tmp = _mm256_extractf128_ps(val1, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[ 4]], tmp, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[ 5]], tmp, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[ 6]], tmp, 2);
        _MM_EXTRACT_FLOAT(ptr[offsets[ 7]], tmp, 3);
        tmp = _mm256_extractf128_ps(val2, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[ 8]], tmp, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[ 9]], tmp, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[10]], tmp, 2);
        _MM_EXTRACT_FLOAT(ptr[offsets[11]], tmp, 3);
        tmp = _mm256_extractf128_ps(val2, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[12]], tmp, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[13]], tmp, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[14]], tmp, 2);
        _MM_EXTRACT_FLOAT(ptr[offsets[15]], tmp, 3);
        tmp = _mm256_extractf128_ps(val3, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[16]], tmp, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[17]], tmp, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[18]], tmp, 2);
        _MM_EXTRACT_FLOAT(ptr[offsets[19]], tmp, 3);
        tmp = _mm256_extractf128_ps(val3, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[20]], tmp, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[21]], tmp, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[22]], tmp, 2);
        _MM_EXTRACT_FLOAT(ptr[offsets[23]], tmp, 3);
        tmp = _mm256_extractf128_ps(val4, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[24]], tmp, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[25]], tmp, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[26]], tmp, 2);
        _MM_EXTRACT_FLOAT(ptr[offsets[27]], tmp, 3);
        tmp = _mm256_extractf128_ps(val4, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[28]], tmp, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[29]], tmp, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[30]], tmp, 2);
        _MM_EXTRACT_FLOAT(ptr[offsets[31]], tmp, 3);
    }

private:
    __m256 val1;
    __m256 val2;
    __m256 val3;
    __m256 val4;
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
    val1(_mm256_sqrt_ps(other.vec.val1)),
    val2(_mm256_sqrt_ps(other.vec.val2)),
    val3(_mm256_sqrt_ps(other.vec.val3)),
    val4(_mm256_sqrt_ps(other.vec.val4))
{}

inline
void short_vec<float, 32>::operator/=(const sqrt_reference<float, 32>& other)
{
    val1 = _mm256_mul_ps(val1, _mm256_rsqrt_ps(other.vec.val1));
    val2 = _mm256_mul_ps(val2, _mm256_rsqrt_ps(other.vec.val2));
    val3 = _mm256_mul_ps(val3, _mm256_rsqrt_ps(other.vec.val3));
    val4 = _mm256_mul_ps(val4, _mm256_rsqrt_ps(other.vec.val4));
}

inline
short_vec<float, 32> short_vec<float, 32>::operator/(const sqrt_reference<float, 32>& other) const
{
    return short_vec<float, 32>(
        _mm256_mul_ps(val1, _mm256_rsqrt_ps(other.vec.val1)),
        _mm256_mul_ps(val2, _mm256_rsqrt_ps(other.vec.val2)),
        _mm256_mul_ps(val3, _mm256_rsqrt_ps(other.vec.val3)),
        _mm256_mul_ps(val4, _mm256_rsqrt_ps(other.vec.val4)));
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

    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << ", " << data1[4] << ", " << data1[5] << ", " << data1[6] << ", " << data1[7]
         << ", " << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
         << ", " << data2[4] << ", " << data2[5] << ", " << data2[6] << ", " << data2[7]
         << ", " << data3[0] << ", " << data3[1] << ", " << data3[2] << ", " << data3[3]
         << ", " << data3[4] << ", " << data3[5] << ", " << data3[6] << ", " << data3[7]
         << ", " << data4[0] << ", " << data4[1] << ", " << data4[2] << ", " << data4[3]
         << ", " << data4[4] << ", " << data4[5] << ", " << data4[6] << ", " << data4[7]
         << "]";
    return __os;
}

}

#endif
#endif
#endif

#endif
