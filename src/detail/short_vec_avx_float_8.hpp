/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_FLOAT_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_FLOAT_8_HPP

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
    typedef short_vec<float, 8> mask_type;
    typedef short_vec_strategy::avx strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 8>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm256_broadcast_ss(&data))
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(const __m256& val1) :
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
    short_vec(const sqrt_reference<float, 8>& other);

    inline
    bool any() const
    {
        // merge both 128-bit lanes of AVX register:
        __m128 buf0 = _mm_or_ps(
            _mm256_extractf128_ps(val1, 0),
            _mm256_extractf128_ps(val1, 1));
        // shuffle upper 64-bit half down to first 64 bits so we can
        // "or" both together:
        __m128 buf1 = _mm_shuffle_ps(buf0, buf0, (3 << 2) | (2 << 0));
        buf1 = _mm_or_ps(buf0, buf1);
        // another shuffle to extract 2nd least significant float
        // member and or it together with least significant float
        // member:
        __m128 buf2 = _mm_shuffle_ps(buf1, buf1, (1 << 0));
        return _mm_cvtss_f32(buf1) || _mm_cvtss_f32(buf2);
    }

    inline
    float get(int i) const
    {
        __m128 buf;
        if (i < 4) {
            buf = _mm256_extractf128_ps(val1, 0);
        } else {
            buf = _mm256_extractf128_ps(val1, 1);
        }

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
    void operator-=(const short_vec<float, 8>& other)
    {
        val1 = _mm256_sub_ps(val1, other.val1);
    }

    inline
    short_vec<float, 8> operator-(const short_vec<float, 8>& other) const
    {
        return _mm256_sub_ps(val1, other.val1);
    }

    inline
    void operator+=(const short_vec<float, 8>& other)
    {
        val1 = _mm256_add_ps(val1, other.val1);
    }

    inline
    short_vec<float, 8> operator+(const short_vec<float, 8>& other) const
    {
        return _mm256_add_ps(val1, other.val1);
    }

    inline
    void operator*=(const short_vec<float, 8>& other)
    {
        val1 = _mm256_mul_ps(val1, other.val1);
    }

    inline
    short_vec<float, 8> operator*(const short_vec<float, 8>& other) const
    {
        return _mm256_mul_ps(val1, other.val1);
    }

    inline
    void operator/=(const short_vec<float, 8>& other)
    {
        val1 = _mm256_mul_ps(val1, _mm256_rcp_ps(other.val1));
    }

    inline
    void operator/=(const sqrt_reference<float, 8>& other);

    inline
    short_vec<float, 8> operator/(const short_vec<float, 8>& other) const
    {
        return _mm256_mul_ps(val1, _mm256_rcp_ps(other.val1));
    }

    inline
    short_vec<float, 8> operator/(const sqrt_reference<float, 8>& other) const;

    inline
    short_vec<float, 8> operator<(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm256_cmp_ps(val1, other.val1, _CMP_LT_OS));
    }

    inline
    short_vec<float, 8> operator<=(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm256_cmp_ps(val1, other.val1, _CMP_LE_OS));
    }

    inline
    short_vec<float, 8> operator==(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm256_cmp_ps(val1, other.val1, _CMP_EQ_OQ));
    }

    inline
    short_vec<float, 8> operator>(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm256_cmp_ps(val1, other.val1, _CMP_GT_OS));
    }

    inline
    short_vec<float, 8> operator>=(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            _mm256_cmp_ps(val1, other.val1, _CMP_GE_OS));
    }

    inline
    short_vec<float, 8> sqrt() const
    {
        return _mm256_sqrt_ps(val1);
    }

    inline
    void load(const float *data)
    {
        val1 = _mm256_loadu_ps(data);
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        val1 = _mm256_load_ps(data);
    }

    inline
    void store(float *data) const
    {
        _mm256_storeu_ps(data, val1);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_store_ps(data, val1);
    }

    inline
    void store_nt(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_stream_ps(data, val1);
    }

#ifdef __AVX2__
    inline
    void gather(const float *ptr, const int *offsets)
    {
        __m256i indices;
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets));
        val1    = _mm256_i32gather_ps(ptr, indices, 4);
    }
#else
    inline
    void gather(const float *ptr, const int *offsets)
    {
        __m128 tmp;
        tmp  = _mm_load_ss(ptr + offsets[0]);
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[1], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[2], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[3], _MM_MK_INSERTPS_NDX(0,3,0));
        val1 = _mm256_insertf128_ps(val1, tmp, 0);
        tmp  = _mm_load_ss(ptr + offsets[4]);
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[5], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[6], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[7], _MM_MK_INSERTPS_NDX(0,3,0));
        val1 = _mm256_insertf128_ps(val1, tmp, 1);
    }
#endif

    inline
    void scatter(float *ptr, const int *offsets) const
    {
        __m128 tmp;
        tmp = _mm256_extractf128_ps(val1, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[0]], tmp, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[1]], tmp, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[2]], tmp, 2);
        _MM_EXTRACT_FLOAT(ptr[offsets[3]], tmp, 3);
        tmp = _mm256_extractf128_ps(val1, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[4]], tmp, 0);
        _MM_EXTRACT_FLOAT(ptr[offsets[5]], tmp, 1);
        _MM_EXTRACT_FLOAT(ptr[offsets[6]], tmp, 2);
        _MM_EXTRACT_FLOAT(ptr[offsets[7]], tmp, 3);
    }

private:
    __m256 val1;
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
    val1(_mm256_sqrt_ps(other.vec.val1))
{}

inline
void short_vec<float, 8>::operator/=(const sqrt_reference<float, 8>& other)
{
    val1 = _mm256_mul_ps(val1, _mm256_rsqrt_ps(other.vec.val1));
}

inline
short_vec<float, 8> short_vec<float, 8>::operator/(const sqrt_reference<float, 8>& other) const
{
    return _mm256_mul_ps(val1, _mm256_rsqrt_ps(other.vec.val1));
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
    __os << "[" << data1[0] << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3]  << ", " << data1[4]  << ", " << data1[5]  << ", " << data1[6]  << ", " << data1[7] << "]";
    return __os;
}

}

#endif

#endif
