/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_FLOAT_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX_FLOAT_16_HPP

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
    typedef short_vec<float, 16> mask_type;
    typedef short_vec_strategy::avx strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 16>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm256_broadcast_ss(&data)),
        val2(_mm256_broadcast_ss(&data))
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(const __m256& val1, const __m256& val2) :
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
    short_vec(const sqrt_reference<float, 16>& other);

    inline
    bool any() const
    {
        __m256 buf0 = _mm256_or_ps(val1, val2);
        return (0 == _mm256_testz_si256(
                    _mm256_castps_si256(buf0),
                    _mm256_castps_si256(buf0)));
    }

    inline
    float get(int i) const
    {
        __m256 buf0;
        if (i < 8) {
            buf0 = val1;
        } else {
            buf0 = val2;
        }

        i &= 7;

        __m128 buf1;
        if (i < 4) {
            buf1 = _mm256_extractf128_ps(buf0, 0);
        } else {
            buf1 = _mm256_extractf128_ps(buf0, 1);
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
    void operator-=(const short_vec<float, 16>& other)
    {
        val1 = _mm256_sub_ps(val1, other.val1);
        val2 = _mm256_sub_ps(val2, other.val2);
    }

    inline
    short_vec<float, 16> operator-(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm256_sub_ps(val1, other.val1),
            _mm256_sub_ps(val2, other.val2));
    }

    inline
    void operator+=(const short_vec<float, 16>& other)
    {
        val1 = _mm256_add_ps(val1, other.val1);
        val2 = _mm256_add_ps(val2, other.val2);
    }

    inline
    short_vec<float, 16> operator+(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm256_add_ps(val1, other.val1),
            _mm256_add_ps(val2, other.val2));
    }

    inline
    void operator*=(const short_vec<float, 16>& other)
    {
        val1 = _mm256_mul_ps(val1, other.val1);
        val2 = _mm256_mul_ps(val2, other.val2);
    }

    inline
    short_vec<float, 16> operator*(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm256_mul_ps(val1, other.val1),
            _mm256_mul_ps(val2, other.val2));
    }

    inline
    void operator/=(const short_vec<float, 16>& other)
    {
        val1 = _mm256_mul_ps(val1, _mm256_rcp_ps(other.val1));
        val2 = _mm256_mul_ps(val2, _mm256_rcp_ps(other.val2));
    }

    inline
    void operator/=(const sqrt_reference<float, 16>& other);

    inline
    short_vec<float, 16> operator/(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm256_mul_ps(val1, _mm256_rcp_ps(other.val1)),
            _mm256_mul_ps(val2, _mm256_rcp_ps(other.val2)));
    }

    inline
    short_vec<float, 16> operator/(const sqrt_reference<float, 16>& other) const;

    inline
    short_vec<float, 16> operator<(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm256_cmp_ps(val1, other.val1, _CMP_LT_OS),
            _mm256_cmp_ps(val2, other.val2, _CMP_LT_OS));
    }

    inline
    short_vec<float, 16> operator<=(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm256_cmp_ps(val1, other.val1, _CMP_LE_OS),
            _mm256_cmp_ps(val2, other.val2, _CMP_LE_OS));
    }

    inline
    short_vec<float, 16> operator==(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm256_cmp_ps(val1, other.val1, _CMP_EQ_OQ),
            _mm256_cmp_ps(val2, other.val2, _CMP_EQ_OQ));
    }

    inline
    short_vec<float, 16> operator>(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm256_cmp_ps(val1, other.val1, _CMP_GT_OS),
            _mm256_cmp_ps(val2, other.val2, _CMP_GT_OS));
    }

    inline
    short_vec<float, 16> operator>=(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm256_cmp_ps(val1, other.val1, _CMP_GE_OS),
            _mm256_cmp_ps(val2, other.val2, _CMP_GE_OS));
    }

    inline
    short_vec<float, 16> sqrt() const
    {
        return short_vec<float, 16>(
            _mm256_sqrt_ps(val1),
            _mm256_sqrt_ps(val2));
    }

    inline
    void load(const float *data)
    {
        val1 = _mm256_loadu_ps(data + 0);
        val2 = _mm256_loadu_ps(data + 8);
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        val1 = _mm256_load_ps(data + 0);
        val2 = _mm256_load_ps(data + 8);
    }

    inline
    void store(float *data) const
    {
        _mm256_storeu_ps(data + 0, val1);
        _mm256_storeu_ps(data + 8, val2);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_store_ps(data + 0, val1);
        _mm256_store_ps(data + 8, val2);
    }

    inline
    void store_nt(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        _mm256_stream_ps(data + 0, val1);
        _mm256_stream_ps(data + 8, val2);
    }

#ifdef __AVX2__
    inline
    void gather(const float *ptr, const int *offsets)
    {
        __m256i indices;
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets));
        val1    = _mm256_i32gather_ps(ptr, indices, 4);
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets + 8));
        val2    = _mm256_i32gather_ps(ptr, indices, 4);
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
        tmp  = _mm_load_ss(ptr + offsets[8]);
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[ 9], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[10], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[11], _MM_MK_INSERTPS_NDX(0,3,0));
        val2 = _mm256_insertf128_ps(val2, tmp, 0);
        tmp  = _mm_load_ss(ptr + offsets[12]);
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[13], _MM_MK_INSERTPS_NDX(0,1,0));
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[14], _MM_MK_INSERTPS_NDX(0,2,0));
        SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[15], _MM_MK_INSERTPS_NDX(0,3,0));
        val2 = _mm256_insertf128_ps(val2, tmp, 1);
    }
#endif

    inline
    void scatter(float *ptr, const int *offsets) const
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
    }

    inline
    void blend(const mask_type& mask, const short_vec<float, 16>& other)
    {
        val1  = _mm256_blendv_ps(val1,  other.val1,  mask.val1);
        val2  = _mm256_blendv_ps(val2,  other.val2,  mask.val2);
    }

private:
    __m256 val1;
    __m256 val2;
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
    val1(_mm256_sqrt_ps(other.vec.val1)),
    val2(_mm256_sqrt_ps(other.vec.val2))
{}

inline
void short_vec<float, 16>::operator/=(const sqrt_reference<float, 16>& other)
{
    val1 = _mm256_mul_ps(val1, _mm256_rsqrt_ps(other.vec.val1));
    val2 = _mm256_mul_ps(val2, _mm256_rsqrt_ps(other.vec.val2));
}

inline
short_vec<float, 16> short_vec<float, 16>::operator/(const sqrt_reference<float, 16>& other) const
{
    return short_vec<float, 16>(
        _mm256_mul_ps(val1, _mm256_rsqrt_ps(other.vec.val1)),
        _mm256_mul_ps(val2, _mm256_rsqrt_ps(other.vec.val2)));
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
    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << ", " << data1[4] << ", " << data1[5] << ", " << data1[6] << ", " << data1[7]
         << ", " << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
         << ", " << data2[4] << ", " << data2[5] << ", " << data2[6] << ", " << data2[7]
         << "]";
    return __os;
}

}

#endif

#endif
