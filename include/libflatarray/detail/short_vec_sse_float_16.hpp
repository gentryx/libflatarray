/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SSE_FLOAT_16_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE2) ||            \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SSE4_1)

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
class short_vec<float, 16> : public short_vec_base<float, 16>
{
public:
    static const std::size_t ARITY = 16;
    typedef short_vec<float, 16> mask_type;
    typedef short_vec_strategy::sse strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 16>& vec);

    inline
    short_vec(const float data = 0) :
        val{_mm_set1_ps(data),
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
        const __m128& val4) :
        val{val1,
            val2,
            val3,
            val4}
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
        __m128 buf1 = _mm_or_ps(
            _mm_or_ps(val[ 0], val[ 1]),
            _mm_or_ps(val[ 2], val[ 3]));

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
    void operator-=(const short_vec<float, 16>& other)
    {
        val[ 0] = _mm_sub_ps(val[ 0], other.val[ 0]);
        val[ 1] = _mm_sub_ps(val[ 1], other.val[ 1]);
        val[ 2] = _mm_sub_ps(val[ 2], other.val[ 2]);
        val[ 3] = _mm_sub_ps(val[ 3], other.val[ 3]);
    }

    inline
    short_vec<float, 16> operator-(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_sub_ps(val[ 0], other.val[ 0]),
            _mm_sub_ps(val[ 1], other.val[ 1]),
            _mm_sub_ps(val[ 2], other.val[ 2]),
            _mm_sub_ps(val[ 3], other.val[ 3]));
    }

    inline
    void operator+=(const short_vec<float, 16>& other)
    {
        val[ 0] = _mm_add_ps(val[ 0], other.val[ 0]);
        val[ 1] = _mm_add_ps(val[ 1], other.val[ 1]);
        val[ 2] = _mm_add_ps(val[ 2], other.val[ 2]);
        val[ 3] = _mm_add_ps(val[ 3], other.val[ 3]);
    }

    inline
    short_vec<float, 16> operator+(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_add_ps(val[ 0], other.val[ 0]),
            _mm_add_ps(val[ 1], other.val[ 1]),
            _mm_add_ps(val[ 2], other.val[ 2]),
            _mm_add_ps(val[ 3], other.val[ 3]));
    }

    inline
    void operator*=(const short_vec<float, 16>& other)
    {
        val[ 0] = _mm_mul_ps(val[ 0], other.val[ 0]);
        val[ 1] = _mm_mul_ps(val[ 1], other.val[ 1]);
        val[ 2] = _mm_mul_ps(val[ 2], other.val[ 2]);
        val[ 3] = _mm_mul_ps(val[ 3], other.val[ 3]);
    }

    inline
    short_vec<float, 16> operator*(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_mul_ps(val[ 0], other.val[ 0]),
            _mm_mul_ps(val[ 1], other.val[ 1]),
            _mm_mul_ps(val[ 2], other.val[ 2]),
            _mm_mul_ps(val[ 3], other.val[ 3]));
    }

    inline
    void operator/=(const short_vec<float, 16>& other)
    {
        val[ 0] = _mm_div_ps(val[ 0], other.val[ 0]);
        val[ 1] = _mm_div_ps(val[ 1], other.val[ 1]);
        val[ 2] = _mm_div_ps(val[ 2], other.val[ 2]);
        val[ 3] = _mm_div_ps(val[ 3], other.val[ 3]);
    }

    inline
    void operator/=(const sqrt_reference<float, 16>& other);

    inline
    short_vec<float, 16> operator/(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_div_ps(val[ 0], other.val[ 0]),
            _mm_div_ps(val[ 1], other.val[ 1]),
            _mm_div_ps(val[ 2], other.val[ 2]),
            _mm_div_ps(val[ 3], other.val[ 3]));
    }

    inline
    short_vec<float, 16> operator/(const sqrt_reference<float, 16>& other) const;

    inline
    short_vec<float, 16> operator<(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_cmplt_ps(val[ 0], other.val[ 0]),
            _mm_cmplt_ps(val[ 1], other.val[ 1]),
            _mm_cmplt_ps(val[ 2], other.val[ 2]),
            _mm_cmplt_ps(val[ 3], other.val[ 3]));
    }

    inline
    short_vec<float, 16> operator<=(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_cmple_ps(val[ 0], other.val[ 0]),
            _mm_cmple_ps(val[ 1], other.val[ 1]),
            _mm_cmple_ps(val[ 2], other.val[ 2]),
            _mm_cmple_ps(val[ 3], other.val[ 3]));
    }

    inline
    short_vec<float, 16> operator==(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_cmpeq_ps(val[ 0], other.val[ 0]),
            _mm_cmpeq_ps(val[ 1], other.val[ 1]),
            _mm_cmpeq_ps(val[ 2], other.val[ 2]),
            _mm_cmpeq_ps(val[ 3], other.val[ 3]));
    }

    inline
    short_vec<float, 16> operator>(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_cmpgt_ps(val[ 0], other.val[ 0]),
            _mm_cmpgt_ps(val[ 1], other.val[ 1]),
            _mm_cmpgt_ps(val[ 2], other.val[ 2]),
            _mm_cmpgt_ps(val[ 3], other.val[ 3]));
    }

    inline
    short_vec<float, 16> operator>=(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm_cmpge_ps(val[ 0], other.val[ 0]),
            _mm_cmpge_ps(val[ 1], other.val[ 1]),
            _mm_cmpge_ps(val[ 2], other.val[ 2]),
            _mm_cmpge_ps(val[ 3], other.val[ 3]));
    }

    inline
    short_vec<float, 16> sqrt() const
    {
        return short_vec<float, 16>(
            _mm_sqrt_ps(val[ 0]),
            _mm_sqrt_ps(val[ 1]),
            _mm_sqrt_ps(val[ 2]),
            _mm_sqrt_ps(val[ 3]));
    }

    inline
    void load(const float *data)
    {
        val[ 0] = _mm_loadu_ps(data +  0);
        val[ 1] = _mm_loadu_ps(data +  4);
        val[ 2] = _mm_loadu_ps(data +  8);
        val[ 3] = _mm_loadu_ps(data + 12);
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        val[ 0] = _mm_load_ps(data +  0);
        val[ 1] = _mm_load_ps(data +  4);
        val[ 2] = _mm_load_ps(data +  8);
        val[ 3] = _mm_load_ps(data + 12);
    }

    inline
    void store(float *data) const
    {
        _mm_storeu_ps(data +  0, val[ 0]);
        _mm_storeu_ps(data +  4, val[ 1]);
        _mm_storeu_ps(data +  8, val[ 2]);
        _mm_storeu_ps(data + 12, val[ 3]);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_store_ps(data +  0, val[ 0]);
        _mm_store_ps(data +  4, val[ 1]);
        _mm_store_ps(data +  8, val[ 2]);
        _mm_store_ps(data + 12, val[ 3]);
    }

    inline
    void store_nt(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        _mm_stream_ps(data +  0, val[ 0]);
        _mm_stream_ps(data +  4, val[ 1]);
        _mm_stream_ps(data +  8, val[ 2]);
        _mm_stream_ps(data + 12, val[ 3]);
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
   }
#endif

    inline
    void blend(const mask_type& mask, const short_vec<float, 16>& other)
    {
#ifdef __SSE4_1__
        val[ 0] = _mm_blendv_ps(val[ 0], other.val[ 0], mask.val[ 0]);
        val[ 1] = _mm_blendv_ps(val[ 1], other.val[ 1], mask.val[ 1]);
        val[ 2] = _mm_blendv_ps(val[ 2], other.val[ 2], mask.val[ 2]);
        val[ 3] = _mm_blendv_ps(val[ 3], other.val[ 3], mask.val[ 3]);
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
#endif
    }

private:
    __m128 val[ 0];
    __m128 val[ 1];
    __m128 val[ 2];
    __m128 val[ 3];
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
    val[ 0](_mm_sqrt_ps(other.vec.val[ 0])),
    val[ 1](_mm_sqrt_ps(other.vec.val[ 1])),
    val[ 2](_mm_sqrt_ps(other.vec.val[ 2])),
    val[ 3](_mm_sqrt_ps(other.vec.val[ 3]))
{}

inline
void short_vec<float, 16>::operator/=(const sqrt_reference<float, 16>& other)
{
    val[ 0] = _mm_mul_ps(val[ 0], _mm_rsqrt_ps(other.vec.val[ 0]));
    val[ 1] = _mm_mul_ps(val[ 1], _mm_rsqrt_ps(other.vec.val[ 1]));
    val[ 2] = _mm_mul_ps(val[ 2], _mm_rsqrt_ps(other.vec.val[ 2]));
    val[ 3] = _mm_mul_ps(val[ 3], _mm_rsqrt_ps(other.vec.val[ 3]));
}

inline
short_vec<float, 16> short_vec<float, 16>::operator/(const sqrt_reference<float, 16>& other) const
{
    return short_vec<float, 16>(
        _mm_mul_ps(val[ 0], _mm_rsqrt_ps(other.vec.val[ 0])),
        _mm_mul_ps(val[ 1], _mm_rsqrt_ps(other.vec.val[ 1])),
        _mm_mul_ps(val[ 2], _mm_rsqrt_ps(other.vec.val[ 2])),
        _mm_mul_ps(val[ 3], _mm_rsqrt_ps(other.vec.val[ 3])));
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
    const float *data1 = reinterpret_cast<const float *>(&vec.val[ 0]!);
    const float *data2 = reinterpret_cast<const float *>(&vec.val[ 1]);
    const float *data3 = reinterpret_cast<const float *>(&vec.val[ 2]);
    const float *data4 = reinterpret_cast<const float *>(&vec.val[ 3]);
    __os << "["
         << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3] << ", "
         << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3] << ", "
         << data3[0] << ", " << data3[1] << ", " << data3[2] << ", " << data3[3] << ", "
         << data4[0] << ", " << data4[1] << ", " << data4[2] << ", " << data4[3] << "]";
    return __os;
}

}

#endif

#endif
