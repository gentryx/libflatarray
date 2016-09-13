/**
 * Copyright 2015 Di Xiao
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_16_HPP

#if LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_ARM_NEON

#include <arm_neon.h>
#include <libflatarray/config.h>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <iostream>

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

namespace LibFlatArray {

template<>
class short_vec<float, 16>
{
public:
    static const int ARITY = 16;

    typedef short_vec_strategy::neon strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
            std::basic_ostream<_CharT, _Traits>& __os,
            const short_vec<float, 16>& vec);

    inline
    short_vec(const float data = 0) :
        val1(vdupq_n_f32(data)),
        val2(vdupq_n_f32(data)),
        val3(vdupq_n_f32(data)),
        val4(vdupq_n_f32(data))
    {}

    inline
    short_vec(const float *data) :
        val1(vld1q_f32( (data + 0) )),
        val2(vld1q_f32( (data + 4) )),
        val3(vld1q_f32( (data + 8) )),
        val4(vld1q_f32( (data + 12) ))
    {}

    inline
    short_vec(const float32x4_t& val1, const float32x4_t& val2,
        const float32x4_t& val3, const float32x4_t& val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
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
    void operator-=(const short_vec<float, 16>& other)
    {
        val1 = vsubq_f32(val1, other.val1);
        val2 = vsubq_f32(val2, other.val2);
        val3 = vsubq_f32(val3, other.val3);
        val4 = vsubq_f32(val4, other.val4);
    }

    inline
    short_vec<float, 16> operator-(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            vsubq_f32(val1, other.val1), vsubq_f32(val2, other.val2),
            vsubq_f32(val3, other.val3), vsubq_f32(val4, other.val4)
            );
    }

    inline
    void operator+=(const short_vec<float, 16>& other)
    {
        val1 = vaddq_f32(val1, other.val1);
        val2 = vaddq_f32(val2, other.val2);
        val3 = vaddq_f32(val3, other.val3);
        val4 = vaddq_f32(val4, other.val4);
    }

    inline
    short_vec<float, 16> operator+(const short_vec<float, 16>& other) const
    {
        short_vec<float, 16> ret(
            vaddq_f32(val1, other.val1), vaddq_f32(val2, other.val2),
            vaddq_f32(val3, other.val3), vaddq_f32(val4, other.val4)
            );
        return ret;
    }

    inline
    void operator*=(const short_vec<float, 16>& other)
    {
        val1 = vmulq_f32(val1, other.val1);
        val2 = vmulq_f32(val2, other.val2);
        val3 = vmulq_f32(val3, other.val3);
        val4 = vmulq_f32(val4, other.val4);
    }

    inline
    short_vec<float, 16> operator*(const short_vec<float, 16>& other) const
    {
        short_vec<float, 16> ret(
            vmulq_f32(val1, other.val1), vmulq_f32(val2, other.val2),
            vmulq_f32(val3, other.val3), vmulq_f32(val4, other.val4));
        return ret;
    }

    // Code created with the help of Stack Overflow question
    // http://stackoverflow.com/questions/3808808/how-to-get-element-by-class-in-javascript
    // Question by Taylor:
    // http://stackoverflow.com/users/853570/darkmax
    // Answer by Andrew Dunn:
    // http://stackoverflow.com/users/142434/stephen-canon
    inline
    void operator/=(const short_vec<float, 16>& other)
    {
        int iterations = 1;
        // get an initial estimate of 1/b.
        float32x4_t reciprocal1 = vrecpeq_f32(other.val1);
        float32x4_t reciprocal2 = vrecpeq_f32(other.val2);
        float32x4_t reciprocal3 = vrecpeq_f32(other.val3);
        float32x4_t reciprocal4 = vrecpeq_f32(other.val4);

        // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
        // application's accuracy requirements, you may be able to get away with only
        // one refinement (instead of the two used here).  Be sure to test!
#ifdef LIBFLATARRAY_WITH_INCREASED_PRECISION
        iterations = 2;
#endif
        for (int i = 0; i < iterations; ++i) {
            reciprocal1 = vmulq_f32(vrecpsq_f32(other.val1, reciprocal1), reciprocal1);
            reciprocal2 = vmulq_f32(vrecpsq_f32(other.val2, reciprocal2), reciprocal2);
            reciprocal3 = vmulq_f32(vrecpsq_f32(other.val3, reciprocal3), reciprocal3);
            reciprocal4 = vmulq_f32(vrecpsq_f32(other.val4, reciprocal4), reciprocal4);
        }

        // and finally, compute a/b = a*(1/b)
        val1 = vmulq_f32(val1, reciprocal1);
        val2 = vmulq_f32(val2, reciprocal2);
        val3 = vmulq_f32(val3, reciprocal3);
        val4 = vmulq_f32(val4, reciprocal4);
    }

    // Code created with the help of Stack Overflow question
    // http://stackoverflow.com/questions/3808808/how-to-get-element-by-class-in-javascript
    // Question by Taylor:
    // http://stackoverflow.com/users/853570/darkmax
    // Answer by Andrew Dunn:
    // http://stackoverflow.com/users/142434/stephen-canon
    inline
    short_vec<float, 16> operator/(const short_vec<float, 16>& other) const
    {
        int iterations = 1;
        // get an initial estimate of 1/b.
        float32x4_t reciprocal1 = vrecpeq_f32(other.val1);
        float32x4_t reciprocal2 = vrecpeq_f32(other.val2);
        float32x4_t reciprocal3 = vrecpeq_f32(other.val3);
        float32x4_t reciprocal4 = vrecpeq_f32(other.val4);

        // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
        // application's accuracy requirements, you may be able to get away with only
        // one refinement (instead of the two used here).  Be sure to test!
#ifdef LIBFLATARRAY_WITH_INCREASED_PRECISION
        iterations = 2;
#endif
        for (int i = 0; i < iterations; ++i) {
            reciprocal1 = vmulq_f32(vrecpsq_f32(other.val1, reciprocal1), reciprocal1);
            reciprocal2 = vmulq_f32(vrecpsq_f32(other.val2, reciprocal2), reciprocal2);
            reciprocal3 = vmulq_f32(vrecpsq_f32(other.val3, reciprocal3), reciprocal3);
            reciprocal4 = vmulq_f32(vrecpsq_f32(other.val4, reciprocal4), reciprocal4);
        }

        // and finally, compute a/b = a*(1/b)
        float32x4_t result1 = vmulq_f32(val1, reciprocal1);
        float32x4_t result2 = vmulq_f32(val2, reciprocal2);
        float32x4_t result3 = vmulq_f32(val3, reciprocal3);
        float32x4_t result4 = vmulq_f32(val4, reciprocal4);

        short_vec<float, 16> ret(
            result1,
            result2,
            result3,
            result4
            );
        return ret;
    }

    // Copyright (c) 2011, The WebRTC project authors. All rights reserved.
    inline
    short_vec<float, 16> sqrt() const
    {
        // note that vsqrtq_f32 is to be implemented in the gcc compiler
        int i, iterations = 1;
        float32x4_t x1 = vrsqrteq_f32(val1);
        float32x4_t x2 = vrsqrteq_f32(val2);
        float32x4_t x3 = vrsqrteq_f32(val3);
        float32x4_t x4 = vrsqrteq_f32(val4);

        // Code to handle sqrt(0).
        // If the input to sqrtf() is zero, a zero will be returned.
        // If the input to vrsqrteq_f32() is zero, positive infinity is returned.
        const uint32x4_t vec_p_inf = vdupq_n_u32(0x7F800000);
        // check for divide by zero
        const uint32x4_t div_by_zero1 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x1));
        const uint32x4_t div_by_zero2 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x2));
        const uint32x4_t div_by_zero3 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x3));
        const uint32x4_t div_by_zero4 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x4));
        // zero out the positive infinity results
        x1 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero1),
                                            vreinterpretq_u32_f32(x1)));
        x2 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero2),
                                            vreinterpretq_u32_f32(x2)));
        x3 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero3),
                                            vreinterpretq_u32_f32(x3)));
        x4 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero4),
                                            vreinterpretq_u32_f32(x4)));
        // from arm documentation
        // The Newton-Raphson iteration:
        //     x[n+1] = x[n] * (3 - d * (x[n] * x[n])) / 2)
        // converges to (1/√d) if x0 is the result of VRSQRTE applied to d.
        //
        // Note: The precision did not improve after 2 iterations.
#ifdef LIBFLATARRAY_WITH_INCREASED_PRECISION
        iterations = 2;
#endif
        for (i = 0; i < iterations; ++i) {
            x1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x1, x1), val1), x1);
            x2 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x2, x2), val2), x2);
            x3 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x3, x3), val3), x3);
            x4 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x4, x4), val4), x4);
        }

        // sqrt(s) = s * 1/sqrt(s)
        float32x4_t result1 = vmulq_f32(val1, x1);
        float32x4_t result2 = vmulq_f32(val2, x2);
        float32x4_t result3 = vmulq_f32(val3, x3);
        float32x4_t result4 = vmulq_f32(val4, x4);
        short_vec<float, 16> ret(result1, result2, result3, result4);

        return ret;
    }

    inline
    void load(const float *data)
    {
        val1 = vld1q_f32((data + 0));
        val2 = vld1q_f32((data + 4));
        val3 = vld1q_f32((data + 8));
        val4 = vld1q_f32((data + 12));
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        load(data);
    }

    inline
    void store(float *data) const
    {
        vst1q_f32(data, val1);
        vst1q_f32(data + 4, val2);
        vst1q_f32(data + 8, val3);
        vst1q_f32(data + 12, val4);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        store(data);
    }

    inline
    void store_nt(float *data) const
    {
        // in arm only stnp support non-temporal hint, thus need to
        // break into two registers. (use helper val2)
        // see if it can get optimized by compiler

        // the mapping between Q registers and D registers

        // stnp is for arm 64 (armv16)
#if __LP64__
        register float32x4_t tmp1 asm ("q0");
        tmp1 = val1;
        register float32x4_t tmp2 asm ("q1");
        tmp2 = val2;
        register float32x4_t tmp3 asm ("q2");
        tmp3 = val3;
        register float32x4_t tmp4 asm ("q3");
        tmp4 = val4;
        asm("stnp d0, d1, %[store]"
            :[store] "=m" (data)
            );
        asm("stnp d2, d3, %[store]"
            :[store] "=m" (data + 4)
            );
        asm("stnp d4, d5, %[store]"
            :[store] "=m" (data + 8)
            );
        asm("stnp d6, d7, %[store]"
            :[store] "=m" (data + 12)
            );
#else
        store(data);
#endif
    }

    // dummy approach. NEON only supports loading in fixed interleaving
    inline
    void gather(const float *ptr, const int *offsets)
    {
        float data[16];
        data[0] = ptr[offsets[0]];
        data[1] = ptr[offsets[1]];
        data[2] = ptr[offsets[2]];
        data[3] = ptr[offsets[3]];
        data[4] = ptr[offsets[4]];
        data[5] = ptr[offsets[5]];
        data[6] = ptr[offsets[6]];
        data[7] = ptr[offsets[7]];
        data[8] = ptr[offsets[8]];
        data[9] = ptr[offsets[9]];
        data[10] = ptr[offsets[10]];
        data[11] = ptr[offsets[11]];
        data[12] = ptr[offsets[12]];
        data[13] = ptr[offsets[13]];
        data[14] = ptr[offsets[14]];
        data[15] = ptr[offsets[15]];
        load(data);
    }

    // dummy approach
    inline
    void scatter(float *ptr, const int *offsets) const
    {
        const float *data1 = reinterpret_cast<const float *>(&val1);
        const float *data2 = reinterpret_cast<const float *>(&val2);
        const float *data3 = reinterpret_cast<const float *>(&val3);
        const float *data4 = reinterpret_cast<const float *>(&val4);
        ptr[offsets[0]] = data1[0];
        ptr[offsets[1]] = data1[1];
        ptr[offsets[2]] = data1[2];
        ptr[offsets[3]] = data1[3];
        ptr[offsets[4]] = data2[0];
        ptr[offsets[5]] = data2[1];
        ptr[offsets[6]] = data2[2];
        ptr[offsets[7]] = data2[3];
        ptr[offsets[8]] = data3[0];
        ptr[offsets[9]] = data3[1];
        ptr[offsets[10]] = data3[2];
        ptr[offsets[11]] = data3[3];
        ptr[offsets[12]] = data4[0];
        ptr[offsets[13]] = data4[1];
        ptr[offsets[14]] = data4[2];
        ptr[offsets[15]] = data4[3];
    }

private:
    float32x4_t val1;
    float32x4_t val2;
    float32x4_t val3;
    float32x4_t val4;
};

inline
void operator<<(float *data, const short_vec<float, 16>& vec)
{
    vec.store(data);
}

inline
    short_vec<float, 16> sqrt(const short_vec<float, 16>& vec)
{
    return vec.sqrt();
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
        << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
        << ", "
        << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
        << ", "
        << data3[0] << ", " << data3[1] << ", " << data3[2] << ", " << data3[3]
        << ", "
        << data4[0] << ", " << data4[1] << ", " << data4[2] << ", " << data4[3]
        << "]";
    return __os;
}

}

#endif

#endif
