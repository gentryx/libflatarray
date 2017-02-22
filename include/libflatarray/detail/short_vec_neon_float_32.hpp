/**
 * Copyright 2015 Di Xiao
 * Copyright 2016-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_32_HPP

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
class short_vec<float, 32>
{
public:
    static const std::size_t ARITY = 32;

    typedef short_vec_strategy::neon strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
            std::basic_ostream<_CharT, _Traits>& __os,
            const short_vec<float, 32>& vec);

    inline
    short_vec(const float data = 0) :
        val{vdupq_n_f32(data),
            vdupq_n_f32(data),
            vdupq_n_f32(data),
            vdupq_n_f32(data),
            vdupq_n_f32(data),
            vdupq_n_f32(data),
            vdupq_n_f32(data),
            vdupq_n_f32(data)}
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(
        const float32x4_t& val1,
        const float32x4_t& val2,
        const float32x4_t& val3,
        const float32x4_t& val4,
        const float32x4_t& val5,
        const float32x4_t& val6,
        const float32x4_t& val7,
        const float32x4_t& val8) :
        val{val1,
            val2,
            val3,
            val4,
            val5,
            val6,
            val7,
            val8}
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
    void operator-=(const short_vec<float, 32>& other)
    {
        val[ 0] = vsubq_f32(val[ 0], other.val[ 0]);
        val[ 1] = vsubq_f32(val[ 1], other.val[ 1]);
        val[ 2] = vsubq_f32(val[ 2], other.val[ 2]);
        val[ 3] = vsubq_f32(val[ 3], other.val[ 3]);
        val[ 4] = vsubq_f32(val[ 4], other.val[ 4]);
        val[ 5] = vsubq_f32(val[ 5], other.val[ 5]);
        val[ 6] = vsubq_f32(val[ 6], other.val[ 6]);
        val[ 7] = vsubq_f32(val[ 7], other.val[ 7]);
    }

    inline
    short_vec<float, 32> operator-(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            vsubq_f32(val[ 0], other.val[ 0]), vsubq_f32(val[ 1], other.val[ 1]),
            vsubq_f32(val[ 2], other.val[ 2]), vsubq_f32(val[ 3], other.val[ 3]),
            vsubq_f32(val[ 4], other.val[ 4]), vsubq_f32(val[ 5], other.val[ 5]),
            vsubq_f32(val[ 6], other.val[ 6]), vsubq_f32(val[ 7], other.val[ 7])
            );
    }

    inline
    void operator+=(const short_vec<float, 32>& other)
    {
        val[ 0] = vaddq_f32(val[ 0], other.val[ 0]);
        val[ 1] = vaddq_f32(val[ 1], other.val[ 1]);
        val[ 2] = vaddq_f32(val[ 2], other.val[ 2]);
        val[ 3] = vaddq_f32(val[ 3], other.val[ 3]);
        val[ 4] = vaddq_f32(val[ 4], other.val[ 4]);
        val[ 5] = vaddq_f32(val[ 5], other.val[ 5]);
        val[ 6] = vaddq_f32(val[ 6], other.val[ 6]);
        val[ 7] = vaddq_f32(val[ 7], other.val[ 7]);
    }

    inline
    short_vec<float, 32> operator+(const short_vec<float, 32>& other) const
    {
        short_vec<float, 32> ret(
            vaddq_f32(val[ 0], other.val[ 0]), vaddq_f32(val[ 1], other.val[ 1]),
            vaddq_f32(val[ 2], other.val[ 2]), vaddq_f32(val[ 3], other.val[ 3]),
            vaddq_f32(val[ 4], other.val[ 4]), vaddq_f32(val[ 5], other.val[ 5]),
            vaddq_f32(val[ 6], other.val[ 6]), vaddq_f32(val[ 7], other.val[ 7])
            );
        return ret;
    }

    inline
    void operator*=(const short_vec<float, 32>& other)
    {
        val[ 0] = vmulq_f32(val[ 0], other.val[ 0]);
        val[ 1] = vmulq_f32(val[ 1], other.val[ 1]);
        val[ 2] = vmulq_f32(val[ 2], other.val[ 2]);
        val[ 3] = vmulq_f32(val[ 3], other.val[ 3]);
        val[ 4] = vmulq_f32(val[ 4], other.val[ 4]);
        val[ 5] = vmulq_f32(val[ 5], other.val[ 5]);
        val[ 6] = vmulq_f32(val[ 6], other.val[ 6]);
        val[ 7] = vmulq_f32(val[ 7], other.val[ 7]);
    }

    inline
    short_vec<float, 32> operator*(const short_vec<float, 32>& other) const
    {
        short_vec<float, 32> ret(
            vmulq_f32(val[ 0], other.val[ 0]), vmulq_f32(val[ 1], other.val[ 1]),
            vmulq_f32(val[ 2], other.val[ 2]), vmulq_f32(val[ 3], other.val[ 3]),
            vmulq_f32(val[ 4], other.val[ 4]), vmulq_f32(val[ 5], other.val[ 5]),
            vmulq_f32(val[ 6], other.val[ 6]), vmulq_f32(val[ 7], other.val[ 7])
            );
        return ret;
    }

    // Code created with the help of Stack Overflow question
    // http://stackoverflow.com/questions/3808808/how-to-get-element-by-class-in-javascript
    // Question by Taylor:
    // http://stackoverflow.com/users/853570/darkmax
    // Answer by Andrew Dunn:
    // http://stackoverflow.com/users/142434/stephen-canon
    inline
    void operator/=(const short_vec<float, 32>& other)
    {
        int iterations = 1;
        // get an initial estimate of 1/b.
        float32x4_t reciprocal1 = vrecpeq_f32(other.val[ 0]);
        float32x4_t reciprocal2 = vrecpeq_f32(other.val[ 1]);
        float32x4_t reciprocal3 = vrecpeq_f32(other.val[ 2]);
        float32x4_t reciprocal4 = vrecpeq_f32(other.val[ 3]);
        float32x4_t reciprocal5 = vrecpeq_f32(other.val[ 4]);
        float32x4_t reciprocal6 = vrecpeq_f32(other.val[ 5]);
        float32x4_t reciprocal7 = vrecpeq_f32(other.val[ 6]);
        float32x4_t reciprocal8 = vrecpeq_f32(other.val[ 7]);

        // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
        // application's accuracy requirements, you may be able to get away with only
        // one refinement (instead of the two used here).  Be sure to test!
#ifdef LIBFLATARRAY_WITH_INCREASED_PRECISION
        iterations = 2;
#endif
        for (int i = 0; i < iterations; ++i) {
            reciprocal1 = vmulq_f32(vrecpsq_f32(other.val[ 0], reciprocal1), reciprocal1);
            reciprocal2 = vmulq_f32(vrecpsq_f32(other.val[ 1], reciprocal2), reciprocal2);
            reciprocal3 = vmulq_f32(vrecpsq_f32(other.val[ 2], reciprocal3), reciprocal3);
            reciprocal4 = vmulq_f32(vrecpsq_f32(other.val[ 3], reciprocal4), reciprocal4);
            reciprocal5 = vmulq_f32(vrecpsq_f32(other.val[ 4], reciprocal5), reciprocal5);
            reciprocal6 = vmulq_f32(vrecpsq_f32(other.val[ 5], reciprocal6), reciprocal6);
            reciprocal7 = vmulq_f32(vrecpsq_f32(other.val[ 6], reciprocal7), reciprocal7);
            reciprocal8 = vmulq_f32(vrecpsq_f32(other.val[ 7], reciprocal8), reciprocal8);
        }

        // and finally, compute a/b = a*(1/b)
        val[ 0] = vmulq_f32(val[ 0], reciprocal1);
        val[ 1] = vmulq_f32(val[ 1], reciprocal2);
        val[ 2] = vmulq_f32(val[ 2], reciprocal3);
        val[ 3] = vmulq_f32(val[ 3], reciprocal4);
        val[ 4] = vmulq_f32(val[ 4], reciprocal5);
        val[ 5] = vmulq_f32(val[ 5], reciprocal6);
        val[ 6] = vmulq_f32(val[ 6], reciprocal7);
        val[ 7] = vmulq_f32(val[ 7], reciprocal8);
    }

    // Code created with the help of Stack Overflow question
    // http://stackoverflow.com/questions/3808808/how-to-get-element-by-class-in-javascript
    // Question by Taylor:
    // http://stackoverflow.com/users/853570/darkmax
    // Answer by Andrew Dunn:
    // http://stackoverflow.com/users/142434/stephen-canon
    inline
    short_vec<float, 32> operator/(const short_vec<float, 32>& other) const
    {
        int iterations = 1;
        // get an initial estimate of 1/b.
        float32x4_t reciprocal1 = vrecpeq_f32(other.val[ 0]);
        float32x4_t reciprocal2 = vrecpeq_f32(other.val[ 1]);
        float32x4_t reciprocal3 = vrecpeq_f32(other.val[ 2]);
        float32x4_t reciprocal4 = vrecpeq_f32(other.val[ 3]);
        float32x4_t reciprocal5 = vrecpeq_f32(other.val[ 4]);
        float32x4_t reciprocal6 = vrecpeq_f32(other.val[ 5]);
        float32x4_t reciprocal7 = vrecpeq_f32(other.val[ 6]);
        float32x4_t reciprocal8 = vrecpeq_f32(other.val[ 7]);

        // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
        // application's accuracy requirements, you may be able to get away with only
        // one refinement (instead of the two used here).  Be sure to test!
#ifdef LIBFLATARRAY_WITH_INCREASED_PRECISION
        iterations = 2;
#endif
        for (int i = 0; i < iterations; ++i) {
            reciprocal1 = vmulq_f32(vrecpsq_f32(other.val[ 0], reciprocal1), reciprocal1);
            reciprocal2 = vmulq_f32(vrecpsq_f32(other.val[ 1], reciprocal2), reciprocal2);
            reciprocal3 = vmulq_f32(vrecpsq_f32(other.val[ 2], reciprocal3), reciprocal3);
            reciprocal4 = vmulq_f32(vrecpsq_f32(other.val[ 3], reciprocal4), reciprocal4);
            reciprocal5 = vmulq_f32(vrecpsq_f32(other.val[ 4], reciprocal5), reciprocal5);
            reciprocal6 = vmulq_f32(vrecpsq_f32(other.val[ 5], reciprocal6), reciprocal6);
            reciprocal7 = vmulq_f32(vrecpsq_f32(other.val[ 6], reciprocal7), reciprocal7);
            reciprocal8 = vmulq_f32(vrecpsq_f32(other.val[ 7], reciprocal8), reciprocal8);
        }

        // and finally, compute a/b = a*(1/b)
        float32x4_t result1 = vmulq_f32(val[ 0], reciprocal1);
        float32x4_t result2 = vmulq_f32(val[ 1], reciprocal2);
        float32x4_t result3 = vmulq_f32(val[ 2], reciprocal3);
        float32x4_t result4 = vmulq_f32(val[ 3], reciprocal4);
        float32x4_t result5 = vmulq_f32(val[ 4], reciprocal5);
        float32x4_t result6 = vmulq_f32(val[ 5], reciprocal6);
        float32x4_t result7 = vmulq_f32(val[ 6], reciprocal7);
        float32x4_t result8 = vmulq_f32(val[ 7], reciprocal8);

        short_vec<float, 32> ret(
            result1,
            result2,
            result3,
            result4,
            result5,
            result6,
            result7,
            result8);

        return ret;
    }

    // Copyright (c) 2011, The WebRTC project authors. All rights reserved.
    inline
    short_vec<float, 32> sqrt() const
    {
        // note that vsqrtq_f32 is to be implemented in the gcc compiler
        int i, iterations=1;
        float32x4_t x1 = vrsqrteq_f32(val[ 0]);
        float32x4_t x2 = vrsqrteq_f32(val[ 1]);
        float32x4_t x3 = vrsqrteq_f32(val[ 2]);
        float32x4_t x4 = vrsqrteq_f32(val[ 3]);
        float32x4_t x5 = vrsqrteq_f32(val[ 4]);
        float32x4_t x6 = vrsqrteq_f32(val[ 5]);
        float32x4_t x7 = vrsqrteq_f32(val[ 6]);
        float32x4_t x8 = vrsqrteq_f32(val[ 7]);

        // Code to handle sqrt(0).
        // If the input to sqrtf() is zero, a zero will be returned.
        // If the input to vrsqrteq_f32() is zero, positive infinity is returned.
        const uint32x4_t vec_p_inf = vdupq_n_u32(0x7F800000);
        // check for divide by zero
        const uint32x4_t div_by_zero1 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x1));
        const uint32x4_t div_by_zero2 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x2));
        const uint32x4_t div_by_zero3 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x3));
        const uint32x4_t div_by_zero4 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x4));
        const uint32x4_t div_by_zero5 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x5));
        const uint32x4_t div_by_zero6 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x6));
        const uint32x4_t div_by_zero7 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x7));
        const uint32x4_t div_by_zero8 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x8));
        // zero out the positive infinity results
        x1 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero1),
                                             vreinterpretq_u32_f32(x1)));
        x2 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero2),
                                             vreinterpretq_u32_f32(x2)));
        x3 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero3),
                                             vreinterpretq_u32_f32(x3)));
        x4 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero4),
                                             vreinterpretq_u32_f32(x4)));
        x5 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero5),
                                             vreinterpretq_u32_f32(x5)));
        x6 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero6),
                                             vreinterpretq_u32_f32(x6)));
        x7 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero7),
                                             vreinterpretq_u32_f32(x7)));
        x8 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero8),
                                             vreinterpretq_u32_f32(x8)));
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
            x1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x1, x1), val[ 0]), x1);
            x2 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x2, x2), val[ 1]), x2);
            x3 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x3, x3), val[ 2]), x3);
            x4 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x4, x4), val[ 3]), x4);
            x5 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x5, x5), val[ 4]), x5);
            x6 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x6, x6), val[ 5]), x6);
            x7 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x7, x7), val[ 6]), x7);
            x8 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x8, x8), val[ 7]), x8);
        }
        // sqrt(s) = s * 1/sqrt(s)
        float32x4_t result1 = vmulq_f32(val[ 0], x1);
        float32x4_t result2 = vmulq_f32(val[ 1], x2);
        float32x4_t result3 = vmulq_f32(val[ 2], x3);
        float32x4_t result4 = vmulq_f32(val[ 3], x4);
        float32x4_t result5 = vmulq_f32(val[ 4], x5);
        float32x4_t result6 = vmulq_f32(val[ 5], x6);
        float32x4_t result7 = vmulq_f32(val[ 6], x7);
        float32x4_t result8 = vmulq_f32(val[ 7], x8);
        short_vec<float, 32> ret(
            result1, result2, result3, result4,
            result5, result6, result7, result8
            );
        return ret;
    }

    inline
    void load(const float *data)
    {
        val[ 0] = vld1q_f32((data + 0));
        val[ 1] = vld1q_f32((data + 4));
        val[ 2] = vld1q_f32((data + 8));
        val[ 3] = vld1q_f32((data + 12));
        val[ 4] = vld1q_f32((data + 16));
        val[ 5] = vld1q_f32((data + 20));
        val[ 6] = vld1q_f32((data + 24));
        val[ 7] = vld1q_f32((data + 28));
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 128);
        load(data);
    }

    inline
    void store(float *data) const
    {
        vst1q_f32(data + 0, val[ 0]);
        vst1q_f32(data + 4, val[ 1]);
        vst1q_f32(data + 8, val[ 2]);
        vst1q_f32(data + 12, val[ 3]);
        vst1q_f32(data + 16, val[ 4]);
        vst1q_f32(data + 20, val[ 5]);
        vst1q_f32(data + 24, val[ 6]);
        vst1q_f32(data + 28, val[ 7]);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 128);
        store(data);
    }

    inline
    void store_nt(float *data) const
    {
        // in arm only stnp support non-temporal hint, thus need to
        // break into two registers. (use helper val[ 1])
        // see if it can get optimized by compiler

        // the mapping between Q registers and D registers

        // stnp is for arm 64 (armv32)
#if __LP64__
        register float32x4_t tmp1 asm ("q0");
        tmp1 = val[ 0];
        register float32x4_t tmp2 asm ("q1");
        tmp2 = val[ 1];
        register float32x4_t tmp3 asm ("q2");
        tmp3 = val[ 2];
        register float32x4_t tmp4 asm ("q3");
        tmp4 = val[ 3];
        register float32x4_t tmp5 asm ("q4");
        tmp5 = val[ 4];
        register float32x4_t tmp6 asm ("q5");
        tmp6 = val[ 5];
        register float32x4_t tmp7 asm ("q6");
        tmp7 = val[ 6];
        register float32x4_t tmp8 asm ("q7");
        tmp8 = val[ 7];
        asm("stnp d0, d1, %[store]" :[store] "=m" (data + 0));
        asm("stnp d2, d3, %[store]" :[store] "=m" (data + 4));
        asm("stnp d4, d5, %[store]" :[store] "=m" (data + 8));
        asm("stnp d6, d7, %[store]" :[store] "=m" (data + 12));
        asm("stnp d7, d8, %[store]" :[store] "=m" (data + 16));
        asm("stnp d8, d9, %[store]" :[store] "=m" (data + 20));
        asm("stnp d9, d10, %[store]" :[store] "=m" (data + 24));
        asm("stnp d10, d11, %[store]" :[store] "=m" (data + 28));
#else
        store(data);
#endif
    }

    // dummy approach. NEON only supports loading in fixed interleaving
    inline
    void gather(const float *ptr, const int *offsets)
    {
        float data[32];
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
        data[16] = ptr[offsets[16]];
        data[17] = ptr[offsets[17]];
        data[18] = ptr[offsets[18]];
        data[19] = ptr[offsets[19]];
        data[20] = ptr[offsets[20]];
        data[21] = ptr[offsets[21]];
        data[22] = ptr[offsets[22]];
        data[23] = ptr[offsets[23]];
        data[24] = ptr[offsets[24]];
        data[25] = ptr[offsets[25]];
        data[26] = ptr[offsets[26]];
        data[27] = ptr[offsets[27]];
        data[28] = ptr[offsets[28]];
        data[29] = ptr[offsets[29]];
        data[30] = ptr[offsets[30]];
        data[31] = ptr[offsets[31]];
        load(data);
    }

    // dummy approach
    inline
    void scatter(float *ptr, const int *offsets) const
    {
        const float *data1 = reinterpret_cast<const float *>(&val[ 0]);
        const float *data2 = reinterpret_cast<const float *>(&val[ 1]);
        const float *data3 = reinterpret_cast<const float *>(&val[ 2]);
        const float *data4 = reinterpret_cast<const float *>(&val[ 3]);
        const float *data5 = reinterpret_cast<const float *>(&val[ 4]);
        const float *data6 = reinterpret_cast<const float *>(&val[ 5]);
        const float *data7 = reinterpret_cast<const float *>(&val[ 6]);
        const float *data8 = reinterpret_cast<const float *>(&val[ 7]);
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
        ptr[offsets[16]] = data5[0];
        ptr[offsets[17]] = data5[1];
        ptr[offsets[18]] = data5[2];
        ptr[offsets[19]] = data5[3];
        ptr[offsets[20]] = data6[0];
        ptr[offsets[21]] = data6[1];
        ptr[offsets[22]] = data6[2];
        ptr[offsets[23]] = data6[3];
        ptr[offsets[24]] = data7[0];
        ptr[offsets[25]] = data7[1];
        ptr[offsets[26]] = data7[2];
        ptr[offsets[27]] = data7[3];
        ptr[offsets[28]] = data8[0];
        ptr[offsets[29]] = data8[1];
        ptr[offsets[30]] = data8[2];
        ptr[offsets[31]] = data8[3];
    }

private:
    float32x4_t val[8];
};

inline
void operator<<(float *data, const short_vec<float, 32>& vec)
{
    vec.store(data);
}

inline
    short_vec<float, 32> sqrt(const short_vec<float, 32>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 32>& vec)
{
    const float *data1 = reinterpret_cast<const float *>(&vec.val[ 0]);
    const float *data2 = reinterpret_cast<const float *>(&vec.val[ 1]);
    const float *data3 = reinterpret_cast<const float *>(&vec.val[ 2]);
    const float *data4 = reinterpret_cast<const float *>(&vec.val[ 3]);
    const float *data5 = reinterpret_cast<const float *>(&vec.val[ 4]);
    const float *data6 = reinterpret_cast<const float *>(&vec.val[ 5]);
    const float *data7 = reinterpret_cast<const float *>(&vec.val[ 6]);
    const float *data8 = reinterpret_cast<const float *>(&vec.val[ 7]);
    __os << "["
        << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
        << ", "
        << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
        << ", "
        << data3[0] << ", " << data3[1] << ", " << data3[2] << ", " << data3[3]
        << ", "
        << data4[0] << ", " << data4[1] << ", " << data4[2] << ", " << data4[3]
        << ", "
        << data5[0] << ", " << data5[1] << ", " << data5[2] << ", " << data5[3]
        << ", "
        << data6[0] << ", " << data6[1] << ", " << data6[2] << ", " << data6[3]
        << ", "
        << data7[0] << ", " << data7[1] << ", " << data7[2] << ", " << data7[3]
        << ", "
        << data8[0] << ", " << data8[1] << ", " << data8[2] << ", " << data8[3]
        << "]";
    return __os;
}

}

#endif

#endif
