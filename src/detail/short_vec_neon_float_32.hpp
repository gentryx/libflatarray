/**
 * Copyright 2015 Di Xiao
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_32_HPP

#ifdef __ARM_NEON__
#include <arm_neon.h>
#include <stdlib.h>
#include <libflatarray/detail/short_vec_helpers.hpp>

#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

template<>
class short_vec<float, 32>
{
public:
    static const int ARITY = 32;

    typedef short_vec_strategy::neon strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
            std::basic_ostream<_CharT, _Traits>& __os,
            const short_vec<float, 32>& vec);

    inline
    short_vec(const float data = 0) :
        val1(vdupq_n_f32(data)), val2(vdupq_n_f32(data)),
        val3(vdupq_n_f32(data)), val4(vdupq_n_f32(data)),
        val5(vdupq_n_f32(data)), val6(vdupq_n_f32(data)),
        val7(vdupq_n_f32(data)), val8(vdupq_n_f32(data))
    {}

    inline
    short_vec(const float *data) :
        val1(vld1q_f32((data + 0))), val2(vld1q_f32((data + 4))),
        val3(vld1q_f32((data + 8))), val4(vld1q_f32((data + 12))),
        val5(vld1q_f32((data + 16))), val6(vld1q_f32((data + 20))),
        val7(vld1q_f32((data + 24))), val8(vld1q_f32((data + 28)))
    {}

    inline
    short_vec(const float32x4_t& val1, const float32x4_t& val2,
        const float32x4_t& val3, const float32x4_t& val4,
        const float32x4_t& val5, const float32x4_t& val6,
        const float32x4_t& val7, const float32x4_t& val8
        ) :
        val1(val1), val2(val2), 
        val3(val3), val4(val4), 
        val5(val5), val6(val6), 
        val7(val7), val8(val8)
    {}

    inline
    void operator-=(const short_vec<float, 32>& other)
    {
        val1 = vsubq_f32(val1, other.val1);
        val2 = vsubq_f32(val2, other.val2);
        val3 = vsubq_f32(val3, other.val3);
        val4 = vsubq_f32(val4, other.val4);
        val5 = vsubq_f32(val5, other.val5);
        val6 = vsubq_f32(val6, other.val6);
        val7 = vsubq_f32(val7, other.val7);
        val8 = vsubq_f32(val8, other.val8);
    }

    inline
    short_vec<float, 32> operator-(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            vsubq_f32(val1, other.val1), vsubq_f32(val2, other.val2),
            vsubq_f32(val3, other.val3), vsubq_f32(val4, other.val4),
            vsubq_f32(val5, other.val5), vsubq_f32(val6, other.val6),
            vsubq_f32(val7, other.val7), vsubq_f32(val8, other.val8)
            );
    }

    inline
    void operator+=(const short_vec<float, 32>& other)
    {
        val1 = vaddq_f32(val1, other.val1);
        val2 = vaddq_f32(val2, other.val2);
        val3 = vaddq_f32(val3, other.val3);
        val4 = vaddq_f32(val4, other.val4);
        val5 = vaddq_f32(val5, other.val5);
        val6 = vaddq_f32(val6, other.val6);
        val7 = vaddq_f32(val7, other.val7);
        val8 = vaddq_f32(val8, other.val8);
    }

    inline
    short_vec<float, 32> operator+(const short_vec<float, 32>& other) const
    {
        short_vec<float, 32> ret(
            vaddq_f32(val1, other.val1), vaddq_f32(val2, other.val2),
            vaddq_f32(val3, other.val3), vaddq_f32(val4, other.val4),
            vaddq_f32(val5, other.val5), vaddq_f32(val6, other.val6),
            vaddq_f32(val7, other.val7), vaddq_f32(val8, other.val8)
            );
        return ret;
    }

    inline
    void operator*=(const short_vec<float, 32>& other)
    {
        val1 = vmulq_f32(val1, other.val1);
        val2 = vmulq_f32(val2, other.val2);
        val3 = vmulq_f32(val3, other.val3);
        val4 = vmulq_f32(val4, other.val4);
        val5 = vmulq_f32(val5, other.val5);
        val6 = vmulq_f32(val6, other.val6);
        val7 = vmulq_f32(val7, other.val7);
        val8 = vmulq_f32(val8, other.val8);
    }

    inline
    short_vec<float, 32> operator*(const short_vec<float, 32>& other) const
    {
        short_vec<float, 32> ret(
            vmulq_f32(val1, other.val1), vmulq_f32(val2, other.val2),
            vmulq_f32(val3, other.val3), vmulq_f32(val4, other.val4),
            vmulq_f32(val5, other.val5), vmulq_f32(val6, other.val6),
            vmulq_f32(val7, other.val7), vmulq_f32(val8, other.val8)
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
        // get an initial estimate of 1/b.
        float32x4_t reciprocal1 = vrecpeq_f32(other.val1);
        float32x4_t reciprocal2 = vrecpeq_f32(other.val2);
        float32x4_t reciprocal3 = vrecpeq_f32(other.val3);
        float32x4_t reciprocal4 = vrecpeq_f32(other.val4);
        float32x4_t reciprocal5 = vrecpeq_f32(other.val5);
        float32x4_t reciprocal6 = vrecpeq_f32(other.val6);
        float32x4_t reciprocal7 = vrecpeq_f32(other.val7);
        float32x4_t reciprocal8 = vrecpeq_f32(other.val8);

        // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
        // application's accuracy requirements, you may be able to get away with only
        // one refinement (instead of the two used here).  Be sure to test!
        reciprocal1 = vmulq_f32(vrecpsq_f32(other.val1, reciprocal1), reciprocal1);
        reciprocal1 = vmulq_f32(vrecpsq_f32(other.val1, reciprocal1), reciprocal1);
        reciprocal2 = vmulq_f32(vrecpsq_f32(other.val2, reciprocal2), reciprocal2);
        reciprocal2 = vmulq_f32(vrecpsq_f32(other.val2, reciprocal2), reciprocal2);
        reciprocal3 = vmulq_f32(vrecpsq_f32(other.val3, reciprocal3), reciprocal3);
        reciprocal3 = vmulq_f32(vrecpsq_f32(other.val3, reciprocal3), reciprocal3);
        reciprocal4 = vmulq_f32(vrecpsq_f32(other.val4, reciprocal4), reciprocal4);
        reciprocal4 = vmulq_f32(vrecpsq_f32(other.val4, reciprocal4), reciprocal4);
        reciprocal5 = vmulq_f32(vrecpsq_f32(other.val5, reciprocal5), reciprocal5);
        reciprocal5 = vmulq_f32(vrecpsq_f32(other.val5, reciprocal5), reciprocal5);
        reciprocal6 = vmulq_f32(vrecpsq_f32(other.val6, reciprocal6), reciprocal6);
        reciprocal6 = vmulq_f32(vrecpsq_f32(other.val6, reciprocal6), reciprocal6);
        reciprocal7 = vmulq_f32(vrecpsq_f32(other.val7, reciprocal7), reciprocal7);
        reciprocal7 = vmulq_f32(vrecpsq_f32(other.val7, reciprocal7), reciprocal7);
        reciprocal8 = vmulq_f32(vrecpsq_f32(other.val8, reciprocal8), reciprocal8);
        reciprocal8 = vmulq_f32(vrecpsq_f32(other.val8, reciprocal8), reciprocal8);

        // and finally, compute a/b = a*(1/b)
        val1 = vmulq_f32(val1, reciprocal1);
        val2 = vmulq_f32(val2, reciprocal2);
        val3 = vmulq_f32(val3, reciprocal3);
        val4 = vmulq_f32(val4, reciprocal4);
        val5 = vmulq_f32(val5, reciprocal5);
        val6 = vmulq_f32(val6, reciprocal6);
        val7 = vmulq_f32(val7, reciprocal7);
        val8 = vmulq_f32(val8, reciprocal8);
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
        // get an initial estimate of 1/b.
        float32x4_t reciprocal1 = vrecpeq_f32(other.val1);
        float32x4_t reciprocal2 = vrecpeq_f32(other.val2);
        float32x4_t reciprocal3 = vrecpeq_f32(other.val3);
        float32x4_t reciprocal4 = vrecpeq_f32(other.val4);
        float32x4_t reciprocal5 = vrecpeq_f32(other.val5);
        float32x4_t reciprocal6 = vrecpeq_f32(other.val6);
        float32x4_t reciprocal7 = vrecpeq_f32(other.val7);
        float32x4_t reciprocal8 = vrecpeq_f32(other.val8);

        // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
        // application's accuracy requirements, you may be able to get away with only
        // one refinement (instead of the two used here).  Be sure to test!
        reciprocal1 = vmulq_f32(vrecpsq_f32(other.val1, reciprocal1), reciprocal1);
        reciprocal1 = vmulq_f32(vrecpsq_f32(other.val1, reciprocal1), reciprocal1);
        reciprocal2 = vmulq_f32(vrecpsq_f32(other.val2, reciprocal2), reciprocal2);
        reciprocal2 = vmulq_f32(vrecpsq_f32(other.val2, reciprocal2), reciprocal2);
        reciprocal3 = vmulq_f32(vrecpsq_f32(other.val3, reciprocal3), reciprocal3);
        reciprocal3 = vmulq_f32(vrecpsq_f32(other.val3, reciprocal3), reciprocal3);
        reciprocal4 = vmulq_f32(vrecpsq_f32(other.val4, reciprocal4), reciprocal4);
        reciprocal4 = vmulq_f32(vrecpsq_f32(other.val4, reciprocal4), reciprocal4);
        reciprocal5 = vmulq_f32(vrecpsq_f32(other.val5, reciprocal5), reciprocal5);
        reciprocal5 = vmulq_f32(vrecpsq_f32(other.val5, reciprocal5), reciprocal5);
        reciprocal6 = vmulq_f32(vrecpsq_f32(other.val6, reciprocal6), reciprocal6);
        reciprocal6 = vmulq_f32(vrecpsq_f32(other.val6, reciprocal6), reciprocal6);
        reciprocal7 = vmulq_f32(vrecpsq_f32(other.val7, reciprocal7), reciprocal7);
        reciprocal7 = vmulq_f32(vrecpsq_f32(other.val7, reciprocal7), reciprocal7);
        reciprocal8 = vmulq_f32(vrecpsq_f32(other.val8, reciprocal8), reciprocal8);
        reciprocal8 = vmulq_f32(vrecpsq_f32(other.val8, reciprocal8), reciprocal8);

        // and finally, compute a/b = a*(1/b)
        float32x4_t result1 = vmulq_f32(val1, reciprocal1);
        float32x4_t result2 = vmulq_f32(val2, reciprocal2);
        float32x4_t result3 = vmulq_f32(val3, reciprocal3);
        float32x4_t result4 = vmulq_f32(val4, reciprocal4);
        float32x4_t result5 = vmulq_f32(val5, reciprocal5);
        float32x4_t result6 = vmulq_f32(val6, reciprocal6);
        float32x4_t result7 = vmulq_f32(val7, reciprocal7);
        float32x4_t result8 = vmulq_f32(val8, reciprocal8);

        short_vec<float, 32> ret(
            result1,
            result2,
            result3,
            result4,
            result5,
            result6,
            result7,
            result8
            );
        return ret;
    }

    // Copyright (c) 2011, The WebRTC project authors. All rights reserved.
    inline
    short_vec<float, 32> sqrt() const
    {
        // note that vsqrtq_f32 is to be implemented in the gcc compiler
        int i;
        float32x4_t x1 = vrsqrteq_f32(val1);
        float32x4_t x2 = vrsqrteq_f32(val2);
        float32x4_t x3 = vrsqrteq_f32(val3);
        float32x4_t x4 = vrsqrteq_f32(val4);
        float32x4_t x5 = vrsqrteq_f32(val5);
        float32x4_t x6 = vrsqrteq_f32(val6);
        float32x4_t x7 = vrsqrteq_f32(val7);
        float32x4_t x8 = vrsqrteq_f32(val8);
  
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
        for (i = 0; i < 2; i++) {
          x1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x1, x1), val1), x1);
          x2 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x2, x2), val2), x2);
          x3 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x3, x3), val3), x3);
          x4 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x4, x4), val4), x4);
          x5 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x5, x5), val5), x5);
          x6 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x6, x6), val6), x6);
          x7 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x7, x7), val7), x7);
          x8 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x8, x8), val8), x8);
        }
        // sqrt(s) = s * 1/sqrt(s)
        float32x4_t result1 = vmulq_f32(val1, x1);
        float32x4_t result2 = vmulq_f32(val2, x2);
        float32x4_t result3 = vmulq_f32(val3, x3);
        float32x4_t result4 = vmulq_f32(val4, x4);
        float32x4_t result5 = vmulq_f32(val5, x5);
        float32x4_t result6 = vmulq_f32(val6, x6);
        float32x4_t result7 = vmulq_f32(val7, x7);
        float32x4_t result8 = vmulq_f32(val8, x8);
        short_vec<float, 32> ret(
            result1, result2, result3, result4,
            result5, result6, result7, result8
            );
        return ret;
    }

    inline
    void load(const float *data)
    {
        val1 = vld1q_f32((data + 0));
        val2 = vld1q_f32((data + 4));
        val3 = vld1q_f32((data + 8));
        val4 = vld1q_f32((data + 12));
        val5 = vld1q_f32((data + 16));
        val6 = vld1q_f32((data + 20));
        val7 = vld1q_f32((data + 24));
        val8 = vld1q_f32((data + 28));
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
        vst1q_f32(data + 0, val1);
        vst1q_f32(data + 4, val2);
        vst1q_f32(data + 8, val3);
        vst1q_f32(data + 12, val4);
        vst1q_f32(data + 16, val5);
        vst1q_f32(data + 20, val6);
        vst1q_f32(data + 24, val7);
        vst1q_f32(data + 28, val8);
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
        // break into two registers. (use helper val2)
        // see if it can get optimized by compiler
        
        // the mapping between Q registers and D registers

        // stnp is for arm 64 (armv32)
        #if __LP64__
            register float32x4_t tmp1 asm ("q0");
            tmp1 = val1;
            register float32x4_t tmp2 asm ("q1");
            tmp2 = val2;
            register float32x4_t tmp3 asm ("q2");
            tmp3 = val3;
            register float32x4_t tmp4 asm ("q3");
            tmp4 = val4;
            register float32x4_t tmp5 asm ("q4");
            tmp5 = val5;
            register float32x4_t tmp6 asm ("q5");
            tmp6 = val6;
            register float32x4_t tmp7 asm ("q6");
            tmp7 = val7;
            register float32x4_t tmp8 asm ("q7");
            tmp8 = val8;
            asm("stnp d0, d1, %[store]" :[store] "=m" (data));
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
    void gather(const float *ptr, const unsigned *offsets)
    {
        float * data = (float *) malloc(32 * sizeof(float));
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
    void scatter(float *ptr, const unsigned *offsets) const
    {
        const float *data1 = reinterpret_cast<const float *>(&val1);
        const float *data2 = reinterpret_cast<const float *>(&val2);
        const float *data3 = reinterpret_cast<const float *>(&val3);
        const float *data4 = reinterpret_cast<const float *>(&val4);
        const float *data5 = reinterpret_cast<const float *>(&val5);
        const float *data6 = reinterpret_cast<const float *>(&val6);
        const float *data7 = reinterpret_cast<const float *>(&val7);
        const float *data8 = reinterpret_cast<const float *>(&val8);
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
    float32x4_t val1, val2, val3, val4;
    float32x4_t val5, val6, val7, val8;
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
    const float *data1 = reinterpret_cast<const float *>(&vec.val1);
    const float *data2 = reinterpret_cast<const float *>(&vec.val2);
    const float *data3 = reinterpret_cast<const float *>(&vec.val3);
    const float *data4 = reinterpret_cast<const float *>(&vec.val4);
    const float *data5 = reinterpret_cast<const float *>(&vec.val5);
    const float *data6 = reinterpret_cast<const float *>(&vec.val6);
    const float *data7 = reinterpret_cast<const float *>(&vec.val7);
    const float *data8 = reinterpret_cast<const float *>(&vec.val8);
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

#endif
