/**
 * Copyright 2015 Di Xiao
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_8_HPP

#ifdef __ARM_NEON__
#include <arm_neon.h>
#include <stdlib.h>
#include <libflatarray/detail/short_vec_helpers.hpp>

#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY, bool INCREASE_PRECISION = 1>
class short_vec;

template<>
class short_vec<float, 8>
{
public:
    static const int ARITY = 8;

    typedef short_vec_strategy::neon strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
            std::basic_ostream<_CharT, _Traits>& __os,
            const short_vec<float, 8>& vec);

    inline
    short_vec(const float data = 0) :
        val1(vdupq_n_f32(data)),
        val2(vdupq_n_f32(data))
    {}

    inline
    short_vec(const float *data) :
        val1(vld1q_f32( (data + 0) )),
        val2(vld1q_f32( (data + 4) ))
    {}

    inline
    short_vec(const float32x4_t& val1, const float32x4_t& val2) :
        val1(val1),
        val2(val2)
    {}

    inline
    void operator-=(const short_vec<float, 8>& other)
    {
        val1 = vsubq_f32(val1, other.val1);
        val2 = vsubq_f32(val2, other.val2);
    }

    inline
    short_vec<float, 8> operator-(const short_vec<float, 8>& other) const
    {
        return short_vec<float, 8>(
            vsubq_f32(val1, other.val1), vsubq_f32(val2, other.val2)
            );
    }

    inline
    void operator+=(const short_vec<float, 8>& other)
    {
        val1 = vaddq_f32(val1, other.val1);
        val2 = vaddq_f32(val2, other.val2);
    }

    inline
    short_vec<float, 8> operator+(const short_vec<float, 8>& other) const
    {
        short_vec<float, 8> ret(
            vaddq_f32(val1, other.val1), vaddq_f32(val2, other.val2)
            );
        return ret;
    }

    inline
    void operator*=(const short_vec<float, 8>& other)
    {
        val1 = vmulq_f32(val1, other.val1);
        val2 = vmulq_f32(val2, other.val2);
    }

    inline
    short_vec<float, 8> operator*(const short_vec<float, 8>& other) const
    {
        short_vec<float, 8> ret(
            vmulq_f32(val1, other.val1), vmulq_f32(val2, other.val2)
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
    void operator/=(const short_vec<float, 8>& other)
    {
        // get an initial estimate of 1/b.
        float32x4_t reciprocal1 = vrecpeq_f32(other.val1);
        float32x4_t reciprocal2 = vrecpeq_f32(other.val2);

        // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
        // application's accuracy requirements, you may be able to get away with only
        // one refinement (instead of the two used here).  Be sure to test!
        reciprocal1 = vmulq_f32(vrecpsq_f32(other.val1, reciprocal1), reciprocal1);
        reciprocal2 = vmulq_f32(vrecpsq_f32(other.val2, reciprocal2), reciprocal2);
        if (INCREASE_PRECISION)
        {
            reciprocal1 = vmulq_f32(vrecpsq_f32(other.val1, reciprocal1), reciprocal1);
            reciprocal2 = vmulq_f32(vrecpsq_f32(other.val2, reciprocal2), reciprocal2);
        }

        // and finally, compute a/b = a*(1/b)
        val1 = vmulq_f32(val1, reciprocal1);
        val2 = vmulq_f32(val2, reciprocal2);
    }

    // Code created with the help of Stack Overflow question
    // http://stackoverflow.com/questions/3808808/how-to-get-element-by-class-in-javascript
    // Question by Taylor:
    // http://stackoverflow.com/users/853570/darkmax
    // Answer by Andrew Dunn:
    // http://stackoverflow.com/users/142434/stephen-canon
    inline
    short_vec<float, 8> operator/(const short_vec<float, 8>& other) const
    {
        // get an initial estimate of 1/b.
        float32x4_t reciprocal1 = vrecpeq_f32(other.val1);
        float32x4_t reciprocal2 = vrecpeq_f32(other.val2);

        // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
        // application's accuracy requirements, you may be able to get away with only
        // one refinement (instead of the two used here).  Be sure to test!
        reciprocal1 = vmulq_f32(vrecpsq_f32(other.val1, reciprocal1), reciprocal1);
        reciprocal2 = vmulq_f32(vrecpsq_f32(other.val2, reciprocal2), reciprocal2);
        if (INCREASE_PRECISION)
        {
            reciprocal1 = vmulq_f32(vrecpsq_f32(other.val1, reciprocal1), reciprocal1);
            reciprocal2 = vmulq_f32(vrecpsq_f32(other.val2, reciprocal2), reciprocal2);
        }

        // and finally, compute a/b = a*(1/b)
        float32x4_t result1 = vmulq_f32(val1, reciprocal1);
        float32x4_t result2 = vmulq_f32(val2, reciprocal2);

        short_vec<float, 8> ret(result1, result2);
        return ret;
    }

    // Copyright (c) 2011, The WebRTC project authors. All rights reserved.
    inline
    short_vec<float, 8> sqrt() const
    {
        // note that vsqrtq_f32 is to be implemented in the gcc compiler
        int i;
        float32x4_t x1 = vrsqrteq_f32(val1);
        float32x4_t x2 = vrsqrteq_f32(val2);
  
        // Code to handle sqrt(0).
        // If the input to sqrtf() is zero, a zero will be returned.
        // If the input to vrsqrteq_f32() is zero, positive infinity is returned.
        const uint32x4_t vec_p_inf = vdupq_n_u32(0x7F800000);
        // check for divide by zero
        const uint32x4_t div_by_zero1 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x1));
        const uint32x4_t div_by_zero2 = vceqq_u32(vec_p_inf, vreinterpretq_u32_f32(x2));
        // zero out the positive infinity results
        x1 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero1),
                                            vreinterpretq_u32_f32(x1)));
        x2 = vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(div_by_zero2),
                                            vreinterpretq_u32_f32(x2)));
        // from arm documentation
        // The Newton-Raphson iteration:
        //     x[n+1] = x[n] * (3 - d * (x[n] * x[n])) / 2)
        // converges to (1/√d) if x0 is the result of VRSQRTE applied to d.
        //
        // Note: The precision did not improve after 2 iterations.
        x1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x1, x1), val1), x1);
        x2 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x2, x2), val2), x2);
        if (INCREASE_PRECISION)
        {
          x1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x1, x1), val1), x1);
          x2 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x2, x2), val2), x2);
        }
        // sqrt(s) = s * 1/sqrt(s)
        float32x4_t result1 = vmulq_f32(val1, x1);
        float32x4_t result2 = vmulq_f32(val2, x2);
        short_vec<float, 8> ret(result1, result2);
        return ret;
    }

    inline
    void load(const float *data)
    {
        val1 = vld1q_f32((data + 0));
        val2 = vld1q_f32((data + 4));
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        load(data);
    }

    inline
    void store(float *data) const
    {
        vst1q_f32(data, val1);
        vst1q_f32(data + 4, val2);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        store(data);
    }

    inline
    void store_nt(float *data) const
    {
        // in arm only stnp support non-temporal hint, thus need to
        // break into two registers. (use helper val2)
        // see if it can get optimized by compiler
        
        // the mapping between Q registers and D registers

        // stnp is for arm 64 (armv8)
        #if __LP64__
            register float32x4_t tmp1 asm ("q0");
            tmp1 = val1;
            register float32x4_t tmp2 asm ("q1");
            tmp2 = val2;
            asm("stnp d0, d1, %[store]"
                :[store] "=m" (data)
            );
            asm("stnp d2, d3, %[store]"
                :[store] "=m" (data + 4)
            );
        #else
            store(data);
        #endif
    }

    // dummy approach. NEON only supports loading in fixed interleaving
    inline
    void gather(const float *ptr, const unsigned *offsets)
    {
        float data[8];
        data[0] = ptr[offsets[0]];
        data[1] = ptr[offsets[1]];
        data[2] = ptr[offsets[2]];
        data[3] = ptr[offsets[3]];
        data[4] = ptr[offsets[4]];
        data[5] = ptr[offsets[5]];
        data[6] = ptr[offsets[6]];
        data[7] = ptr[offsets[7]];
        load(data);
    }

    // dummy approach
    inline
    void scatter(float *ptr, const unsigned *offsets) const
    {
        const float *data1 = reinterpret_cast<const float *>(&val1);
        const float *data2 = reinterpret_cast<const float *>(&val2);
        ptr[offsets[0]] = data1[0];
        ptr[offsets[1]] = data1[1];
        ptr[offsets[2]] = data1[2];
        ptr[offsets[3]] = data1[3];
        ptr[offsets[4]] = data2[0];
        ptr[offsets[5]] = data2[1];
        ptr[offsets[6]] = data2[2];
        ptr[offsets[7]] = data2[3];
    }

private:
    float32x4_t val1;
    float32x4_t val2;
};

inline
void operator<<(float *data, const short_vec<float, 8>& vec)
{
    vec.store(data);
}

inline
    short_vec<float, 8> sqrt(const short_vec<float, 8>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 8>& vec)
{
    const float *data1 = reinterpret_cast<const float *>(&vec.val1);
    const float *data2 = reinterpret_cast<const float *>(&vec.val2);
    __os << "["
        << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
        << ", "
        << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
        << "]";
    return __os;
}

}

#endif
#endif

#endif
