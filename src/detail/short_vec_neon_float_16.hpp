#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_16_HPP

#ifdef __ARM_NEON__
#include <arm_neon.h>
#include <stdlib.h>
#include <libflatarray/detail/short_vec_helpers.hpp>

#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

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
            vmulq_f32(val3, other.val3), vmulq_f32(val4, other.val4)
            );
        return ret;
    }

    inline
    void operator/=(const short_vec<float, 16>& other)
    {
        val1 = vmulq_f32(val1, vrecpeq_f32(other.val1));
        val2 = vmulq_f32(val2, vrecpeq_f32(other.val2));
        val3 = vmulq_f32(val3, vrecpeq_f32(other.val3));
        val4 = vmulq_f32(val4, vrecpeq_f32(other.val4));
    }

    inline
    short_vec<float, 16> operator/(const short_vec<float, 16>& other) const
    {
        short_vec<float, 16> ret(
            vmulq_f32(val1, vrecpeq_f32(other.val1)),
            vmulq_f32(val2, vrecpeq_f32(other.val2)),
            vmulq_f32(val3, vrecpeq_f32(other.val3)),
            vmulq_f32(val4, vrecpeq_f32(other.val4))
            );
        return ret;
    }

    inline
    short_vec<float, 16> sqrt() const
    {
        // seems that vsqrtq_f32 is not yet implemented in the compiler
        //short_vec<float, 16> ret(vsqrtq_f32(val1));
        float32x4_t reciprocal1 = vrsqrteq_f32(val1);
        float32x4_t reciprocal2 = vrsqrteq_f32(val2);
        float32x4_t reciprocal3 = vrsqrteq_f32(val3);
        float32x4_t reciprocal4 = vrsqrteq_f32(val4);
        short_vec<float, 16> ret(
            vmulq_f32(val1, reciprocal1), vmulq_f32(val2, reciprocal2),
            vmulq_f32(val3, reciprocal3), vmulq_f32(val4, reciprocal4)
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
    void gather(const float *ptr, const unsigned *offsets)
    {
        float * data = (float *) malloc(16 * sizeof(float));
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
    void scatter(float *ptr, const unsigned *offsets) const
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

#endif
