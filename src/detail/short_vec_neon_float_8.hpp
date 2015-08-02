#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_8_HPP

#ifdef __ARM_NEON__
#include <arm_neon.h>
#include <stdlib.h>
#include <libflatarray/detail/short_vec_helpers.hpp>

#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
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

    inline
    void operator/=(const short_vec<float, 8>& other)
    {
        val1 = vmulq_f32(val1, vrecpeq_f32(other.val1));
        val2 = vmulq_f32(val2, vrecpeq_f32(other.val2));
    }

    inline
    short_vec<float, 8> operator/(const short_vec<float, 8>& other) const
    {
        short_vec<float, 8> ret(
            vmulq_f32(val1, vrecpeq_f32(other.val1)),
            vmulq_f32(val2, vrecpeq_f32(other.val2))
            );
        return ret;
    }

    inline
    short_vec<float, 8> sqrt() const
    {
        // seems that vsqrtq_f32 is not yet implemented in the compiler
        //short_vec<float, 8> ret(vsqrtq_f32(val1));
        float32x4_t reciprocal1 = vrsqrteq_f32(val1);
        float32x4_t reciprocal2 = vrsqrteq_f32(val2);
        short_vec<float, 8> ret(
            vmulq_f32(val1, reciprocal1), vmulq_f32(val2, reciprocal2)
            );
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
            register float32x4_t tmp1 asm ("q1");
            tmp1 = val2;
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
        float * data = (float *) malloc(8 * sizeof(float));
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
