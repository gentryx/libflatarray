#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_4_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_4_HPP

#ifdef __ARM_NEON__
#include <arm_neon.h>
#include <stdlib.h>
#include <libflatarray/detail/short_vec_helpers.hpp>

#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

template<>
class short_vec<float, 4>
{
public:
    static const int ARITY = 4;

    typedef short_vec_strategy::neon strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
            std::basic_ostream<_CharT, _Traits>& __os,
            const short_vec<float, 4>& vec);

    inline
    short_vec(const float data = 0) :
        val1(vdupq_n_f32(data))
    {}

    inline
    short_vec(const float *data) :
        val1(vld1q_f32( (data + 0) ))
    {}

    inline
    short_vec(const float32x4_t& val1) :
        val1(val1)
    {}

    inline
    void operator-=(const short_vec<float, 4>& other)
    {
        val1 = vsubq_f32(val1, other.val1);
    }

    inline
    short_vec<float, 4> operator-(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(vsubq_f32(val1, other.val1));
    }

    inline
    void operator+=(const short_vec<float, 4>& other)
    {
        val1 = vaddq_f32(val1, other.val1);
    }

    inline
    short_vec<float, 4> operator+(const short_vec<float, 4>& other) const
    {
        short_vec<float, 4> ret(vaddq_f32(val1, other.val1));
        return ret;
    }

    inline
    void operator*=(const short_vec<float, 4>& other)
    {
        val1 = vmulq_f32(val1, other.val1);
    }

    inline
    short_vec<float, 4> operator*(const short_vec<float, 4>& other) const
    {
        short_vec<float, 4> ret(vmulq_f32(val1, other.val1));
        return ret;
    }

    inline
    void operator/=(const short_vec<float, 4>& other)
    {
        val1 = vmulq_f32(val1, vrecpeq_f32(other.val1));
    }

    inline
    short_vec<float, 4> operator/(const short_vec<float, 4>& other) const
    {
        short_vec<float, 4> ret(vmulq_f32(val1, vrecpeq_f32(other.val1)));
        return ret;
    }

    inline
    short_vec<float, 4> sqrt() const
    {
        // seems that vsqrtq_f32 is not yet implemented in the compiler
        //short_vec<float, 4> ret(vsqrtq_f32(val1));
        float32x4_t reciprocal = vrsqrteq_f32(val1);
        short_vec<float, 4> ret(vmulq_f32(val1, reciprocal));
        return ret;
    }

    inline
    void load(const float *data)
    {
        val1 = vld1q_f32((data + 0));
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
        load(data);
    }

    inline
    void store(float *data) const
    {
        vst1q_f32(data, val1);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 16);
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
            register float32x4_t val2 asm ("q0");
            val2 = val1;
            asm("stnp d0, d1, %[store]"
                :[store] "=m" (data)
            );
        #else
            store(data);
        #endif
    }

    // dummy approach. NEON only supports loading in fixed interleaving
    inline
    void gather(const float *ptr, const unsigned *offsets)
    {
        float * data = (float *) malloc(4 * sizeof(float));
        data[0] = ptr[offsets[0]];
        data[1] = ptr[offsets[1]];
        data[2] = ptr[offsets[2]];
        data[3] = ptr[offsets[3]];
        load(data);
    }

    // dummy approach
    inline
    void scatter(float *ptr, const unsigned *offsets) const
    {
        const float *data1 = reinterpret_cast<const float *>(&val1);
        ptr[offsets[0]] = data1[0];
        ptr[offsets[1]] = data1[1];
        ptr[offsets[2]] = data1[2];
        ptr[offsets[3]] = data1[3];
    }

private:
    float32x4_t val1;
};

inline
void operator<<(float *data, const short_vec<float, 4>& vec)
{
    vec.store(data);
}

inline
    short_vec<float, 4> sqrt(const short_vec<float, 4>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 4>& vec)
{
    const float *data1 = reinterpret_cast<const float *>(&vec.val1);
    __os << "["  << data1[0]  << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3]
        << "]";
    return __os;
}

}

#endif
#endif

#endif
