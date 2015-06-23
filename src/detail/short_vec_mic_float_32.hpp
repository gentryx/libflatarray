#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_MIC_FLOAT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_MIC_FLOAT_32_HPP

#ifdef __MIC__

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

#ifndef __CUDA_ARCH__

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
class short_vec<float, 32>
{
public:
    static const int ARITY = 32;

    typedef short_vec_strategy::mic strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 32>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm512_set1_ps(data)),
        val2(_mm512_set1_ps(data))
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(const __m512& val1, const __m512& val2) :
        val1(val1), val2(val2)
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
    short_vec(const sqrt_reference<float, 32>& other);

    inline
    void operator-=(const short_vec<float, 32>& other)
    {
        val1 = _mm512_sub_ps(val1, other.val1);
        val2 = _mm512_sub_ps(val2, other.val2);
    }

    inline
    short_vec<float, 32> operator-(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm512_sub_ps(val1, other.val1),
            _mm512_sub_ps(val2, other.val2));
    }

    inline
    void operator+=(const short_vec<float, 32>& other)
    {
        val1 = _mm512_add_ps(val1, other.val1);
        val2 = _mm512_add_ps(val2, other.val2);
    }

    inline
    short_vec<float, 32> operator+(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm512_add_ps(val1, other.val1),
            _mm512_add_ps(val2, other.val2));
    }

    inline
    void operator*=(const short_vec<float, 32>& other)
    {
        val1 = _mm512_mul_ps(val1, other.val1);
        val2 = _mm512_mul_ps(val2, other.val2);
    }

    inline
    short_vec<float, 32> operator*(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm512_mul_ps(val1, other.val1),
            _mm512_mul_ps(val2, other.val2));
    }

    inline
    void operator/=(const short_vec<float, 32>& other)
    {
        val1 = _mm512_mul_ps(val1, _mm512_rcp23_ps(other.val1));
        val2 = _mm512_mul_ps(val2, _mm512_rcp23_ps(other.val2));
    }

    inline
    void operator/=(const sqrt_reference<float, 32>& other);

    inline
    short_vec<float, 32> operator/(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
            _mm512_mul_ps(val1, _mm512_rcp23_ps(other.val1)),
            _mm512_mul_ps(val2, _mm512_rcp23_ps(other.val2)));
    }

    inline
    short_vec<float, 32> operator/(const sqrt_reference<float, 32>& other) const;

    inline
    short_vec<float, 32> sqrt() const
    {
        return short_vec<float, 32>(
            _mm512_sqrt_ps(val1),
            _mm512_sqrt_ps(val2));
    }

    inline
    void load(const float *data)
    {
        val1 = _mm512_loadunpacklo_ps(val1, data +   0);
        val1 = _mm512_loadunpackhi_ps(val1, data +  16);
        val2 = _mm512_loadunpacklo_ps(val2, data +  16);
        val2 = _mm512_loadunpackhi_ps(val2, data +  32);
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        val1 = _mm512_load_ps(data +  0);
        val2 = _mm512_load_ps(data + 16);
    }

    inline
    void store(float *data) const
    {
        _mm512_packstorelo_ps(data +   0, val1);
        _mm512_packstorehi_ps(data +  16, val1);
        _mm512_packstorelo_ps(data +  16, val2);
        _mm512_packstorehi_ps(data +  32, val2);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_store_ps(data +  0, val1);
        _mm512_store_ps(data + 16, val2);
    }

    inline
    void store_nt(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_storenr_ps(data +  0, val1);
        _mm512_storenr_ps(data + 16, val2);
    }

    inline
    void gather(const float *ptr, const unsigned *offsets)
    {
        __m512i indices;
        SHORTVEC_ASSERT_ALIGNED(offsets, 64);
        indices = _mm512_load_epi32(offsets + 0);
        val1    = _mm512_i32gather_ps(indices, ptr, 4);
        indices = _mm512_load_epi32(offsets + 16);
        val2    = _mm512_i32gather_ps(indices, ptr, 4);
    }

    inline
    void scatter(float *ptr, const unsigned *offsets) const
    {
        __m512i indices;
        SHORTVEC_ASSERT_ALIGNED(offsets, 64);
        indices = _mm512_load_epi32(offsets + 0);
        _mm512_i32scatter_ps(ptr, indices, val1, 4);
        indices = _mm512_load_epi32(offsets + 16);
        _mm512_i32scatter_ps(ptr, indices, val2, 4);
    }

private:
    __m512 val1;
    __m512 val2;
};

inline
void operator<<(float *data, const short_vec<float, 32>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<float, 32>
{
public:
    template<typename OTHER_CARGO, int OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<float, 32>& vec) :
        vec(vec)
    {}

private:
    short_vec<float, 32> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<float, 32>::short_vec(const sqrt_reference<float, 32>& other) :
    val1(_mm512_sqrt_ps(other.vec.val1)),
    val2(_mm512_sqrt_ps(other.vec.val2))
{}

inline
void short_vec<float, 32>::operator/=(const sqrt_reference<float, 32>& other)
{
    val1 = _mm512_mul_ps(val1, _mm512_rsqrt23_ps(other.vec.val1));
    val2 = _mm512_mul_ps(val2, _mm512_rsqrt23_ps(other.vec.val2));
}

inline
short_vec<float, 32> short_vec<float, 32>::operator/(const sqrt_reference<float, 32>& other) const
{
    return short_vec<float, 32>(
        _mm512_mul_ps(val1, _mm512_rsqrt23_ps(other.vec.val1)),
        _mm512_mul_ps(val2, _mm512_rsqrt23_ps(other.vec.val2)));
}

inline
sqrt_reference<float, 32> sqrt(const short_vec<float, 32>& vec)
{
    return sqrt_reference<float, 32>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 32>& vec)
{
    const float *data1 = reinterpret_cast<const float *>(&vec.val1);
    const float *data2 = reinterpret_cast<const float *>(&vec.val2);
    __os << "["  << data1[ 0] << ", " << data1[ 1] << ", " << data1[ 2] << ", " << data1[ 3]
         << ", " << data1[ 4] << ", " << data1[ 5] << ", " << data1[ 6] << ", " << data1[ 7]
         << ", " << data1[ 8] << ", " << data1[ 9] << ", " << data1[10] << ", " << data1[11]
         << ", " << data1[12] << ", " << data1[13] << ", " << data1[14] << ", " << data1[15]
         << ", " << data2[ 0] << ", " << data2[ 1] << ", " << data2[ 2] << ", " << data2[ 3]
         << ", " << data2[ 4] << ", " << data2[ 5] << ", " << data2[ 6] << ", " << data2[ 7]
         << ", " << data2[ 8] << ", " << data2[ 9] << ", " << data2[10] << ", " << data2[11]
         << ", " << data2[12] << ", " << data2[13] << ", " << data2[14] << ", " << data2[15]
         << "]";
    return __os;
}

}

#endif
#endif

#endif
