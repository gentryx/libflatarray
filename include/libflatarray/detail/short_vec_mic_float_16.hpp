/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_MIC_FLOAT_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_MIC_FLOAT_16_HPP

#if LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_MIC

#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <immintrin.h>
#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

#ifdef _MSC_BUILD
#pragma warning( pop )
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
class short_vec<float, 16>
{
public:
    static const std::size_t ARITY = 16;

    typedef short_vec_strategy::mic strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 16>& vec);

    inline
    short_vec(const float data = 0) :
        val(_mm512_set1_ps(data))
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(const __m512& val) :
        val(val)
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
    void operator-=(const short_vec<float, 16>& other)
    {
        val = _mm512_sub_ps(val, other.val);
    }

    inline
    short_vec<float, 16> operator-(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm512_sub_ps(val, other.val));
    }

    inline
    void operator+=(const short_vec<float, 16>& other)
    {
        val = _mm512_add_ps(val, other.val);
    }

    inline
    short_vec<float, 16> operator+(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm512_add_ps(val, other.val));
    }

    inline
    void operator*=(const short_vec<float, 16>& other)
    {
        val = _mm512_mul_ps(val, other.val);
    }

    inline
    short_vec<float, 16> operator*(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm512_mul_ps(val, other.val));
    }

    inline
    void operator/=(const short_vec<float, 16>& other)
    {
        val = _mm512_mul_ps(val, _mm512_rcp23_ps(other.val));
    }

    inline
    void operator/=(const sqrt_reference<float, 16>& other);

    inline
    short_vec<float, 16> operator/(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
            _mm512_mul_ps(val, _mm512_rcp23_ps(other.val)));
    }

    inline
    short_vec<float, 16> operator/(const sqrt_reference<float, 16>& other) const;

    inline
    short_vec<float, 16> sqrt() const
    {
        return short_vec<float, 16>(
            _mm512_sqrt_ps(val));
    }

    inline
    void load(const float *data)
    {
        val = _mm512_loadunpacklo_ps(val, data +  0);
        val = _mm512_loadunpackhi_ps(val, data + 16);
    }

    inline
    void load_aligned(const float *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        val = _mm512_load_ps(data);
    }

    inline
    void store(float *data) const
    {
        _mm512_packstorelo_ps(data +  0, val);
        _mm512_packstorehi_ps(data + 16, val);
    }

    inline
    void store_aligned(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_store_ps(data, val);
    }

    inline
    void store_nt(float *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_storenr_ps(data, val);
    }

    inline
    void gather(const float *ptr, const int *offsets)
    {
        __m512i indices;
        SHORTVEC_ASSERT_ALIGNED(offsets, 64);
        indices = _mm512_load_epi32(offsets);
        val    = _mm512_i32gather_ps(indices, ptr, 4);
    }

    inline
    void scatter(float *ptr, const int *offsets) const
    {
        __m512i indices;
        SHORTVEC_ASSERT_ALIGNED(offsets, 64);
        indices = _mm512_load_epi32(offsets);
        _mm512_i32scatter_ps(ptr, indices, val, 4);
    }

private:
    __m512 val;
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
    val(_mm512_sqrt_ps(other.vec.val))
{}

inline
void short_vec<float, 16>::operator/=(const sqrt_reference<float, 16>& other)
{
    val = _mm512_mul_ps(val, _mm512_rsqrt23_ps(other.vec.val));
}

inline
short_vec<float, 16> short_vec<float, 16>::operator/(const sqrt_reference<float, 16>& other) const
{
    return short_vec<float, 16>(
        _mm512_mul_ps(val, _mm512_rsqrt23_ps(other.vec.val)));
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
    const float *data1 = reinterpret_cast<const float *>(&vec.val);
    __os << "["  << data1[ 0] << ", " << data1[ 1] << ", " << data1[ 2] << ", " << data1[ 3]
         << ", " << data1[ 4] << ", " << data1[ 5] << ", " << data1[ 6] << ", " << data1[ 7]
         << ", " << data1[ 8] << ", " << data1[ 9] << ", " << data1[10] << ", " << data1[11]
         << ", " << data1[12] << ", " << data1[13] << ", " << data1[14] << ", " << data1[15]
         << "]";
    return __os;
}

}

#endif

#endif
