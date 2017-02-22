/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_QPX_DOUBLE_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_QPX_DOUBLE_16_HPP

#if LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_QPX

#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>

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
class short_vec<double, 16>
{
public:
    static const std::size_t ARITY = 16;

    inline
    short_vec(const double data = 0) :
        val{vec_splats(data),
            vec_splats(data),
            vec_splats(data),
            vec_splats(data)}
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(
        const vector4double& val1,
        const vector4double& val2,
        const vector4double& val3,
        const vector4double& val4) :
        val{val1,
            val2,
            val3,
            val4}
    {}

#ifdef LIBFLATARRAY_WITH_CPP14
    inline
    short_vec(const std::initializer_list<double>& il)
    {
        const double *ptr = static_cast<const double *>(&(*il.begin()));
        load(ptr);
    }
#endif

    inline
    short_vec(const sqrt_reference<double, 16>& other);

    inline
    void operator-=(const short_vec<double, 16>& other)
    {
        val[ 0] = vec_sub(val[ 0], other.val[ 0]);
        val[ 1] = vec_sub(val[ 1], other.val[ 1]);
        val[ 2] = vec_sub(val[ 2], other.val[ 2]);
        val[ 3] = vec_sub(val[ 3], other.val[ 3]);
    }

    inline
    short_vec<double, 16> operator-(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            vec_sub(val[ 0], other.val[ 0]),
            vec_sub(val[ 1], other.val[ 1]),
            vec_sub(val[ 2], other.val[ 2]),
            vec_sub(val[ 3], other.val[ 3]));
    }

    inline
    void operator+=(const short_vec<double, 16>& other)
    {
        val[ 0] = vec_add(val[ 0], other.val[ 0]);
        val[ 1] = vec_add(val[ 1], other.val[ 1]);
        val[ 2] = vec_add(val[ 2], other.val[ 2]);
        val[ 3] = vec_add(val[ 3], other.val[ 3]);
    }

    inline
    short_vec<double, 16> operator+(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            vec_add(val[ 0], other.val[ 0]),
            vec_add(val[ 1], other.val[ 1]),
            vec_add(val[ 2], other.val[ 2]),
            vec_add(val[ 3], other.val[ 3]));
    }

    inline
    void operator*=(const short_vec<double, 16>& other)
    {
        val[ 0] = vec_add(val[ 0], other.val[ 0]);
        val[ 1] = vec_add(val[ 1], other.val[ 1]);
        val[ 2] = vec_add(val[ 2], other.val[ 2]);
        val[ 3] = vec_add(val[ 3], other.val[ 3]);
    }

    inline
    short_vec<double, 16> operator*(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            vec_mul(val[ 0], other.val[ 0]),
            vec_mul(val[ 1], other.val[ 1]),
            vec_mul(val[ 2], other.val[ 2]),
            vec_mul(val[ 3], other.val[ 3]));
    }

    inline
    void operator/=(const sqrt_reference<double, 16>& other);

    inline
    void operator/=(const short_vec<double, 16>& other)
    {
        val[ 0] = vec_swdiv_nochk(val[ 0], other.val[ 0]);
        val[ 1] = vec_swdiv_nochk(val[ 1], other.val[ 1]);
        val[ 2] = vec_swdiv_nochk(val[ 2], other.val[ 2]);
        val[ 3] = vec_swdiv_nochk(val[ 3], other.val[ 3]);
    }

    inline
    short_vec<double, 16> operator/(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            vec_swdiv_nochk(val[ 0], other.val[ 0]),
            vec_swdiv_nochk(val[ 1], other.val[ 1]),
            vec_swdiv_nochk(val[ 2], other.val[ 2]),
            vec_swdiv_nochk(val[ 3], other.val[ 3]));
    }

    inline
    short_vec<double, 16> operator/(const sqrt_reference<double, 16>& other) const;

    inline
    short_vec<double, 16> sqrt() const
    {
        return short_vec<double, 16>(
            vec_swsqrt(val[ 0]),
            vec_swsqrt(val[ 1]),
            vec_swsqrt(val[ 2]),
            vec_swsqrt(val[ 3]));
    }

    inline
    void load(const double *data)
    {
        val[ 0] = vec_ld(0, const_cast<double *>(data +  0));
        val[ 1] = vec_ld(0, const_cast<double *>(data +  4));
        val[ 2] = vec_ld(0, const_cast<double *>(data +  8));
        val[ 3] = vec_ld(0, const_cast<double *>(data + 12));
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        val[ 0] = vec_lda(0, const_cast<double *>(data +  0));
        val[ 1] = vec_lda(0, const_cast<double *>(data +  4));
        val[ 2] = vec_lda(0, const_cast<double *>(data +  8));
        val[ 3] = vec_lda(0, const_cast<double *>(data + 12));
    }

    inline
    void store(double *data) const
    {
        vec_st(val[ 0], 0, data +  0);
        vec_st(val[ 1], 0, data +  4);
        vec_st(val[ 2], 0, data +  8);
        vec_st(val[ 3], 0, data + 12);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        vec_sta(val[ 0], 0, data +  0);
        vec_sta(val[ 1], 0, data +  4);
        vec_sta(val[ 2], 0, data +  8);
        vec_sta(val[ 3], 0, data + 12);
    }

    inline
    void store_nt(double *data) const
    {
        store(data);
    }

    inline
    void gather(const double *ptr, const int *offsets)
    {
        double *base = const_cast<double *>(ptr);
        val[ 0] = vec_insert(base[offsets[ 0]], val[ 0], 0);
        val[ 0] = vec_insert(base[offsets[ 1]], val[ 0], 1);
        val[ 0] = vec_insert(base[offsets[ 2]], val[ 0], 2);
        val[ 0] = vec_insert(base[offsets[ 3]], val[ 0], 3);

        val[ 1] = vec_insert(base[offsets[ 4]], val[ 1], 0);
        val[ 1] = vec_insert(base[offsets[ 5]], val[ 1], 1);
        val[ 1] = vec_insert(base[offsets[ 6]], val[ 1], 2);
        val[ 1] = vec_insert(base[offsets[ 7]], val[ 1], 3);

        val[ 2] = vec_insert(base[offsets[ 8]], val[ 2], 0);
        val[ 2] = vec_insert(base[offsets[ 9]], val[ 2], 1);
        val[ 2] = vec_insert(base[offsets[10]], val[ 2], 2);
        val[ 2] = vec_insert(base[offsets[11]], val[ 2], 3);

        val[ 3] = vec_insert(base[offsets[12]], val[ 3], 0);
        val[ 3] = vec_insert(base[offsets[13]], val[ 3], 1);
        val[ 3] = vec_insert(base[offsets[14]], val[ 3], 2);
        val[ 3] = vec_insert(base[offsets[15]], val[ 3], 3);
    }

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        ptr[offsets[ 0]] = vec_extract(val[ 0], 0);
        ptr[offsets[ 1]] = vec_extract(val[ 0], 1);
        ptr[offsets[ 2]] = vec_extract(val[ 0], 2);
        ptr[offsets[ 3]] = vec_extract(val[ 0], 3);

        ptr[offsets[ 4]] = vec_extract(val[ 1], 0);
        ptr[offsets[ 5]] = vec_extract(val[ 1], 1);
        ptr[offsets[ 6]] = vec_extract(val[ 1], 2);
        ptr[offsets[ 7]] = vec_extract(val[ 1], 3);

        ptr[offsets[ 8]] = vec_extract(val[ 2], 0);
        ptr[offsets[ 9]] = vec_extract(val[ 2], 1);
        ptr[offsets[10]] = vec_extract(val[ 2], 2);
        ptr[offsets[11]] = vec_extract(val[ 2], 3);

        ptr[offsets[12]] = vec_extract(val[ 3], 0);
        ptr[offsets[13]] = vec_extract(val[ 3], 1);
        ptr[offsets[14]] = vec_extract(val[ 3], 2);
        ptr[offsets[15]] = vec_extract(val[ 3], 3);
    }

private:
    vector4double val[4];
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(double *data, const short_vec<double, 16>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<double, 16>
{
public:
    template<typename OTHER_CARGO, std::size_t OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<double, 16>& vec) :
        vec(vec)
    {}

private:
    short_vec<double, 16> vec;
};

inline
short_vec<double, 16>::short_vec(const sqrt_reference<double, 16>& other) :
    val[ 0](vec_swsqrt(other.vec.val[ 0])),
    val[ 1](vec_swsqrt(other.vec.val[ 1])),
    val[ 2](vec_swsqrt(other.vec.val[ 2])),
    val[ 3](vec_swsqrt(other.vec.val[ 3]))
{}

inline
void short_vec<double, 16>::operator/=(const sqrt_reference<double, 16>& other)
{
    val[ 0] = vec_mul(val[ 0], vec_rsqrte(other.vec.val[ 0]));
    val[ 1] = vec_mul(val[ 1], vec_rsqrte(other.vec.val[ 1]));
    val[ 2] = vec_mul(val[ 2], vec_rsqrte(other.vec.val[ 2]));
    val[ 3] = vec_mul(val[ 3], vec_rsqrte(other.vec.val[ 3]));
}

inline
short_vec<double, 16> short_vec<double, 16>::operator/(const sqrt_reference<double, 16>& other) const
{
    return short_vec<double, 16>(
        vec_mul(val[ 0], vec_rsqrte(other.vec.val[ 0])),
        vec_mul(val[ 1], vec_rsqrte(other.vec.val[ 1])),
        vec_mul(val[ 2], vec_rsqrte(other.vec.val[ 2])),
        vec_mul(val[ 3], vec_rsqrte(other.vec.val[ 3])));
}

inline
sqrt_reference<double, 16> sqrt(const short_vec<double, 16>& vec)
{
    return sqrt_reference<double, 16>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 16>& vec)
{
    const double *data1 = reinterpret_cast<const double *>(&vec.val[ 0]);
    const double *data2 = reinterpret_cast<const double *>(&vec.val[ 1]);
    const double *data3 = reinterpret_cast<const double *>(&vec.val[ 2]);
    const double *data4 = reinterpret_cast<const double *>(&vec.val[ 3]);
    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << ", " << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
         << ", " << data3[0] << ", " << data3[1] << ", " << data3[2] << ", " << data3[3]
         << ", " << data4[0] << ", " << data4[1] << ", " << data4[2] << ", " << data4[3]
         << "]";
    return __os;
}

}

#endif

#endif
