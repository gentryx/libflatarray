/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_QPX_DOUBLE_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_QPX_DOUBLE_8_HPP

#ifdef __VECTOR4DOUBLE__

#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>

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
class short_vec<double, 8>
{
public:
    static const int ARITY = 8;

    inline
    short_vec(const double data = 0) :
        val1(vec_splats(data)),
        val2(vec_splats(data))
    {}

    inline
    short_vec(const double *data) :
        val1(vec_ld(0, const_cast<double *>(data + 0))),
        val2(vec_ld(0, const_cast<double *>(data + 4)))
    {}

    inline
    short_vec(const vector4double& val1, const vector4double& val2) :
        val1(val1),
        val2(val2)
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
    short_vec(const sqrt_reference<double, 8>& other);

    inline
    void operator-=(const short_vec<double, 8>& other)
    {
        val1 = vec_sub(val1, other.val1);
        val2 = vec_sub(val2, other.val2);
    }

    inline
    short_vec<double, 8> operator-(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            vec_sub(val1, other.val1),
            vec_sub(val2, other.val2));
    }

    inline
    void operator+=(const short_vec<double, 8>& other)
    {
        val1 = vec_add(val1, other.val1);
        val2 = vec_add(val2, other.val2);
    }

    inline
    short_vec<double, 8> operator+(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            vec_add(val1, other.val1),
            vec_add(val2, other.val2));
    }

    inline
    void operator*=(const short_vec<double, 8>& other)
    {
        val1 = vec_add(val1, other.val1);
        val2 = vec_add(val2, other.val2);
    }

    inline
    short_vec<double, 8> operator*(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            vec_mul(val1, other.val1),
            vec_mul(val2, other.val2));
    }

    inline
    void operator/=(const sqrt_reference<double, 8>& other);

    inline
    void operator/=(const short_vec<double, 8>& other)
    {
        val1 = vec_swdiv_nochk(val1, other.val1);
        val2 = vec_swdiv_nochk(val2, other.val2);
    }

    inline
    short_vec<double, 8> operator/(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            vec_swdiv_nochk(val1, other.val1),
            vec_swdiv_nochk(val2, other.val2));
    }

    inline
    short_vec<double, 8> operator/(const sqrt_reference<double, 8>& other) const;

    inline
    short_vec<double, 8> sqrt() const
    {
        return short_vec<double, 8>(
            vec_swsqrt(val1),
            vec_swsqrt(val2));
    }

    inline
    void load(const double *data)
    {
        val1 = vec_ld(0, const_cast<double *>(data + 0));
        val2 = vec_ld(0, const_cast<double *>(data + 4));
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        val1 = vec_lda(0, const_cast<double*>(data + 0));
        val2 = vec_lda(0, const_cast<double*>(data + 4));
    }

    inline
    void store(double *data) const
    {
        vec_st(val1, 0, data + 0);
        vec_st(val2, 0, data + 4);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        vec_sta(val1, 0, data + 0);
        vec_sta(val2, 0, data + 4);
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
        val1 = vec_insert(base[offsets[0]], val1, 0);
        val1 = vec_insert(base[offsets[1]], val1, 1);
        val1 = vec_insert(base[offsets[2]], val1, 2);
        val1 = vec_insert(base[offsets[3]], val1, 3);

        val2 = vec_insert(base[offsets[4]], val2, 0);
        val2 = vec_insert(base[offsets[5]], val2, 1);
        val2 = vec_insert(base[offsets[6]], val2, 2);
        val2 = vec_insert(base[offsets[7]], val2, 3);
    }

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = vec_extract(val1, 0);
        ptr[offsets[1]] = vec_extract(val1, 1);
        ptr[offsets[2]] = vec_extract(val1, 2);
        ptr[offsets[3]] = vec_extract(val1, 3);

        ptr[offsets[4]] = vec_extract(val2, 0);
        ptr[offsets[5]] = vec_extract(val2, 1);
        ptr[offsets[6]] = vec_extract(val2, 2);
        ptr[offsets[7]] = vec_extract(val2, 3);
    }

private:
    vector4double val1;
    vector4double val2;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(double *data, const short_vec<double, 8>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<double, 8>
{
public:
    template<typename OTHER_CARGO, int OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<double, 8>& vec) :
        vec(vec)
    {}

private:
    short_vec<double, 8> vec;
};

inline
short_vec<double, 8>::short_vec(const sqrt_reference<double, 8>& other) :
    val1(vec_swsqrt(other.vec.val1)),
    val2(vec_swsqrt(other.vec.val2))
{}

inline
void short_vec<double, 8>::operator/=(const sqrt_reference<double, 8>& other)
{
    val1 = vec_mul(val1, vec_rsqrte(other.vec.val1));
    val2 = vec_mul(val2, vec_rsqrte(other.vec.val2));
}

inline
short_vec<double, 8> short_vec<double, 8>::operator/(const sqrt_reference<double, 8>& other) const
{
    return short_vec<double, 8>(
        vec_mul(val1, vec_rsqrte(other.vec.val1)),
        vec_mul(val2, vec_rsqrte(other.vec.val2)));
}

inline
sqrt_reference<double, 8> sqrt(const short_vec<double, 8>& vec)
{
    return sqrt_reference<double, 8>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 8>& vec)
{
    const double *data1 = reinterpret_cast<const double *>(&vec.val1);
    const double *data2 = reinterpret_cast<const double *>(&vec.val2);
    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << ", " << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
         << "]";
    return __os;
}

}

#endif
#endif

#endif
