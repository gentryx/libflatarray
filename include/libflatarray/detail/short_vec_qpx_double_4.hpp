/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_QPX_DOUBLE_4_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_QPX_DOUBLE_4_HPP

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
class short_vec<double, 4>
{
public:
    static const std::size_t ARITY = 4;

    inline
    short_vec(const double data = 0) :
        val1(vec_splats(data))
    {}

    inline
    short_vec(const double *data) :
        val1(vec_ld(0, const_cast<double*>(data)))
    {}

    inline
    short_vec(const vector4double& val1) :
        val1(val1)
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
    short_vec(const sqrt_reference<double, 4>& other);

    inline
    void operator-=(const short_vec<double, 4>& other)
    {
        val1 = vec_sub(val1, other.val1);
    }

    inline
    short_vec<double, 4> operator-(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            vec_sub(val1, other.val1));
    }

    inline
    void operator+=(const short_vec<double, 4>& other)
    {
        val1 = vec_add(val1, other.val1);
    }

    inline
    short_vec<double, 4> operator+(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            vec_add(val1, other.val1));
    }

    inline
    void operator*=(const short_vec<double, 4>& other)
    {
        val1 = vec_add(val1, other.val1);
    }

    inline
    short_vec<double, 4> operator*(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            vec_mul(val1, other.val1));
    }

    inline
    void operator/=(const sqrt_reference<double, 4>& other);

    inline
    void operator/=(const short_vec<double, 4>& other)
    {
        val1 = vec_swdiv_nochk(val1, other.val1);
    }

    inline
    short_vec<double, 4> operator/(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            vec_swdiv_nochk(val1, other.val1));
    }

    inline
    short_vec<double, 4> operator/(const sqrt_reference<double, 4>& other) const;

    inline
    short_vec<double, 4> sqrt() const
    {
        return short_vec<double, 4>(
            vec_swsqrt(val1));
    }

    inline
    void load(const double *data)
    {
        val1 = vec_ld(0, const_cast<double*>(data));
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        val1 = vec_lda(0, const_cast<double*>(data));
    }

    inline
    void store(double *data) const
    {
        vec_st(val1, 0, data);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 32);
        vec_sta(val1, 0, data);
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
    }

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = vec_extract(val1, 0);
        ptr[offsets[1]] = vec_extract(val1, 1);
        ptr[offsets[2]] = vec_extract(val1, 2);
        ptr[offsets[3]] = vec_extract(val1, 3);
    }

private:
    vector4double val1;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(double *data, const short_vec<double, 4>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<double, 4>
{
public:
    template<typename OTHER_CARGO, std::size_t OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<double, 4>& vec) :
        vec(vec)
    {}

private:
    short_vec<double, 4> vec;
};

inline
short_vec<double, 4>::short_vec(const sqrt_reference<double, 4>& other) :
    val1(vec_swsqrt(other.vec.val1))
{}

inline
void short_vec<double, 4>::operator/=(const sqrt_reference<double, 4>& other)
{
    val1 = vec_mul(val1, vec_rsqrte(other.vec.val1));
}

inline
short_vec<double, 4> short_vec<double, 4>::operator/(const sqrt_reference<double, 4>& other) const
{
    return vec_mul(val1, vec_rsqrte(other.vec.val1));
}

inline
sqrt_reference<double, 4> sqrt(const short_vec<double, 4>& vec)
{
    return sqrt_reference<double, 4>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 4>& vec)
{
    const double *data1 = reinterpret_cast<const double *>(&vec.val1);
    __os << "[" << data1[0] << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3] << "]";
    return __os;
}

}

#endif

#endif
