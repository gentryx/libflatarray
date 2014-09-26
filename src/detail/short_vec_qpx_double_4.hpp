/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_QPX_DOUBLE_4_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_QPX_DOUBLE_4_HPP

#ifdef __VECTOR4DOUBLE__

#include <libflatarray/detail/sqrt_reference.hpp>

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
class short_vec<double, 4>
{
public:
    static const int ARITY = 4;

    inline
    short_vec(const double& data = 0) :
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

    inline
    short_vec(const sqrt_reference<double, 4> other);

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
    void store(double *data) const
    {
        vec_st(val1, 0, data);
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
    template<typename OTHER_CARGO, int OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<double, 4>& vec) :
        vec(vec)
    {}

private:
    short_vec<double, 4> vec;
};

inline
short_vec<double, 4>::short_vec(const sqrt_reference<double, 4> other) :
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

sqrt_reference<double, 4> sqrt(const short_vec<double, 4>& vec)
{
    return sqrt_reference<double, 4>(vec);
}

}

#endif
#endif

#endif
