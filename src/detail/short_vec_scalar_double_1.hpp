/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_DOUBLE_1_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_DOUBLE_1_HPP

#include <cmath>
#include <libflatarray/config.h>

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<double, 1>
{
public:
    static const int ARITY = 1;

    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 1>& vec);

    inline
    short_vec(const double data = 0) :
        val1(data)
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

#ifdef LIBFLATARRAY_WITH_CPP14
    inline
    short_vec(const std::initializer_list<double>& il)
    {
        const double *ptr = static_cast<const double *>(&(*il.begin()));
        load(ptr);
    }
#endif

    inline
    void operator-=(const short_vec<double, 1>& other)
    {
        val1 -= other.val1;
    }

    inline
    short_vec<double, 1> operator-(const short_vec<double, 1>& other) const
    {
        return short_vec<double, 1>(
            val1 - other.val1);
    }

    inline
    void operator+=(const short_vec<double, 1>& other)
    {
        val1 += other.val1;
    }

    inline
    short_vec<double, 1> operator+(const short_vec<double, 1>& other) const
    {
        return short_vec<double, 1>(
            val1 + other.val1);
    }

    inline
    void operator*=(const short_vec<double, 1>& other)
    {
        val1 *= other.val1;
    }

    inline
    short_vec<double, 1> operator*(const short_vec<double, 1>& other) const
    {
        return short_vec<double, 1>(
            val1 * other.val1);
    }

    inline
    void operator/=(const short_vec<double, 1>& other)
    {
        val1 /= other.val1;
    }

    inline
    short_vec<double, 1> operator/(const short_vec<double, 1>& other) const
    {
        return short_vec<double, 1>(
            val1 / other.val1);
    }

    inline
    short_vec<double, 1> sqrt() const
    {
        return short_vec<double, 1>(
            std::sqrt(val1));
    }

    inline
    void load(const double *data)
    {
        val1 = data[0];
    }

    inline
    void load_aligned(const double *data)
    {
        load(data);
    }

    inline
    void store(double *data) const
    {
        *(data + 0) = val1;
    }

    inline
    void store_aligned(double *data) const
    {
        store(data);
    }

    inline
    void store_nt(double *data) const
    {
        store(data);
    }

    inline
    void gather(const double *ptr, const int *offsets)
    {
        val1 = ptr[offsets[0]];
    }

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = val1;
    }

private:
    double val1;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(double *data, const short_vec<double, 1>& vec)
{
    vec.store(data);
}

inline
short_vec<double, 1> sqrt(const short_vec<double, 1>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 1>& vec)
{
    __os << "[" << vec.val1 << "]";
    return __os;
}

}

#endif
