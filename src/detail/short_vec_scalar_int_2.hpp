/**
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_2_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_2_HPP

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
class short_vec<int, 2>
{
public:
    static const int ARITY = 2;

    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 2>& vec);

    inline
    short_vec(const int data = 0) :
        val1(data),
        val2(data)
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

    inline
    short_vec(const int val1, const int val2) :
        val1(val1),
        val2(val2)
    {}

#ifdef LIBFLATARRAY_WITH_CPP14
    inline
    short_vec(const std::initializer_list<int>& il)
    {
        const int *ptr = static_cast<const int *>(&(*il.begin()));
        load(ptr);
    }
#endif

    inline
    void operator-=(const short_vec<int, 2>& other)
    {
        val1 -= other.val1;
        val2 -= other.val2;
    }

    inline
    short_vec<int, 2> operator-(const short_vec<int, 2>& other) const
    {
        return short_vec<int, 2>(
            val1 - other.val1,
            val2 - other.val2);
    }

    inline
    void operator+=(const short_vec<int, 2>& other)
    {
        val1 += other.val1;
        val2 += other.val2;
    }

    inline
    short_vec<int, 2> operator+(const short_vec<int, 2>& other) const
    {
        return short_vec<int, 2>(
            val1 + other.val1,
            val2 + other.val2);
    }

    inline
    void operator*=(const short_vec<int, 2>& other)
    {
        val1 *= other.val1;
        val2 *= other.val2;
    }

    inline
    short_vec<int, 2> operator*(const short_vec<int, 2>& other) const
    {
        return short_vec<int, 2>(
            val1 * other.val1,
            val2 * other.val2);
    }

    inline
    void operator/=(const short_vec<int, 2>& other)
    {
        val1 /= other.val1;
        val2 /= other.val2;
    }

    inline
    short_vec<int, 2> operator/(const short_vec<int, 2>& other) const
    {
        return short_vec<int, 2>(
            val1 / other.val1,
            val2 / other.val2);
    }

    inline
    short_vec<int, 2> sqrt() const
    {
        return short_vec<int, 2>(
            std::sqrt(val1),
            std::sqrt(val2));
    }

    inline
    void load(const int *data)
    {
        val1 = data[0];
        val2 = data[1];
    }

    inline
    void load_aligned(const int *data)
    {
        load(data);
    }

    inline
    void store(int *data) const
    {
        *(data +  0) = val1;
        *(data +  1) = val2;
    }

    inline
    void store_aligned(int *data) const
    {
        store(data);
    }

    inline
    void store_nt(int *data) const
    {
        store(data);
    }

    inline
    void gather(const int *ptr, const unsigned *offsets)
    {
        val1 = ptr[offsets[0]];
        val2 = ptr[offsets[1]];
    }

    inline
    void scatter(int *ptr, const unsigned *offsets) const
    {
        ptr[offsets[0]] = val1;
        ptr[offsets[1]] = val2;
    }

private:
    int val1;
    int val2;
};

inline
void operator<<(int *data, const short_vec<int, 2>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<int, 2> sqrt(const short_vec<int, 2>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<int, 2>& vec)
{
    __os << "["  << vec.val1 << ", " << vec.val2
         << "]";
    return __os;
}

}

#endif
