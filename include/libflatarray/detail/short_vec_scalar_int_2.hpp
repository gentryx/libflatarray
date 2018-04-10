/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_2_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_2_HPP

#include <libflatarray/config.h>
#include <libflatarray/short_vec_base.hpp>

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibFlatArray {

template<typename CARGO, std::size_t ARITY>
class short_vec;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

// Don't warn about these functions being stripped from an executable
// as they're not being used, that's actually expected behavior.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

template<>
class short_vec<int, 2> : public short_vec_base<int, 2>
{
public:
    static const std::size_t ARITY = 2;

    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 2>& vec);

    inline
    short_vec(const int data = 0) :
        val{data, data}
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

    inline
    short_vec(const int val1, const int val2) :
        val{val1,
            val2}
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
        val[ 0] -= other.val[ 0];
        val[ 1] -= other.val[ 1];
    }

    inline
    short_vec<int, 2> operator-(const short_vec<int, 2>& other) const
    {
        return short_vec<int, 2>(
            val[ 0] - other.val[ 0],
            val[ 1] - other.val[ 1]);
    }

    inline
    void operator+=(const short_vec<int, 2>& other)
    {
        val[ 0] += other.val[ 0];
        val[ 1] += other.val[ 1];
    }

    inline
    short_vec<int, 2> operator+(const short_vec<int, 2>& other) const
    {
        return short_vec<int, 2>(
            val[ 0] + other.val[ 0],
            val[ 1] + other.val[ 1]);
    }

    inline
    void operator*=(const short_vec<int, 2>& other)
    {
        val[ 0] *= other.val[ 0];
        val[ 1] *= other.val[ 1];
    }

    inline
    short_vec<int, 2> operator*(const short_vec<int, 2>& other) const
    {
        return short_vec<int, 2>(
            val[ 0] * other.val[ 0],
            val[ 1] * other.val[ 1]);
    }

    inline
    void operator/=(const short_vec<int, 2>& other)
    {
        val[ 0] /= other.val[ 0];
        val[ 1] /= other.val[ 1];
    }

    inline
    short_vec<int, 2> operator/(const short_vec<int, 2>& other) const
    {
        return short_vec<int, 2>(
            val[ 0] / other.val[ 0],
            val[ 1] / other.val[ 1]);
    }

    inline
    short_vec<int, 2> sqrt() const
    {
        return short_vec<int, 2>(
            static_cast<int>(std::sqrt(val[ 0])),
            static_cast<int>(std::sqrt(val[ 1])));
    }

    inline
    void load(const int *data)
    {
        val[ 0] = data[0];
        val[ 1] = data[1];
    }

    inline
    void load_aligned(const int *data)
    {
        load(data);
    }

    inline
    void store(int *data) const
    {
        *(data +  0) = val[ 0];
        *(data +  1) = val[ 1];
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
    void gather(const int *ptr, const int *offsets)
    {
        val[ 0] = ptr[offsets[0]];
        val[ 1] = ptr[offsets[1]];
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = val[ 0];
        ptr[offsets[1]] = val[ 1];
    }

private:
    int val[2];
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

// not inlining is ok, as is inlining:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 4711 )
#endif

inline
template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<int, 2>& vec)
{
    __os << "["  << vec.val[ 0] << ", " << vec.val[ 1]
         << "]";
    return __os;
}

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#endif
