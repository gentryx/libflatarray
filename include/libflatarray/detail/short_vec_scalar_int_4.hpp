/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_4_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_4_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SCALAR) ||          \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_ARM_NEON) ||        \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_MIC) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_QPX)

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
class short_vec<int, 4> : public short_vec_base<int, 4>
{
public:
    static const std::size_t ARITY = 4;

    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 4>& vec);

    inline
    short_vec(const int data = 0) :
        val{data,
            data,
            data,
            data}
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

    inline
    short_vec(
        const int val1,
        const int val2,
        const int val3,
        const int val4) :
        val{val1,
            val2,
            val3,
            val4}
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
    void operator-=(const short_vec<int, 4>& other)
    {
        val[ 0] -= other.val[ 0];
        val[ 1] -= other.val[ 1];
        val[ 2] -= other.val[ 2];
        val[ 3] -= other.val[ 3];
    }

    inline
    short_vec<int, 4> operator-(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            val[ 0] - other.val[ 0],
            val[ 1] - other.val[ 1],
            val[ 2] - other.val[ 2],
            val[ 3] - other.val[ 3]);
    }

    inline
    void operator+=(const short_vec<int, 4>& other)
    {
        val[ 0] += other.val[ 0];
        val[ 1] += other.val[ 1];
        val[ 2] += other.val[ 2];
        val[ 3] += other.val[ 3];
    }

    inline
    short_vec<int, 4> operator+(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            val[ 0] + other.val[ 0],
            val[ 1] + other.val[ 1],
            val[ 2] + other.val[ 2],
            val[ 3] + other.val[ 3]);
    }

    inline
    void operator*=(const short_vec<int, 4>& other)
    {
        val[ 0] *= other.val[ 0];
        val[ 1] *= other.val[ 1];
        val[ 2] *= other.val[ 2];
        val[ 3] *= other.val[ 3];
    }

    inline
    short_vec<int, 4> operator*(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            val[ 0] * other.val[ 0],
            val[ 1] * other.val[ 1],
            val[ 2] * other.val[ 2],
            val[ 3] * other.val[ 3]);
    }

    inline
    void operator/=(const short_vec<int, 4>& other)
    {
        val[ 0] /= other.val[ 0];
        val[ 1] /= other.val[ 1];
        val[ 2] /= other.val[ 2];
        val[ 3] /= other.val[ 3];
    }

    inline
    short_vec<int, 4> operator/(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            val[ 0] / other.val[ 0],
            val[ 1] / other.val[ 1],
            val[ 2] / other.val[ 2],
            val[ 3] / other.val[ 3]);
    }

    inline
    short_vec<int, 4> sqrt() const
    {
        return short_vec<int, 4>(
            static_cast<int>(std::sqrt(val[ 0])),
            static_cast<int>(std::sqrt(val[ 1])),
            static_cast<int>(std::sqrt(val[ 2])),
            static_cast<int>(std::sqrt(val[ 3])));
    }

    inline
    void load(const int *data)
    {
        val[ 0] = data[0];
        val[ 1] = data[1];
        val[ 2] = data[2];
        val[ 3] = data[3];
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
        *(data +  2) = val[ 2];
        *(data +  3) = val[ 3];
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
        val[ 2] = ptr[offsets[2]];
        val[ 3] = ptr[offsets[3]];
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = val[ 0];
        ptr[offsets[1]] = val[ 1];
        ptr[offsets[2]] = val[ 2];
        ptr[offsets[3]] = val[ 3];
    }

private:
    int val[4];
};

inline
void operator<<(int *data, const short_vec<int, 4>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<int, 4> sqrt(const short_vec<int, 4>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<int, 4>& vec)
{
    __os << "["
         << vec.val[ 0] << ", "
         << vec.val[ 1] << ", "
         << vec.val[ 2] << ", "
         << vec.val[ 3] << "]";
    return __os;
}

}

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#endif

#endif
