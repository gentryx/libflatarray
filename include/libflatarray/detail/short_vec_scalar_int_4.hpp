/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016 Andreas Schäfer
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

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

namespace LibFlatArray {

template<typename CARGO, std::size_t ARITY>
class short_vec;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<int, 4>
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
        val1(data),
        val2(data),
        val3(data),
        val4(data)
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
        val1( val1),
        val2( val2),
        val3( val3),
        val4( val4)
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
        val1  -= other.val1;
        val2  -= other.val2;
        val3  -= other.val3;
        val4  -= other.val4;
    }

    inline
    short_vec<int, 4> operator-(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            val1  - other.val1,
            val2  - other.val2,
            val3  - other.val3,
            val4  - other.val4);
    }

    inline
    void operator+=(const short_vec<int, 4>& other)
    {
        val1  += other.val1;
        val2  += other.val2;
        val3  += other.val3;
        val4  += other.val4;
    }

    inline
    short_vec<int, 4> operator+(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            val1  + other.val1,
            val2  + other.val2,
            val3  + other.val3,
            val4  + other.val4);
    }

    inline
    void operator*=(const short_vec<int, 4>& other)
    {
        val1  *= other.val1;
        val2  *= other.val2;
        val3  *= other.val3;
        val4  *= other.val4;
    }

    inline
    short_vec<int, 4> operator*(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            val1  * other.val1,
            val2  * other.val2,
            val3  * other.val3,
            val4  * other.val4);
    }

    inline
    void operator/=(const short_vec<int, 4>& other)
    {
        val1  /= other.val1;
        val2  /= other.val2;
        val3  /= other.val3;
        val4  /= other.val4;
    }

    inline
    short_vec<int, 4> operator/(const short_vec<int, 4>& other) const
    {
        return short_vec<int, 4>(
            val1  / other.val1,
            val2  / other.val2,
            val3  / other.val3,
            val4  / other.val4);
    }

    inline
    short_vec<int, 4> sqrt() const
    {
        return short_vec<int, 4>(
            static_cast<int>(std::sqrt(val1)),
            static_cast<int>(std::sqrt(val2)),
            static_cast<int>(std::sqrt(val3)),
            static_cast<int>(std::sqrt(val4)));
    }

    inline
    void load(const int *data)
    {
        val1 = data[0];
        val2 = data[1];
        val3 = data[2];
        val4 = data[3];
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
        *(data +  2) = val3;
        *(data +  3) = val4;
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
        val1 = ptr[offsets[0]];
        val2 = ptr[offsets[1]];
        val3 = ptr[offsets[2]];
        val4 = ptr[offsets[3]];
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = val1;
        ptr[offsets[1]] = val2;
        ptr[offsets[2]] = val3;
        ptr[offsets[3]] = val4;
    }

private:
    int val1;
    int val2;
    int val3;
    int val4;
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
    __os << "["  << vec.val1  << ", " << vec.val2  << ", " << vec.val3  << ", " << vec.val4
         << "]";
    return __os;
}

}

#endif

#endif
