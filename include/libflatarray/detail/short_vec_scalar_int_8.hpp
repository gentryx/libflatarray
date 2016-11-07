/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_8_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SCALAR) ||          \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_ARM_NEON) ||        \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_MIC) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_QPX)

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
class short_vec<int, 8>
{
public:
    static const int ARITY = 8;

    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 8>& vec);

    inline
    short_vec(const int data = 0) :
        val1(data),
        val2(data),
        val3(data),
        val4(data),
        val5(data),
        val6(data),
        val7(data),
        val8(data)
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
        const int val4,
        const int val5,
        const int val6,
        const int val7,
        const int val8) :
        val1( val1),
        val2( val2),
        val3( val3),
        val4( val4),
        val5( val5),
        val6( val6),
        val7( val7),
        val8( val8)
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
    void operator-=(const short_vec<int, 8>& other)
    {
        val1  -= other.val1;
        val2  -= other.val2;
        val3  -= other.val3;
        val4  -= other.val4;
        val5  -= other.val5;
        val6  -= other.val6;
        val7  -= other.val7;
        val8  -= other.val8;
    }

    inline
    short_vec<int, 8> operator-(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            val1  - other.val1,
            val2  - other.val2,
            val3  - other.val3,
            val4  - other.val4,
            val5  - other.val5,
            val6  - other.val6,
            val7  - other.val7,
            val8  - other.val8);
    }

    inline
    void operator+=(const short_vec<int, 8>& other)
    {
        val1  += other.val1;
        val2  += other.val2;
        val3  += other.val3;
        val4  += other.val4;
        val5  += other.val5;
        val6  += other.val6;
        val7  += other.val7;
        val8  += other.val8;
    }

    inline
    short_vec<int, 8> operator+(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            val1  + other.val1,
            val2  + other.val2,
            val3  + other.val3,
            val4  + other.val4,
            val5  + other.val5,
            val6  + other.val6,
            val7  + other.val7,
            val8  + other.val8);
    }

    inline
    void operator*=(const short_vec<int, 8>& other)
    {
        val1  *= other.val1;
        val2  *= other.val2;
        val3  *= other.val3;
        val4  *= other.val4;
        val5  *= other.val5;
        val6  *= other.val6;
        val7  *= other.val7;
        val8  *= other.val8;
    }

    inline
    short_vec<int, 8> operator*(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            val1  * other.val1,
            val2  * other.val2,
            val3  * other.val3,
            val4  * other.val4,
            val5  * other.val5,
            val6  * other.val6,
            val7  * other.val7,
            val8  * other.val8);
    }

    inline
    void operator/=(const short_vec<int, 8>& other)
    {
        val1  /= other.val1;
        val2  /= other.val2;
        val3  /= other.val3;
        val4  /= other.val4;
        val5  /= other.val5;
        val6  /= other.val6;
        val7  /= other.val7;
        val8  /= other.val8;
    }

    inline
    short_vec<int, 8> operator/(const short_vec<int, 8>& other) const
    {
        return short_vec<int, 8>(
            val1  / other.val1,
            val2  / other.val2,
            val3  / other.val3,
            val4  / other.val4,
            val5  / other.val5,
            val6  / other.val6,
            val7  / other.val7,
            val8  / other.val8);
    }

    inline
    short_vec<int, 8> sqrt() const
    {
        return short_vec<int, 8>(
            static_cast<int>(std::sqrt(val1)),
            static_cast<int>(std::sqrt(val2)),
            static_cast<int>(std::sqrt(val3)),
            static_cast<int>(std::sqrt(val4)),
            static_cast<int>(std::sqrt(val5)),
            static_cast<int>(std::sqrt(val6)),
            static_cast<int>(std::sqrt(val7)),
            static_cast<int>(std::sqrt(val8)));
    }

    inline
    void load(const int *data)
    {
        val1 = data[0];
        val2 = data[1];
        val3 = data[2];
        val4 = data[3];
        val5 = data[4];
        val6 = data[5];
        val7 = data[6];
        val8 = data[7];
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
        *(data +  4) = val5;
        *(data +  5) = val6;
        *(data +  6) = val7;
        *(data +  7) = val8;
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
        val5 = ptr[offsets[4]];
        val6 = ptr[offsets[5]];
        val7 = ptr[offsets[6]];
        val8 = ptr[offsets[7]];
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = val1;
        ptr[offsets[1]] = val2;
        ptr[offsets[2]] = val3;
        ptr[offsets[3]] = val4;
        ptr[offsets[4]] = val5;
        ptr[offsets[5]] = val6;
        ptr[offsets[6]] = val7;
        ptr[offsets[7]] = val8;
    }

private:
    int val1;
    int val2;
    int val3;
    int val4;
    int val5;
    int val6;
    int val7;
    int val8;
};

inline
void operator<<(int *data, const short_vec<int, 8>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<int, 8> sqrt(const short_vec<int, 8>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<int, 8>& vec)
{
    __os << "["  << vec.val1  << ", " << vec.val2  << ", " << vec.val3  << ", " << vec.val4
         << ", " << vec.val5  << ", " << vec.val6  << ", " << vec.val7  << ", " << vec.val8
         << "]";
    return __os;
}

}

#endif

#endif
