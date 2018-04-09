/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016-2017 Andreas Schäfer
 * Copyright 2018 Google
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_32_HPP

#include <libflatarray/config.h>
#include <libflatarray/short_vec_base.hpp>
#include <libflatarray/detail/macros.hpp>

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

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SCALAR) ||          \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_ARM_NEON) ||        \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_MIC) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_QPX)

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
class short_vec<int, 32> : public short_vec_base<int, 32>
{
public:
    static const std::size_t ARITY = 32;

    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 32>& vec);

    inline
    short_vec(const int data = 0) :
        val{data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
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
        const int val4,
        const int val5,
        const int val6,
        const int val7,
        const int val8,
        const int val9,
        const int val10,
        const int val11,
        const int val12,
        const int val13,
        const int val14,
        const int val15,
        const int val16,
        const int val17,
        const int val18,
        const int val19,
        const int val20,
        const int val21,
        const int val22,
        const int val23,
        const int val24,
        const int val25,
        const int val26,
        const int val27,
        const int val28,
        const int val29,
        const int val30,
        const int val31,
        const int val32) :
        val{val1,
            val2,
            val3,
            val4,
            val5,
            val6,
            val7,
            val8,
            val9,
            val10,
            val11,
            val12,
            val13,
            val14,
            val15,
            val16,
            val17,
            val18,
            val19,
            val20,
            val21,
            val22,
            val23,
            val24,
            val25,
            val26,
            val27,
            val28,
            val29,
            val30,
            val31,
            val32}
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
    void operator-=(const short_vec<int, 32>& other)
    {
        val[ 0] -= other.val[ 0];
        val[ 1] -= other.val[ 1];
        val[ 2] -= other.val[ 2];
        val[ 3] -= other.val[ 3];
        val[ 4] -= other.val[ 4];
        val[ 5] -= other.val[ 5];
        val[ 6] -= other.val[ 6];
        val[ 7] -= other.val[ 7];
        val[ 8] -= other.val[ 8];
        val[ 9] -= other.val[ 9];
        val[10] -= other.val[10];
        val[11] -= other.val[11];
        val[12] -= other.val[12];
        val[13] -= other.val[13];
        val[14] -= other.val[14];
        val[15] -= other.val[15];
        val[16] -= other.val[16];
        val[17] -= other.val[17];
        val[18] -= other.val[18];
        val[19] -= other.val[19];
        val[20] -= other.val[20];
        val[21] -= other.val[21];
        val[22] -= other.val[22];
        val[23] -= other.val[23];
        val[24] -= other.val[24];
        val[25] -= other.val[25];
        val[26] -= other.val[26];
        val[27] -= other.val[27];
        val[28] -= other.val[28];
        val[29] -= other.val[29];
        val[30] -= other.val[30];
        val[31] -= other.val[31];
    }

    inline
    short_vec<int, 32> operator-(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            val[ 0] - other.val[ 0],
            val[ 1] - other.val[ 1],
            val[ 2] - other.val[ 2],
            val[ 3] - other.val[ 3],
            val[ 4] - other.val[ 4],
            val[ 5] - other.val[ 5],
            val[ 6] - other.val[ 6],
            val[ 7] - other.val[ 7],
            val[ 8] - other.val[ 8],
            val[ 9] - other.val[ 9],
            val[10] - other.val[10],
            val[11] - other.val[11],
            val[12] - other.val[12],
            val[13] - other.val[13],
            val[14] - other.val[14],
            val[15] - other.val[15],
            val[16] - other.val[16],
            val[17] - other.val[17],
            val[18] - other.val[18],
            val[19] - other.val[19],
            val[20] - other.val[20],
            val[21] - other.val[21],
            val[22] - other.val[22],
            val[23] - other.val[23],
            val[24] - other.val[24],
            val[25] - other.val[25],
            val[26] - other.val[26],
            val[27] - other.val[27],
            val[28] - other.val[28],
            val[29] - other.val[29],
            val[30] - other.val[30],
            val[31] - other.val[31]);
    }

    inline
    void operator+=(const short_vec<int, 32>& other)
    {
        val[ 0] += other.val[ 0];
        val[ 1] += other.val[ 1];
        val[ 2] += other.val[ 2];
        val[ 3] += other.val[ 3];
        val[ 4] += other.val[ 4];
        val[ 5] += other.val[ 5];
        val[ 6] += other.val[ 6];
        val[ 7] += other.val[ 7];
        val[ 8] += other.val[ 8];
        val[ 9] += other.val[ 9];
        val[10] += other.val[10];
        val[11] += other.val[11];
        val[12] += other.val[12];
        val[13] += other.val[13];
        val[14] += other.val[14];
        val[15] += other.val[15];
        val[16] += other.val[16];
        val[17] += other.val[17];
        val[18] += other.val[18];
        val[19] += other.val[19];
        val[20] += other.val[20];
        val[21] += other.val[21];
        val[22] += other.val[22];
        val[23] += other.val[23];
        val[24] += other.val[24];
        val[25] += other.val[25];
        val[26] += other.val[26];
        val[27] += other.val[27];
        val[28] += other.val[28];
        val[29] += other.val[29];
        val[30] += other.val[30];
        val[31] += other.val[31];
    }

    inline
    short_vec<int, 32> operator+(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            val[ 0] + other.val[ 0],
            val[ 1] + other.val[ 1],
            val[ 2] + other.val[ 2],
            val[ 3] + other.val[ 3],
            val[ 4] + other.val[ 4],
            val[ 5] + other.val[ 5],
            val[ 6] + other.val[ 6],
            val[ 7] + other.val[ 7],
            val[ 8] + other.val[ 8],
            val[ 9] + other.val[ 9],
            val[10] + other.val[10],
            val[11] + other.val[11],
            val[12] + other.val[12],
            val[13] + other.val[13],
            val[14] + other.val[14],
            val[15] + other.val[15],
            val[16] + other.val[16],
            val[17] + other.val[17],
            val[18] + other.val[18],
            val[19] + other.val[19],
            val[20] + other.val[20],
            val[21] + other.val[21],
            val[22] + other.val[22],
            val[23] + other.val[23],
            val[24] + other.val[24],
            val[25] + other.val[25],
            val[26] + other.val[26],
            val[27] + other.val[27],
            val[28] + other.val[28],
            val[29] + other.val[29],
            val[30] + other.val[30],
            val[31] + other.val[31]);
    }

    inline
    void operator*=(const short_vec<int, 32>& other)
    {
        val[ 0] *= other.val[ 0];
        val[ 1] *= other.val[ 1];
        val[ 2] *= other.val[ 2];
        val[ 3] *= other.val[ 3];
        val[ 4] *= other.val[ 4];
        val[ 5] *= other.val[ 5];
        val[ 6] *= other.val[ 6];
        val[ 7] *= other.val[ 7];
        val[ 8] *= other.val[ 8];
        val[ 9] *= other.val[ 9];
        val[10] *= other.val[10];
        val[11] *= other.val[11];
        val[12] *= other.val[12];
        val[13] *= other.val[13];
        val[14] *= other.val[14];
        val[15] *= other.val[15];
        val[16] *= other.val[16];
        val[17] *= other.val[17];
        val[18] *= other.val[18];
        val[19] *= other.val[19];
        val[20] *= other.val[20];
        val[21] *= other.val[21];
        val[22] *= other.val[22];
        val[23] *= other.val[23];
        val[24] *= other.val[24];
        val[25] *= other.val[25];
        val[26] *= other.val[26];
        val[27] *= other.val[27];
        val[28] *= other.val[28];
        val[29] *= other.val[29];
        val[30] *= other.val[30];
        val[31] *= other.val[31];
    }

    inline
    short_vec<int, 32> operator*(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            val[ 0] * other.val[ 0],
            val[ 1] * other.val[ 1],
            val[ 2] * other.val[ 2],
            val[ 3] * other.val[ 3],
            val[ 4] * other.val[ 4],
            val[ 5] * other.val[ 5],
            val[ 6] * other.val[ 6],
            val[ 7] * other.val[ 7],
            val[ 8] * other.val[ 8],
            val[ 9] * other.val[ 9],
            val[10] * other.val[10],
            val[11] * other.val[11],
            val[12] * other.val[12],
            val[13] * other.val[13],
            val[14] * other.val[14],
            val[15] * other.val[15],
            val[16] * other.val[16],
            val[17] * other.val[17],
            val[18] * other.val[18],
            val[19] * other.val[19],
            val[20] * other.val[20],
            val[21] * other.val[21],
            val[22] * other.val[22],
            val[23] * other.val[23],
            val[24] * other.val[24],
            val[25] * other.val[25],
            val[26] * other.val[26],
            val[27] * other.val[27],
            val[28] * other.val[28],
            val[29] * other.val[29],
            val[30] * other.val[30],
            val[31] * other.val[31]);
    }

    inline
    void operator/=(const short_vec<int, 32>& other)
    {
        val[ 0] /= other.val[ 0];
        val[ 1] /= other.val[ 1];
        val[ 2] /= other.val[ 2];
        val[ 3] /= other.val[ 3];
        val[ 4] /= other.val[ 4];
        val[ 5] /= other.val[ 5];
        val[ 6] /= other.val[ 6];
        val[ 7] /= other.val[ 7];
        val[ 8] /= other.val[ 8];
        val[ 9] /= other.val[ 9];
        val[10] /= other.val[10];
        val[11] /= other.val[11];
        val[12] /= other.val[12];
        val[13] /= other.val[13];
        val[14] /= other.val[14];
        val[15] /= other.val[15];
        val[16] /= other.val[16];
        val[17] /= other.val[17];
        val[18] /= other.val[18];
        val[19] /= other.val[19];
        val[20] /= other.val[20];
        val[21] /= other.val[21];
        val[22] /= other.val[22];
        val[23] /= other.val[23];
        val[24] /= other.val[24];
        val[25] /= other.val[25];
        val[26] /= other.val[26];
        val[27] /= other.val[27];
        val[28] /= other.val[28];
        val[29] /= other.val[29];
        val[30] /= other.val[30];
        val[31] /= other.val[31];
    }

    inline
    short_vec<int, 32> operator/(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
            val[ 0] / other.val[ 0],
            val[ 1] / other.val[ 1],
            val[ 2] / other.val[ 2],
            val[ 3] / other.val[ 3],
            val[ 4] / other.val[ 4],
            val[ 5] / other.val[ 5],
            val[ 6] / other.val[ 6],
            val[ 7] / other.val[ 7],
            val[ 8] / other.val[ 8],
            val[ 9] / other.val[ 9],
            val[10] / other.val[10],
            val[11] / other.val[11],
            val[12] / other.val[12],
            val[13] / other.val[13],
            val[14] / other.val[14],
            val[15] / other.val[15],
            val[16] / other.val[16],
            val[17] / other.val[17],
            val[18] / other.val[18],
            val[19] / other.val[19],
            val[20] / other.val[20],
            val[21] / other.val[21],
            val[22] / other.val[22],
            val[23] / other.val[23],
            val[24] / other.val[24],
            val[25] / other.val[25],
            val[26] / other.val[26],
            val[27] / other.val[27],
            val[28] / other.val[28],
            val[29] / other.val[29],
            val[30] / other.val[30],
            val[31] / other.val[31]);
    }

    inline
    short_vec<int, 32> sqrt() const
    {
        return short_vec<int, 32>(
            static_cast<int>(std::sqrt(val[ 0])),
            static_cast<int>(std::sqrt(val[ 1])),
            static_cast<int>(std::sqrt(val[ 2])),
            static_cast<int>(std::sqrt(val[ 3])),
            static_cast<int>(std::sqrt(val[ 4])),
            static_cast<int>(std::sqrt(val[ 5])),
            static_cast<int>(std::sqrt(val[ 6])),
            static_cast<int>(std::sqrt(val[ 7])),
            static_cast<int>(std::sqrt(val[ 8])),
            static_cast<int>(std::sqrt(val[ 9])),
            static_cast<int>(std::sqrt(val[10])),
            static_cast<int>(std::sqrt(val[11])),
            static_cast<int>(std::sqrt(val[12])),
            static_cast<int>(std::sqrt(val[13])),
            static_cast<int>(std::sqrt(val[14])),
            static_cast<int>(std::sqrt(val[15])),
            static_cast<int>(std::sqrt(val[16])),
            static_cast<int>(std::sqrt(val[17])),
            static_cast<int>(std::sqrt(val[18])),
            static_cast<int>(std::sqrt(val[19])),
            static_cast<int>(std::sqrt(val[20])),
            static_cast<int>(std::sqrt(val[21])),
            static_cast<int>(std::sqrt(val[22])),
            static_cast<int>(std::sqrt(val[23])),
            static_cast<int>(std::sqrt(val[24])),
            static_cast<int>(std::sqrt(val[25])),
            static_cast<int>(std::sqrt(val[26])),
            static_cast<int>(std::sqrt(val[27])),
            static_cast<int>(std::sqrt(val[28])),
            static_cast<int>(std::sqrt(val[29])),
            static_cast<int>(std::sqrt(val[30])),
            static_cast<int>(std::sqrt(val[31])));
    }

    inline
    void load(const int *data)
    {
        val[ 0] = data[ 0];
        val[ 1] = data[ 1];
        val[ 2] = data[ 2];
        val[ 3] = data[ 3];
        val[ 4] = data[ 4];
        val[ 5] = data[ 5];
        val[ 6] = data[ 6];
        val[ 7] = data[ 7];
        val[ 8] = data[ 8];
        val[ 9] = data[ 9];
        val[10] = data[10];
        val[11] = data[11];
        val[12] = data[12];
        val[13] = data[13];
        val[14] = data[14];
        val[15] = data[15];
        val[16] = data[16];
        val[17] = data[17];
        val[18] = data[18];
        val[19] = data[19];
        val[20] = data[20];
        val[21] = data[21];
        val[22] = data[22];
        val[23] = data[23];
        val[24] = data[24];
        val[25] = data[25];
        val[26] = data[26];
        val[27] = data[27];
        val[28] = data[28];
        val[29] = data[29];
        val[30] = data[30];
        val[31] = data[31];
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
        *(data +  4) = val[ 4];
        *(data +  5) = val[ 5];
        *(data +  6) = val[ 6];
        *(data +  7) = val[ 7];
        *(data +  8) = val[ 8];
        *(data +  9) = val[ 9];
        *(data + 10) = val[10];
        *(data + 11) = val[11];
        *(data + 12) = val[12];
        *(data + 13) = val[13];
        *(data + 14) = val[14];
        *(data + 15) = val[15];
        *(data + 16) = val[16];
        *(data + 17) = val[17];
        *(data + 18) = val[18];
        *(data + 19) = val[19];
        *(data + 20) = val[20];
        *(data + 21) = val[21];
        *(data + 22) = val[22];
        *(data + 23) = val[23];
        *(data + 24) = val[24];
        *(data + 25) = val[25];
        *(data + 26) = val[26];
        *(data + 27) = val[27];
        *(data + 28) = val[28];
        *(data + 29) = val[29];
        *(data + 30) = val[30];
        *(data + 31) = val[31];
    }

    inline
    void store_aligned(int *data) const
    {
        store(data);
    }

    LIBFLATARRAY_INLINE
    void store_nt(int *data) const
    {
        store(data);
    }

    inline
    void gather(const int *ptr, const int *offsets)
    {
        val[ 0] = ptr[offsets[ 0]];
        val[ 1] = ptr[offsets[ 1]];
        val[ 2] = ptr[offsets[ 2]];
        val[ 3] = ptr[offsets[ 3]];
        val[ 4] = ptr[offsets[ 4]];
        val[ 5] = ptr[offsets[ 5]];
        val[ 6] = ptr[offsets[ 6]];
        val[ 7] = ptr[offsets[ 7]];
        val[ 8] = ptr[offsets[ 8]];
        val[ 9] = ptr[offsets[ 9]];
        val[10] = ptr[offsets[10]];
        val[11] = ptr[offsets[11]];
        val[12] = ptr[offsets[12]];
        val[13] = ptr[offsets[13]];
        val[14] = ptr[offsets[14]];
        val[15] = ptr[offsets[15]];
        val[16] = ptr[offsets[16]];
        val[17] = ptr[offsets[17]];
        val[18] = ptr[offsets[18]];
        val[19] = ptr[offsets[19]];
        val[20] = ptr[offsets[20]];
        val[21] = ptr[offsets[21]];
        val[22] = ptr[offsets[22]];
        val[23] = ptr[offsets[23]];
        val[24] = ptr[offsets[24]];
        val[25] = ptr[offsets[25]];
        val[26] = ptr[offsets[26]];
        val[27] = ptr[offsets[27]];
        val[28] = ptr[offsets[28]];
        val[29] = ptr[offsets[29]];
        val[30] = ptr[offsets[30]];
        val[31] = ptr[offsets[31]];
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = val[ 0];
        ptr[offsets[1]] = val[ 1];
        ptr[offsets[2]] = val[ 2];
        ptr[offsets[3]] = val[ 3];
        ptr[offsets[4]] = val[ 4];
        ptr[offsets[5]] = val[ 5];
        ptr[offsets[6]] = val[ 6];
        ptr[offsets[7]] = val[ 7];
        ptr[offsets[8]] = val[ 8];
        ptr[offsets[9]] = val[ 9];
        ptr[offsets[10]] = val[10];
        ptr[offsets[11]] = val[11];
        ptr[offsets[12]] = val[12];
        ptr[offsets[13]] = val[13];
        ptr[offsets[14]] = val[14];
        ptr[offsets[15]] = val[15];
        ptr[offsets[16]] = val[16];
        ptr[offsets[17]] = val[17];
        ptr[offsets[18]] = val[18];
        ptr[offsets[19]] = val[19];
        ptr[offsets[20]] = val[20];
        ptr[offsets[21]] = val[21];
        ptr[offsets[22]] = val[22];
        ptr[offsets[23]] = val[23];
        ptr[offsets[24]] = val[24];
        ptr[offsets[25]] = val[25];
        ptr[offsets[26]] = val[26];
        ptr[offsets[27]] = val[27];
        ptr[offsets[28]] = val[28];
        ptr[offsets[29]] = val[29];
        ptr[offsets[30]] = val[30];
        ptr[offsets[31]] = val[31];
    }

private:
    int val[32];
};

LIBFLATARRAY_INLINE
void operator<<(int *data, const short_vec<int, 32>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

LIBFLATARRAY_INLINE
short_vec<int, 32> sqrt(const short_vec<int, 32>& vec)
{
    return vec.sqrt();
}

// not inlining is ok:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 )
#endif

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<int, 32>& vec)
{
    __os << "["  << vec.val[ 0] << ", " << vec.val[ 1] << ", " << vec.val[ 2] << ", " << vec.val[ 3]
         << ", " << vec.val[ 4] << ", " << vec.val[ 5] << ", " << vec.val[ 6] << ", " << vec.val[ 7]
         << ", " << vec.val[ 8] << ", " << vec.val[ 9] << ", " << vec.val[10] << ", " << vec.val[11]
         << ", " << vec.val[12] << ", " << vec.val[13] << ", " << vec.val[14] << ", " << vec.val[15]
         << ", " << vec.val[16] << ", " << vec.val[17] << ", " << vec.val[18] << ", " << vec.val[19]
         << ", " << vec.val[20] << ", " << vec.val[21] << ", " << vec.val[22] << ", " << vec.val[23]
         << ", " << vec.val[24] << ", " << vec.val[25] << ", " << vec.val[26] << ", " << vec.val[27]
         << ", " << vec.val[28] << ", " << vec.val[29] << ", " << vec.val[30] << ", " << vec.val[31]
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

#endif
