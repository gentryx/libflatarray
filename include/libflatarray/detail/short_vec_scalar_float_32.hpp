/**
 * Copyright 2014-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_FLOAT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_FLOAT_32_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SCALAR)

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
class short_vec<float, 32> : public short_vec_base<float, 32>
{
public:
    static const std::size_t ARITY = 32;
    typedef unsigned mask_type;
    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 32>& vec);

    inline
    short_vec(const float data = 0) :
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
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(
        const float val1,
        const float val2,
        const float val3,
        const float val4,
        const float val5,
        const float val6,
        const float val7,
        const float val8,
        const float val9,
        const float val10,
        const float val11,
        const float val12,
        const float val13,
        const float val14,
        const float val15,
        const float val16,
        const float val17,
        const float val18,
        const float val19,
        const float val20,
        const float val21,
        const float val22,
        const float val23,
        const float val24,
        const float val25,
        const float val26,
        const float val27,
        const float val28,
        const float val29,
        const float val30,
        const float val31,
        const float val32) :
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
    short_vec(const std::initializer_list<float>& il)
    {
        const float *ptr = static_cast<const float *>(&(*il.begin()));
        load(ptr);
    }
#endif

    inline
    bool any() const
    {
        return
            val[ 0] ||
            val[ 1] ||
            val[ 2] ||
            val[ 3] ||
            val[ 4] ||
            val[ 5] ||
            val[ 6] ||
            val[ 7] ||
            val[ 8] ||
            val[ 9] ||
            val[10] ||
            val[11] ||
            val[12] ||
            val[13] ||
            val[14] ||
            val[15] ||
            val[16] ||
            val[17] ||
            val[18] ||
            val[19] ||
            val[20] ||
            val[21] ||
            val[22] ||
            val[23] ||
            val[24] ||
            val[25] ||
            val[26] ||
            val[27] ||
            val[28] ||
            val[29] ||
            val[30] ||
            val[31];
    }

    inline
    float operator[](const int i) const
    {
        return val[i];
    }

    inline
    void operator-=(const short_vec<float, 32>& other)
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
    short_vec<float, 32> operator-(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
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
    void operator+=(const short_vec<float, 32>& other)
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
    short_vec<float, 32> operator+(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
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
    void operator*=(const short_vec<float, 32>& other)
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
    short_vec<float, 32> operator*(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
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
    void operator/=(const short_vec<float, 32>& other)
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
    short_vec<float, 32> operator/(const short_vec<float, 32>& other) const
    {
        return short_vec<float, 32>(
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

#define LFA_SHORTVEC_COMPARE_HELPER(V1, V2, OP) ((V1) OP (V2))
    inline
    mask_type operator<(const short_vec<float, 32>& other) const
    {
        return
            mask_type((LFA_SHORTVEC_COMPARE_HELPER(val[ 0], other.val[ 0], <) <<  0) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 1], other.val[ 1], <) <<  1) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 2], other.val[ 2], <) <<  2) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 3], other.val[ 3], <) <<  3) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 4], other.val[ 4], <) <<  4) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 5], other.val[ 5], <) <<  5) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 6], other.val[ 6], <) <<  6) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 7], other.val[ 7], <) <<  7) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 8], other.val[ 8], <) <<  8) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 9], other.val[ 9], <) <<  9) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[10], other.val[10], <) << 10) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[11], other.val[11], <) << 11) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[12], other.val[12], <) << 12) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[13], other.val[13], <) << 13) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[14], other.val[14], <) << 14) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[15], other.val[15], <) << 15) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[16], other.val[16], <) << 16) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[17], other.val[17], <) << 17) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[18], other.val[18], <) << 18) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[19], other.val[19], <) << 19) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[20], other.val[20], <) << 20) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[21], other.val[21], <) << 21) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[22], other.val[22], <) << 22) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[23], other.val[23], <) << 23) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[24], other.val[24], <) << 24) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[25], other.val[25], <) << 25) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[26], other.val[26], <) << 26) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[27], other.val[27], <) << 27) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[28], other.val[28], <) << 28) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[29], other.val[29], <) << 29) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[30], other.val[30], <) << 30) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[31], other.val[31], <) << 31));
    }

    inline
    mask_type operator<=(const short_vec<float, 32>& other) const
    {
        return
            mask_type((LFA_SHORTVEC_COMPARE_HELPER(val[ 0], other.val[ 0], <=) <<  0) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 1], other.val[ 1], <=) <<  1) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 2], other.val[ 2], <=) <<  2) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 3], other.val[ 3], <=) <<  3) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 4], other.val[ 4], <=) <<  4) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 5], other.val[ 5], <=) <<  5) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 6], other.val[ 6], <=) <<  6) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 7], other.val[ 7], <=) <<  7) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 8], other.val[ 8], <=) <<  8) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 9], other.val[ 9], <=) <<  9) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[10], other.val[10], <=) << 10) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[11], other.val[11], <=) << 11) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[12], other.val[12], <=) << 12) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[13], other.val[13], <=) << 13) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[14], other.val[14], <=) << 14) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[15], other.val[15], <=) << 15) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[16], other.val[16], <=) << 16) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[17], other.val[17], <=) << 17) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[18], other.val[18], <=) << 18) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[19], other.val[19], <=) << 19) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[20], other.val[20], <=) << 20) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[21], other.val[21], <=) << 21) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[22], other.val[22], <=) << 22) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[23], other.val[23], <=) << 23) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[24], other.val[24], <=) << 24) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[25], other.val[25], <=) << 25) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[26], other.val[26], <=) << 26) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[27], other.val[27], <=) << 27) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[28], other.val[28], <=) << 28) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[29], other.val[29], <=) << 29) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[30], other.val[30], <=) << 30) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[31], other.val[31], <=) << 31));
    }

    inline
    mask_type operator==(const short_vec<float, 32>& other) const
    {
        return
            mask_type((LFA_SHORTVEC_COMPARE_HELPER(val[ 0], other.val[ 0], ==) <<  0) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 1], other.val[ 1], ==) <<  1) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 2], other.val[ 2], ==) <<  2) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 3], other.val[ 3], ==) <<  3) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 4], other.val[ 4], ==) <<  4) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 5], other.val[ 5], ==) <<  5) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 6], other.val[ 6], ==) <<  6) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 7], other.val[ 7], ==) <<  7) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 8], other.val[ 8], ==) <<  8) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 9], other.val[ 9], ==) <<  9) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[10], other.val[10], ==) << 10) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[11], other.val[11], ==) << 11) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[12], other.val[12], ==) << 12) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[13], other.val[13], ==) << 13) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[14], other.val[14], ==) << 14) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[15], other.val[15], ==) << 15) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[16], other.val[16], ==) << 16) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[17], other.val[17], ==) << 17) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[18], other.val[18], ==) << 18) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[19], other.val[19], ==) << 19) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[20], other.val[20], ==) << 20) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[21], other.val[21], ==) << 21) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[22], other.val[22], ==) << 22) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[23], other.val[23], ==) << 23) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[24], other.val[24], ==) << 24) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[25], other.val[25], ==) << 25) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[26], other.val[26], ==) << 26) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[27], other.val[27], ==) << 27) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[28], other.val[28], ==) << 28) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[29], other.val[29], ==) << 29) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[30], other.val[30], ==) << 30) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[31], other.val[31], ==) << 31));
    }

    inline
    mask_type operator>(const short_vec<float, 32>& other) const
    {
        return
            mask_type((LFA_SHORTVEC_COMPARE_HELPER(val[ 0], other.val[ 0], >) <<  0) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 1], other.val[ 1], >) <<  1) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 2], other.val[ 2], >) <<  2) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 3], other.val[ 3], >) <<  3) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 4], other.val[ 4], >) <<  4) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 5], other.val[ 5], >) <<  5) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 6], other.val[ 6], >) <<  6) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 7], other.val[ 7], >) <<  7) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 8], other.val[ 8], >) <<  8) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 9], other.val[ 9], >) <<  9) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[10], other.val[10], >) << 10) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[11], other.val[11], >) << 11) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[12], other.val[12], >) << 12) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[13], other.val[13], >) << 13) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[14], other.val[14], >) << 14) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[15], other.val[15], >) << 15) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[16], other.val[16], >) << 16) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[17], other.val[17], >) << 17) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[18], other.val[18], >) << 18) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[19], other.val[19], >) << 19) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[20], other.val[20], >) << 20) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[21], other.val[21], >) << 21) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[22], other.val[22], >) << 22) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[23], other.val[23], >) << 23) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[24], other.val[24], >) << 24) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[25], other.val[25], >) << 25) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[26], other.val[26], >) << 26) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[27], other.val[27], >) << 27) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[28], other.val[28], >) << 28) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[29], other.val[29], >) << 29) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[30], other.val[30], >) << 30) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[31], other.val[31], >) << 31));
    }

    inline
    mask_type operator>=(const short_vec<float, 32>& other) const
    {
        return
            mask_type((LFA_SHORTVEC_COMPARE_HELPER(val[ 0], other.val[ 0], >=) <<  0) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 1], other.val[ 1], >=) <<  1) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 2], other.val[ 2], >=) <<  2) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 3], other.val[ 3], >=) <<  3) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 4], other.val[ 4], >=) <<  4) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 5], other.val[ 5], >=) <<  5) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 6], other.val[ 6], >=) <<  6) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 7], other.val[ 7], >=) <<  7) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 8], other.val[ 8], >=) <<  8) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[ 9], other.val[ 9], >=) <<  9) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[10], other.val[10], >=) << 10) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[11], other.val[11], >=) << 11) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[12], other.val[12], >=) << 12) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[13], other.val[13], >=) << 13) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[14], other.val[14], >=) << 14) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[15], other.val[15], >=) << 15) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[16], other.val[16], >=) << 16) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[17], other.val[17], >=) << 17) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[18], other.val[18], >=) << 18) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[19], other.val[19], >=) << 19) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[20], other.val[20], >=) << 20) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[21], other.val[21], >=) << 21) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[22], other.val[22], >=) << 22) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[23], other.val[23], >=) << 23) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[24], other.val[24], >=) << 24) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[25], other.val[25], >=) << 25) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[26], other.val[26], >=) << 26) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[27], other.val[27], >=) << 27) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[28], other.val[28], >=) << 28) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[29], other.val[29], >=) << 29) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[30], other.val[30], >=) << 30) +
                      (LFA_SHORTVEC_COMPARE_HELPER(val[31], other.val[31], >=) << 31));
    }
#undef LFA_SHORTVEC_COMPARE_HELPER

    inline
    short_vec<float, 32> sqrt() const
    {
        return short_vec<float, 32>(
            std::sqrt(val[ 0]),
            std::sqrt(val[ 1]),
            std::sqrt(val[ 2]),
            std::sqrt(val[ 3]),
            std::sqrt(val[ 4]),
            std::sqrt(val[ 5]),
            std::sqrt(val[ 6]),
            std::sqrt(val[ 7]),
            std::sqrt(val[ 8]),
            std::sqrt(val[ 9]),
            std::sqrt(val[10]),
            std::sqrt(val[11]),
            std::sqrt(val[12]),
            std::sqrt(val[13]),
            std::sqrt(val[14]),
            std::sqrt(val[15]),
            std::sqrt(val[16]),
            std::sqrt(val[17]),
            std::sqrt(val[18]),
            std::sqrt(val[19]),
            std::sqrt(val[20]),
            std::sqrt(val[21]),
            std::sqrt(val[22]),
            std::sqrt(val[23]),
            std::sqrt(val[24]),
            std::sqrt(val[25]),
            std::sqrt(val[26]),
            std::sqrt(val[27]),
            std::sqrt(val[28]),
            std::sqrt(val[29]),
            std::sqrt(val[30]),
            std::sqrt(val[31]));
    }

    inline
    void load(const float *data)
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
    void load_aligned(const float *data)
    {
        load(data);
    }

    inline
    void store(float *data) const
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
    void store_aligned(float *data) const
    {
        store(data);
    }

    inline
    void store_nt(float *data) const
    {
        store(data);
    }

    inline
    void gather(const float *ptr, const int *offsets)
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
    void scatter(float *ptr, const int *offsets) const
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

    inline
    void blend(const mask_type& mask, const short_vec<float, 32>& other)
    {
        if (mask & (1 << 0)) {
            val[ 0] = other.val[ 0];
        }
        if (mask & (1 << 1)) {
            val[ 1] = other.val[ 1];
        }
        if (mask & (1 << 2)) {
            val[ 2] = other.val[ 2];
        }
        if (mask & (1 << 3)) {
            val[ 3] = other.val[ 3];
        }
        if (mask & (1 << 4)) {
            val[ 4] = other.val[ 4];
        }
        if (mask & (1 << 5)) {
            val[ 5] = other.val[ 5];
        }
        if (mask & (1 << 6)) {
            val[ 6] = other.val[ 6];
        }
        if (mask & (1 << 7)) {
            val[ 7] = other.val[ 7];
        }
        if (mask & (1 << 8)) {
            val[ 8] = other.val[ 8];
        }
        if (mask & (1 << 9)) {
            val[ 9] = other.val[ 9];
        }
        if (mask & (1 << 10)) {
            val[10] = other.val[10];
        }
        if (mask & (1 << 11)) {
            val[11] = other.val[11];
        }
        if (mask & (1 << 12)) {
            val[12] = other.val[12];
        }
        if (mask & (1 << 13)) {
            val[13] = other.val[13];
        }
        if (mask & (1 << 14)) {
            val[14] = other.val[14];
        }
        if (mask & (1 << 15)) {
            val[15] = other.val[15];
        }
        if (mask & (1 << 16)) {
            val[16] = other.val[16];
        }
        if (mask & (1 << 17)) {
            val[17] = other.val[17];
        }
        if (mask & (1 << 18)) {
            val[18] = other.val[18];
        }
        if (mask & (1 << 19)) {
            val[19] = other.val[19];
        }
        if (mask & (1 << 20)) {
            val[20] = other.val[20];
        }
        if (mask & (1 << 21)) {
            val[21] = other.val[21];
        }
        if (mask & (1 << 22)) {
            val[22] = other.val[22];
        }
        if (mask & (1 << 23)) {
            val[23] = other.val[23];
        }
        if (mask & (1 << 24)) {
            val[24] = other.val[24];
        }
        if (mask & (1 << 25)) {
            val[25] = other.val[25];
        }
        if (mask & (1 << 26)) {
            val[26] = other.val[26];
        }
        if (mask & (1 << 27)) {
            val[27] = other.val[27];
        }
        if (mask & (1 << 28)) {
            val[28] = other.val[28];
        }
        if (mask & (1 << 29)) {
            val[29] = other.val[29];
        }
        if (mask & (1 << 30)) {
            val[30] = other.val[30];
        }
        if (mask & (0x80000000)) {
            val[31] = other.val[31];
        }
    }

private:
    float val[32];
};

inline
void operator<<(float *data, const short_vec<float, 32>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<float, 32> sqrt(const short_vec<float, 32>& vec)
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
           const short_vec<float, 32>& vec)
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
