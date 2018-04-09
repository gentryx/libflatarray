/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_DOUBLE_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_DOUBLE_16_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SCALAR) ||          \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_ARM_NEON)

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
class short_vec<double, 16> : public short_vec_base<double, 16>
{
public:
    static const std::size_t ARITY = 16;
    typedef unsigned short mask_type;
    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 16>& vec);

    inline
    short_vec(const double data = 0) :
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
            data}
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(
        const double val1,
        const double val2,
        const double val3,
        const double val4,
        const double val5,
        const double val6,
        const double val7,
        const double val8,
        const double val9,
        const double val10,
        const double val11,
        const double val12,
        const double val13,
        const double val14,
        const double val15,
        const double val16) :
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
            val16}
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
            val[15];
    }

    inline
    double operator[](const int i) const
    {
        return val[i];
    }

    inline
    void operator-=(const short_vec<double, 16>& other)
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
    }

    inline
    short_vec<double, 16> operator-(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
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
            val[15] - other.val[15]);
    }

    inline
    void operator+=(const short_vec<double, 16>& other)
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
    }

    inline
    short_vec<double, 16> operator+(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
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
            val[15] + other.val[15]);
    }

    inline
    void operator*=(const short_vec<double, 16>& other)
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
    }

    inline
    short_vec<double, 16> operator*(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
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
            val[15] * other.val[15]);
    }

    inline
    void operator/=(const short_vec<double, 16>& other)
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
    }

    inline
    short_vec<double, 16> operator/(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
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
            val[15] / other.val[15]);
    }

#define LFA_SHORTVEC_COMPARE_HELPER(V1, V2, OP) ((V1) OP (V2))
    inline
    mask_type operator<(const short_vec<double, 16>& other) const
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
                      (LFA_SHORTVEC_COMPARE_HELPER(val[15], other.val[15], <) << 15));
    }

    inline
    mask_type operator<=(const short_vec<double, 16>& other) const
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
                      (LFA_SHORTVEC_COMPARE_HELPER(val[15], other.val[15], <=) << 15));
    }

    inline
    mask_type operator==(const short_vec<double, 16>& other) const
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
                      (LFA_SHORTVEC_COMPARE_HELPER(val[15], other.val[15], ==) << 15));
    }

    inline
    mask_type operator>(const short_vec<double, 16>& other) const
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
                      (LFA_SHORTVEC_COMPARE_HELPER(val[15], other.val[15], >) << 15));
    }

    inline
    mask_type operator>=(const short_vec<double, 16>& other) const
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
                      (LFA_SHORTVEC_COMPARE_HELPER(val[15], other.val[15], >=) << 15));
    }
#undef LFA_SHORTVEC_COMPARE_HELPER

    inline
    short_vec<double, 16> sqrt() const
    {
        return short_vec<double, 16>(
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
            std::sqrt(val[15]));
    }

    inline
    void load(const double *data)
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
    }

    inline
    void load_aligned(const double *data)
    {
        load(data);
    }

    inline
    void store(double *data) const
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
    }

    inline
    void scatter(double *ptr, const int *offsets) const
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
    }

    inline
    void blend(const mask_type& mask, const short_vec<double, 16>& other)
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
    }

private:
    double val[16];
};

inline
void operator<<(double *data, const short_vec<double, 16>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<double, 16> sqrt(const short_vec<double, 16>& vec)
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
           const short_vec<double, 16>& vec)
{
    __os << "["  << vec.val[ 0] << ", " << vec.val[ 1] << ", " << vec.val[ 2] << ", " << vec.val[ 3]
         << ", " << vec.val[ 4] << ", " << vec.val[ 5] << ", " << vec.val[ 6] << ", " << vec.val[ 7]
         << ", " << vec.val[ 8] << ", " << vec.val[ 9] << ", " << vec.val[10] << ", " << vec.val[11]
         << ", " << vec.val[12] << ", " << vec.val[13] << ", " << vec.val[14] << ", " << vec.val[15]
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
