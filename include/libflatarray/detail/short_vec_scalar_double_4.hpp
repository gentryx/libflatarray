/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_DOUBLE_4_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_DOUBLE_4_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SCALAR) ||          \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_ARM_NEON) ||        \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_MIC)

#include <libflatarray/config.h>
#include <libflatarray/short_vec_base.hpp>

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
class short_vec<double, 4> : public short_vec_base<double, 4>
{
public:
    static const std::size_t ARITY = 4;
    typedef unsigned char mask_type;
    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 4>& vec);

    inline
    short_vec(const double data = 0) :
        val{data,
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
        const double val4) :
        val{val1,
            val2,
            val3,
            val4}
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
            val[ 3];
    }

    inline
    double operator[](const int i) const
    {
        return val[i];
    }

    inline
    void operator-=(const short_vec<double, 4>& other)
    {
        val[ 0] -= other.val[ 0];
        val[ 1] -= other.val[ 1];
        val[ 2] -= other.val[ 2];
        val[ 3] -= other.val[ 3];
    }

    inline
    short_vec<double, 4> operator-(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            val[ 0] - other.val[ 0],
            val[ 1] - other.val[ 1],
            val[ 2] - other.val[ 2],
            val[ 3] - other.val[ 3]);
    }

    inline
    void operator+=(const short_vec<double, 4>& other)
    {
        val[ 0] += other.val[ 0];
        val[ 1] += other.val[ 1];
        val[ 2] += other.val[ 2];
        val[ 3] += other.val[ 3];
    }

    inline
    short_vec<double, 4> operator+(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            val[ 0] + other.val[ 0],
            val[ 1] + other.val[ 1],
            val[ 2] + other.val[ 2],
            val[ 3] + other.val[ 3]);
    }

    inline
    void operator*=(const short_vec<double, 4>& other)
    {
        val[ 0] *= other.val[ 0];
        val[ 1] *= other.val[ 1];
        val[ 2] *= other.val[ 2];
        val[ 3] *= other.val[ 3];
    }

    inline
    short_vec<double, 4> operator*(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            val[ 0] * other.val[ 0],
            val[ 1] * other.val[ 1],
            val[ 2] * other.val[ 2],
            val[ 3] * other.val[ 3]);
    }

    inline
    void operator/=(const short_vec<double, 4>& other)
    {
        val[ 0] /= other.val[ 0];
        val[ 1] /= other.val[ 1];
        val[ 2] /= other.val[ 2];
        val[ 3] /= other.val[ 3];
    }

    inline
    short_vec<double, 4> operator/(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            val[ 0] / other.val[ 0],
            val[ 1] / other.val[ 1],
            val[ 2] / other.val[ 2],
            val[ 3] / other.val[ 3]);
    }

#define LFA_SHORTVEC_COMPARE_HELPER(V1, V2, OP) ((V1) OP (V2))
    inline
    mask_type operator<(const short_vec<double, 4>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 0], other.val[ 0], <) <<  0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 1], other.val[ 1], <) <<  1) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 2], other.val[ 2], <) <<  2) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 3], other.val[ 3], <) <<  3);
    }

    inline
    mask_type operator<=(const short_vec<double, 4>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 0], other.val[ 0], <=) <<  0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 1], other.val[ 1], <=) <<  1) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 2], other.val[ 2], <=) <<  2) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 3], other.val[ 3], <=) <<  3);
    }

    inline
    mask_type operator==(const short_vec<double, 4>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 0], other.val[ 0], ==) <<  0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 1], other.val[ 1], ==) <<  1) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 2], other.val[ 2], ==) <<  2) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 3], other.val[ 3], ==) <<  3);
    }

    inline
    mask_type operator>(const short_vec<double, 4>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 0], other.val[ 0], >) <<  0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 1], other.val[ 1], >) <<  1) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 2], other.val[ 2], >) <<  2) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 3], other.val[ 3], >) <<  3);
    }

    inline
    mask_type operator>=(const short_vec<double, 4>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 0], other.val[ 0], >=) <<  0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 1], other.val[ 1], >=) <<  1) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 2], other.val[ 2], >=) <<  2) +
            (LFA_SHORTVEC_COMPARE_HELPER(val[ 3], other.val[ 3], >=) <<  3);
    }
#undef LFA_SHORTVEC_COMPARE_HELPER

    inline
    short_vec<double, 4> sqrt() const
    {
        return short_vec<double, 4>(
            std::sqrt(val[ 0]),
            std::sqrt(val[ 1]),
            std::sqrt(val[ 2]),
            std::sqrt(val[ 3]));
    }

    inline
    void load(const double *data)
    {
        val[ 0] = data[0];
        val[ 1] = data[1];
        val[ 2] = data[2];
        val[ 3] = data[3];
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
        val[ 0] = ptr[offsets[0]];
        val[ 1] = ptr[offsets[1]];
        val[ 2] = ptr[offsets[2]];
        val[ 3] = ptr[offsets[3]];
    }

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = val[ 0];
        ptr[offsets[1]] = val[ 1];
        ptr[offsets[2]] = val[ 2];
        ptr[offsets[3]] = val[ 3];
    }

    inline
    void blend(const mask_type& mask, const short_vec<double, 4>& other)
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
    }

private:
    double val[4];
};

inline
void operator<<(double *data, const short_vec<double, 4>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<double, 4> sqrt(const short_vec<double, 4>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 4>& vec)
{
    __os << "["
         << vec.val[ 0] << ", "
         << vec.val[ 1] << ", "
         << vec.val[ 2] << ", "
         << vec.val[ 3] << "]";
    return __os;
}

}

#endif

#endif
