/**
 * Copyright 2014-2016 Andreas Schäfer
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
class short_vec<double, 4>
{
public:
    static const int ARITY = 4;

    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 4>& vec);

    inline
    short_vec(const double data = 0) :
        val1(data),
        val2(data),
        val3(data),
        val4(data)
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
        val1( val1),
        val2( val2),
        val3( val3),
        val4( val4)
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
    operator bool() const
    {
        return
            val1 ||
            val2 ||
            val3 ||
            val4;
    }

    inline
    void operator-=(const short_vec<double, 4>& other)
    {
        val1  -= other.val1;
        val2  -= other.val2;
        val3  -= other.val3;
        val4  -= other.val4;
    }

    inline
    short_vec<double, 4> operator-(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            val1  - other.val1,
            val2  - other.val2,
            val3  - other.val3,
            val4  - other.val4);
    }

    inline
    void operator+=(const short_vec<double, 4>& other)
    {
        val1  += other.val1;
        val2  += other.val2;
        val3  += other.val3;
        val4  += other.val4;
    }

    inline
    short_vec<double, 4> operator+(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            val1  + other.val1,
            val2  + other.val2,
            val3  + other.val3,
            val4  + other.val4);
    }

    inline
    void operator*=(const short_vec<double, 4>& other)
    {
        val1  *= other.val1;
        val2  *= other.val2;
        val3  *= other.val3;
        val4  *= other.val4;
    }

    inline
    short_vec<double, 4> operator*(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            val1  * other.val1,
            val2  * other.val2,
            val3  * other.val3,
            val4  * other.val4);
    }

    inline
    void operator/=(const short_vec<double, 4>& other)
    {
        val1  /= other.val1;
        val2  /= other.val2;
        val3  /= other.val3;
        val4  /= other.val4;
    }

    inline
    short_vec<double, 4> operator/(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            val1  / other.val1,
            val2  / other.val2,
            val3  / other.val3,
            val4  / other.val4);
    }

#define LFA_SHORTVEC_COMPARE_HELPER(V1, V2, OP) (((V1) OP (V2)) ? 0xFFFFFFFFFFFFFFFF : 0)
    inline
    short_vec<double, 4> operator<(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, <),
            LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, <),
            LFA_SHORTVEC_COMPARE_HELPER(val3, other.val3, <),
            LFA_SHORTVEC_COMPARE_HELPER(val4, other.val4, <));
    }

    inline
    short_vec<double, 4> operator<=(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val3, other.val3, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val4, other.val4, <=));
    }

    inline
    short_vec<double, 4> operator==(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val3, other.val3, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val4, other.val4, ==));
    }

    inline
    short_vec<double, 4> operator>(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, >),
            LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, >),
            LFA_SHORTVEC_COMPARE_HELPER(val3, other.val3, >),
            LFA_SHORTVEC_COMPARE_HELPER(val4, other.val4, >));
    }

    inline
    short_vec<double, 4> operator>=(const short_vec<double, 4>& other) const
    {
        return short_vec<double, 4>(
            LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val3, other.val3, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val4, other.val4, >=));
    }
#undef LFA_SHORTVEC_COMPARE_HELPER

    inline
    short_vec<double, 4> sqrt() const
    {
        return short_vec<double, 4>(
            std::sqrt(val1),
            std::sqrt(val2),
            std::sqrt(val3),
            std::sqrt(val4));
    }

    inline
    void load(const double *data)
    {
        val1 = data[0];
        val2 = data[1];
        val3 = data[2];
        val4 = data[3];
    }

    inline
    void load_aligned(const double *data)
    {
        load(data);
    }

    inline
    void store(double *data) const
    {
        *(data +  0) = val1;
        *(data +  1) = val2;
        *(data +  2) = val3;
        *(data +  3) = val4;
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
        val2 = ptr[offsets[1]];
        val3 = ptr[offsets[2]];
        val4 = ptr[offsets[3]];
    }

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = val1;
        ptr[offsets[1]] = val2;
        ptr[offsets[2]] = val3;
        ptr[offsets[3]] = val4;
    }

private:
    double val1;
    double val2;
    double val3;
    double val4;
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
    __os << "["  << vec.val1  << ", " << vec.val2  << ", " << vec.val3  << ", " << vec.val4
         << "]";
    return __os;
}

}

#endif

#endif
