/**
 * Copyright 2014-2016 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_FLOAT_2_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_FLOAT_2_HPP

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
class short_vec<float, 2> : public short_vec_base<float, 2>
{
public:
    static const std::size_t ARITY = 2;
    typedef unsigned char mask_type;
    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 2>& vec);

    inline
    short_vec(const float data = 0) :
        val1(data),
        val2(data)
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(const float val1, const float val2) :
        val1(val1),
        val2(val2)
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
            val1 ||
            val2;
    }

    inline
    float operator[](const int i) const
    {
        switch (i) {
        case 0:
            return val1;
        default:
            return val2;
        }
    }

    inline
    void operator-=(const short_vec<float, 2>& other)
    {
        val1 -= other.val1;
        val2 -= other.val2;
    }

    inline
    short_vec<float, 2> operator-(const short_vec<float, 2>& other) const
    {
        return short_vec<float, 2>(
            val1 - other.val1,
            val2 - other.val2);
    }

    inline
    void operator+=(const short_vec<float, 2>& other)
    {
        val1 += other.val1;
        val2 += other.val2;
    }

    inline
    short_vec<float, 2> operator+(const short_vec<float, 2>& other) const
    {
        return short_vec<float, 2>(
            val1 + other.val1,
            val2 + other.val2);
    }

    inline
    void operator*=(const short_vec<float, 2>& other)
    {
        val1 *= other.val1;
        val2 *= other.val2;
    }

    inline
    short_vec<float, 2> operator*(const short_vec<float, 2>& other) const
    {
        return short_vec<float, 2>(
            val1 * other.val1,
            val2 * other.val2);
    }

    inline
    void operator/=(const short_vec<float, 2>& other)
    {
        val1 /= other.val1;
        val2 /= other.val2;
    }

    inline
    short_vec<float, 2> operator/(const short_vec<float, 2>& other) const
    {
        return short_vec<float, 2>(
            val1 / other.val1,
            val2 / other.val2);
    }

#define LFA_SHORTVEC_COMPARE_HELPER(V1, V2, OP) ((V1) OP (V2))
    inline
    mask_type operator<(const short_vec<float, 2>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, <) << 0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, <) << 1);
    }

    inline
    mask_type operator<=(const short_vec<float, 2>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, <=) << 0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, <=) << 1);
    }

    inline
    mask_type operator==(const short_vec<float, 2>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, ==) << 0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, ==) << 1);
    }

    inline
    mask_type operator>(const short_vec<float, 2>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, >) << 0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, >) << 1);
    }

    inline
    mask_type operator>=(const short_vec<float, 2>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, >=) << 0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, >=) << 1);
    }
#undef LFA_SHORTVEC_COMPARE_HELPER

    inline
    short_vec<float, 2> sqrt() const
    {
        return short_vec<float, 2>(
            std::sqrt(val1),
            std::sqrt(val2));
    }

    inline
    void load(const float *data)
    {
        val1 = data[0];
        val2 = data[1];
    }

    inline
    void load_aligned(const float *data)
    {
        load(data);
    }

    inline
    void store(float *data) const
    {
        *(data +  0) = val1;
        *(data +  1) = val2;
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
        val1 = ptr[offsets[0]];
        val2 = ptr[offsets[1]];
    }

    inline
    void scatter(float *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = val1;
        ptr[offsets[1]] = val2;
    }

    inline
    void blend(const mask_type& mask, const short_vec<float, 2>& other)
    {
        if (mask & (1 << 0)) {
            val1 = other.val1;
        }
        if (mask & (1 << 1)) {
            val2 = other.val2;
        }
    }

private:
    float val1;
    float val2;
};

inline
void operator<<(float *data, const short_vec<float, 2>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<float, 2> sqrt(const short_vec<float, 2>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 2>& vec)
{
    __os << "["  << vec.val1 << ", " << vec.val2
         << "]";
    return __os;
}

}

#endif
