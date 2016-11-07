/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_1_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_1_HPP

#include <cmath>
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
class short_vec<int, 1>
{
public:
    static const std::size_t ARITY = 1;
    typedef unsigned char mask_type;
    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 1>& vec);

    inline
    short_vec(const int data = 0) :
        val1(data)
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

#ifdef LIBFLATARRAY_WITH_CPP14
    inline
    short_vec(const std::initializer_list<int>& il)
    {
        const int *ptr = static_cast<const int *>(&(*il.begin()));
        load(ptr);
    }
#endif

    inline
    bool any() const
    {
        return val1;
    }

    inline
    float get(const int i) const
    {
        return val1;
    }

    inline
    void operator-=(const short_vec<int, 1>& other)
    {
        val1 -= other.val1;
    }

    inline
    short_vec<int, 1> operator-(const short_vec<int, 1>& other) const
    {
        return short_vec<int, 1>(
            val1 - other.val1);
    }

    inline
    void operator+=(const short_vec<int, 1>& other)
    {
        val1 += other.val1;
    }

    inline
    short_vec<int, 1> operator+(const short_vec<int, 1>& other) const
    {
        return short_vec<int, 1>(
            val1 + other.val1);
    }

    inline
    void operator*=(const short_vec<int, 1>& other)
    {
        val1 *= other.val1;
    }

    inline
    short_vec<int, 1> operator*(const short_vec<int, 1>& other) const
    {
        return short_vec<int, 1>(
            val1 * other.val1);
    }

    inline
    void operator/=(const short_vec<int, 1>& other)
    {
        val1 /= other.val1;
    }

    inline
    short_vec<int, 1> operator/(const short_vec<int, 1>& other) const
    {
        return short_vec<int, 1>(
            val1 / other.val1);
    }

#define LFA_SHORTVEC_COMPARE_HELPER(V1, V2, OP) ((V1) OP (V2))
    inline
    mask_type operator<(const short_vec<int, 1>& other) const
    {
        return LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, <);
    }

    inline
    mask_type operator<=(const short_vec<int, 1>& other) const
    {
        return LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, <=);
    }

    inline
    mask_type operator==(const short_vec<int, 1>& other) const
    {
        return LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, ==);
    }

    // fixme: this should be a free function?
    inline
    mask_type operator==(int other) const
    {
        return LFA_SHORTVEC_COMPARE_HELPER(val1, other, ==);
    }

    inline
    mask_type operator>(const short_vec<int, 1>& other) const
    {
        return LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, >);
    }

    inline
    mask_type operator>=(const short_vec<int, 1>& other) const
    {
        return LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, >=);
    }
#undef LFA_SHORTVEC_COMPARE_HELPER

    inline
    short_vec<int, 1> sqrt() const
    {
        return short_vec<int, 1>(
            static_cast<int>(std::sqrt(val1)));
    }

    inline
    void load(const int *data)
    {
        val1 = data[0];
    }

    inline
    void load_aligned(const int *data)
    {
        load(data);
    }

    inline
    void store(int *data) const
    {
        *(data + 0) = val1;
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
    }

    inline
    void scatter(int *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = val1;
    }

private:
    int val1;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(int *data, const short_vec<int, 1>& vec)
{
    vec.store(data);
}

inline
short_vec<int, 1> sqrt(const short_vec<int, 1>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<int, 1>& vec)
{
    __os << "[" << vec.val1 << "]";
    return __os;
}

}

#endif
