/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_FLOAT_1_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_FLOAT_1_HPP

#include <cmath>
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
class short_vec<float, 1> : public short_vec_base<float, 1>
{
public:
    static const std::size_t ARITY = 1;
    typedef unsigned char mask_type;
    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 1>& vec);

    inline
    short_vec(const float data = 0) :
        val(data)
    {}

    inline
    short_vec(const float *data)
    {
        load(data);
    }

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
        return val;
    }

    inline
    float operator[](const int /* i */) const
    {
        return val;
    }

    inline
    void operator-=(const short_vec<float, 1>& other)
    {
        val -= other.val;
    }

    inline
    short_vec<float, 1> operator-(const short_vec<float, 1>& other) const
    {
        return short_vec<float, 1>(
            val - other.val);
    }

    inline
    void operator+=(const short_vec<float, 1>& other)
    {
        val += other.val;
    }

    inline
    short_vec<float, 1> operator+(const short_vec<float, 1>& other) const
    {
        return short_vec<float, 1>(
            val + other.val);
    }

    inline
    void operator*=(const short_vec<float, 1>& other)
    {
        val *= other.val;
    }

    inline
    short_vec<float, 1> operator*(const short_vec<float, 1>& other) const
    {
        return short_vec<float, 1>(
            val * other.val);
    }

    inline
    void operator/=(const short_vec<float, 1>& other)
    {
        val /= other.val;
    }

    inline
    short_vec<float, 1> operator/(const short_vec<float, 1>& other) const
    {
        return short_vec<float, 1>(
            val / other.val);
    }

#define LFA_SHORTVEC_COMPARE_HELPER(V1, V2, OP) ((V1) OP (V2))
    inline
    mask_type operator<(const short_vec<float, 1>& other) const
    {
        return LFA_SHORTVEC_COMPARE_HELPER(val, other.val, <);
    }

    inline
    mask_type operator<=(const short_vec<float, 1>& other) const
    {
        return LFA_SHORTVEC_COMPARE_HELPER(val, other.val, <=);
    }

    inline
    mask_type operator==(const short_vec<float, 1>& other) const
    {
        return LFA_SHORTVEC_COMPARE_HELPER(val, other.val, ==);
    }

    inline
    mask_type operator>(const short_vec<float, 1>& other) const
    {
        return LFA_SHORTVEC_COMPARE_HELPER(val, other.val, >);
    }

    inline
    mask_type operator>=(const short_vec<float, 1>& other) const
    {
        return LFA_SHORTVEC_COMPARE_HELPER(val, other.val, >=);
    }
#undef LFA_SHORTVEC_COMPARE_HELPER

    inline
    short_vec<float, 1> sqrt() const
    {
        return short_vec<float, 1>(
            std::sqrt(val));
    }

    inline
    void load(const float *data)
    {
        val = data[0];
    }

    inline
    void load_aligned(const float *data)
    {
        load(data);
    }

    inline
    void store(float *data) const
    {
        *(data + 0) = val;
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
        val = ptr[offsets[0]];
    }

    inline
    void scatter(float *ptr, const int *offsets) const
    {
        ptr[offsets[0]] = val;
    }

    inline
    void blend(const mask_type& mask, const short_vec<float, 1>& other)
    {
        if (mask & 1) {
            val = other.val;
        }
    }

private:
    float val;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(float *data, const short_vec<float, 1>& vec)
{
    vec.store(data);
}

inline
short_vec<float, 1> sqrt(const short_vec<float, 1>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 1>& vec)
{
    __os << "[" << vec.val << "]";
    return __os;
}

}

#endif
