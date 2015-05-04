/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_FLOAT_2_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_FLOAT_2_HPP

#include <libflatarray/detail/short_vec_helpers.hpp>

#ifdef SHORTVEC_HAS_CPP11
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
class short_vec<float, 2>
{
public:
    static const int ARITY = 2;

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

#ifdef SHORTVEC_HAS_CPP11
    inline
    short_vec(const std::initializer_list<float>& il)
    {
        static const unsigned indices[] = { 0, 1 };
        const float    *ptr = reinterpret_cast<const float *>(&(*il.begin()));
        const unsigned *ind = static_cast<const unsigned *>(indices);
        gather(ptr, ind);
    }
#endif

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
    void gather(const float *ptr, const unsigned *offsets)
    {
        val1 = ptr[offsets[0]];
        val2 = ptr[offsets[1]];
    }

    inline
    void scatter(float *ptr, const unsigned *offsets) const
    {
        ptr[offsets[0]] = val1;
        ptr[offsets[1]] = val2;
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
