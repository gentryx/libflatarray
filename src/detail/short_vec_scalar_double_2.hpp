/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_DOUBLE_2_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_DOUBLE_2_HPP

#ifndef __SSE__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<double, 2>
{
public:
    static const int ARITY = 2;

    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 2>& vec);

    inline
    short_vec(const double data = 0) :
        val1(data),
        val2(data)
    {}

    inline
    short_vec(const double *data) :
        val1(*(data +  0)),
        val2(*(data +  1))
    {}

    inline
    short_vec(const double val1, const double val2) :
        val1(val1),
        val2(val2)
    {}

    inline
    void operator-=(const short_vec<double, 2>& other)
    {
        val1 -= other.val1;
        val2 -= other.val2;
    }

    inline
    short_vec<double, 2> operator-(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            val1 - other.val1,
            val2 - other.val2);
    }

    inline
    void operator+=(const short_vec<double, 2>& other)
    {
        val1 += other.val1;
        val2 += other.val2;
    }

    inline
    short_vec<double, 2> operator+(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            val1 + other.val1,
            val2 + other.val2);
    }

    inline
    void operator*=(const short_vec<double, 2>& other)
    {
        val1 *= other.val1;
        val2 *= other.val2;
    }

    inline
    short_vec<double, 2> operator*(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            val1 * other.val1,
            val2 * other.val2);
    }

    inline
    void operator/=(const short_vec<double, 2>& other)
    {
        val1 /= other.val1;
        val2 /= other.val2;
    }

    inline
    short_vec<double, 2> operator/(const short_vec<double, 2>& other) const
    {
        return short_vec<double, 2>(
            val1 / other.val1,
            val2 / other.val2);
    }

    inline
    short_vec<double, 2> sqrt() const
    {
        return short_vec<double, 2>(
            std::sqrt(val1),
            std::sqrt(val2));
    }

    inline
    void store(double *data) const
    {
        *(data +  0) = val1;
        *(data +  1) = val2;
    }

private:
    double val1;
    double val2;
};

inline
void operator<<(double *data, const short_vec<double, 2>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

short_vec<double, 2> sqrt(const short_vec<double, 2>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 2>& vec)
{
    __os << "["  << vec.val1 << ", " << vec.val2
         << "]";
    return __os;
}

}

#endif

#endif
