/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_DOUBLE_8_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_DOUBLE_8_HPP

#ifndef __SSE__
#ifndef __AVX__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<double, 8>
{
public:
    static const int ARITY = 8;

    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 8>& vec);

    inline
    short_vec(const double data = 0) :
        val1(data),
        val2(data),
        val3(data),
        val4(data),
        val5(data),
        val6(data),
        val7(data),
        val8(data)
    {}

    inline
    short_vec(const double *data) :
        val1( *(data +  0)),
        val2( *(data +  1)),
        val3( *(data +  2)),
        val4( *(data +  3)),
        val5( *(data +  4)),
        val6( *(data +  5)),
        val7( *(data +  6)),
        val8( *(data +  7))
    {}

    inline
    short_vec(
        const double val1,
        const double val2,
        const double val3,
        const double val4,
        const double val5,
        const double val6,
        const double val7,
        const double val8) :
        val1( val1),
        val2( val2),
        val3( val3),
        val4( val4),
        val5( val5),
        val6( val6),
        val7( val7),
        val8( val8)
    {}

    inline
    void operator-=(const short_vec<double, 8>& other)
    {
        val1  -= other.val1;
        val2  -= other.val2;
        val3  -= other.val3;
        val4  -= other.val4;
        val5  -= other.val5;
        val6  -= other.val6;
        val7  -= other.val7;
        val8  -= other.val8;
    }

    inline
    short_vec<double, 8> operator-(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            val1  - other.val1,
            val2  - other.val2,
            val3  - other.val3,
            val4  - other.val4,
            val5  - other.val5,
            val6  - other.val6,
            val7  - other.val7,
            val8  - other.val8);
    }

    inline
    void operator+=(const short_vec<double, 8>& other)
    {
        val1  += other.val1;
        val2  += other.val2;
        val3  += other.val3;
        val4  += other.val4;
        val5  += other.val5;
        val6  += other.val6;
        val7  += other.val7;
        val8  += other.val8;
    }

    inline
    short_vec<double, 8> operator+(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            val1  + other.val1,
            val2  + other.val2,
            val3  + other.val3,
            val4  + other.val4,
            val5  + other.val5,
            val6  + other.val6,
            val7  + other.val7,
            val8  + other.val8);
    }

    inline
    void operator*=(const short_vec<double, 8>& other)
    {
        val1  *= other.val1;
        val2  *= other.val2;
        val3  *= other.val3;
        val4  *= other.val4;
        val5  *= other.val5;
        val6  *= other.val6;
        val7  *= other.val7;
        val8  *= other.val8;
    }

    inline
    short_vec<double, 8> operator*(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            val1  * other.val1,
            val2  * other.val2,
            val3  * other.val3,
            val4  * other.val4,
            val5  * other.val5,
            val6  * other.val6,
            val7  * other.val7,
            val8  * other.val8);
    }

    inline
    void operator/=(const short_vec<double, 8>& other)
    {
        val1  /= other.val1;
        val2  /= other.val2;
        val3  /= other.val3;
        val4  /= other.val4;
        val5  /= other.val5;
        val6  /= other.val6;
        val7  /= other.val7;
        val8  /= other.val8;
    }

    inline
    short_vec<double, 8> operator/(const short_vec<double, 8>& other) const
    {
        return short_vec<double, 8>(
            val1  / other.val1,
            val2  / other.val2,
            val3  / other.val3,
            val4  / other.val4,
            val5  / other.val5,
            val6  / other.val6,
            val7  / other.val7,
            val8  / other.val8);
    }

    inline
    short_vec<double, 8> sqrt() const
    {
        return short_vec<double, 8>(
            std::sqrt(val1),
            std::sqrt(val2),
            std::sqrt(val3),
            std::sqrt(val4),
            std::sqrt(val5),
            std::sqrt(val6),
            std::sqrt(val7),
            std::sqrt(val8));
    }

    inline
    void store(double *data) const
    {
        *(data +  0) = val1;
        *(data +  1) = val2;
        *(data +  2) = val3;
        *(data +  3) = val4;
        *(data +  4) = val5;
        *(data +  5) = val6;
        *(data +  6) = val7;
        *(data +  7) = val8;
    }

private:
    double val1;
    double val2;
    double val3;
    double val4;
    double val5;
    double val6;
    double val7;
    double val8;
};

inline
void operator<<(double *data, const short_vec<double, 8>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

short_vec<double, 8> sqrt(const short_vec<double, 8>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 8>& vec)
{
    __os << "["  << vec.val1  << ", " << vec.val2  << ", " << vec.val3  << ", " << vec.val4
         << ", " << vec.val5  << ", " << vec.val6  << ", " << vec.val7  << ", " << vec.val8
         << "]";
    return __os;
}

}

#endif
#endif

#endif
