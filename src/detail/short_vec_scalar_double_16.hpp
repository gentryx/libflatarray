/**
 * Copyright 2014-2016 Andreas Schäfer
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
class short_vec<double, 16>
{
public:
    static const int ARITY = 16;

    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 16>& vec);

    inline
    short_vec(const double data = 0) :
        val1(data),
        val2(data),
        val3(data),
        val4(data),
        val5(data),
        val6(data),
        val7(data),
        val8(data),
        val9(data),
        val10(data),
        val11(data),
        val12(data),
        val13(data),
        val14(data),
        val15(data),
        val16(data)
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
        val1( val1),
        val2( val2),
        val3( val3),
        val4( val4),
        val5( val5),
        val6( val6),
        val7( val7),
        val8( val8),
        val9( val9),
        val10(val10),
        val11(val11),
        val12(val12),
        val13(val13),
        val14(val14),
        val15(val15),
        val16(val16)
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
            val1 ||
            val2 ||
            val3 ||
            val4 ||
            val5 ||
            val6 ||
            val7 ||
            val8 ||
            val9 ||
            val10 ||
            val11 ||
            val12 ||
            val13 ||
            val14 ||
            val15 ||
            val16;
    }

    inline
    void operator-=(const short_vec<double, 16>& other)
    {
        val1  -= other.val1;
        val2  -= other.val2;
        val3  -= other.val3;
        val4  -= other.val4;
        val5  -= other.val5;
        val6  -= other.val6;
        val7  -= other.val7;
        val8  -= other.val8;
        val9  -= other.val9;
        val10 -= other.val10;
        val11 -= other.val11;
        val12 -= other.val12;
        val13 -= other.val13;
        val14 -= other.val14;
        val15 -= other.val15;
        val16 -= other.val16;
    }

    inline
    short_vec<double, 16> operator-(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            val1  - other.val1,
            val2  - other.val2,
            val3  - other.val3,
            val4  - other.val4,
            val5  - other.val5,
            val6  - other.val6,
            val7  - other.val7,
            val8  - other.val8,
            val9  - other.val9,
            val10 - other.val10,
            val11 - other.val11,
            val12 - other.val12,
            val13 - other.val13,
            val14 - other.val14,
            val15 - other.val15,
            val16 - other.val16);
    }

    inline
    void operator+=(const short_vec<double, 16>& other)
    {
        val1  += other.val1;
        val2  += other.val2;
        val3  += other.val3;
        val4  += other.val4;
        val5  += other.val5;
        val6  += other.val6;
        val7  += other.val7;
        val8  += other.val8;
        val9  += other.val9;
        val10 += other.val10;
        val11 += other.val11;
        val12 += other.val12;
        val13 += other.val13;
        val14 += other.val14;
        val15 += other.val15;
        val16 += other.val16;
    }

    inline
    short_vec<double, 16> operator+(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            val1  + other.val1,
            val2  + other.val2,
            val3  + other.val3,
            val4  + other.val4,
            val5  + other.val5,
            val6  + other.val6,
            val7  + other.val7,
            val8  + other.val8,
            val9  + other.val9,
            val10 + other.val10,
            val11 + other.val11,
            val12 + other.val12,
            val13 + other.val13,
            val14 + other.val14,
            val15 + other.val15,
            val16 + other.val16);
    }

    inline
    void operator*=(const short_vec<double, 16>& other)
    {
        val1  *= other.val1;
        val2  *= other.val2;
        val3  *= other.val3;
        val4  *= other.val4;
        val5  *= other.val5;
        val6  *= other.val6;
        val7  *= other.val7;
        val8  *= other.val8;
        val9  *= other.val9;
        val10 *= other.val10;
        val11 *= other.val11;
        val12 *= other.val12;
        val13 *= other.val13;
        val14 *= other.val14;
        val15 *= other.val15;
        val16 *= other.val16;
    }

    inline
    short_vec<double, 16> operator*(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            val1  * other.val1,
            val2  * other.val2,
            val3  * other.val3,
            val4  * other.val4,
            val5  * other.val5,
            val6  * other.val6,
            val7  * other.val7,
            val8  * other.val8,
            val9  * other.val9,
            val10 * other.val10,
            val11 * other.val11,
            val12 * other.val12,
            val13 * other.val13,
            val14 * other.val14,
            val15 * other.val15,
            val16 * other.val16);
    }

    inline
    void operator/=(const short_vec<double, 16>& other)
    {
        val1  /= other.val1;
        val2  /= other.val2;
        val3  /= other.val3;
        val4  /= other.val4;
        val5  /= other.val5;
        val6  /= other.val6;
        val7  /= other.val7;
        val8  /= other.val8;
        val9  /= other.val9;
        val10 /= other.val10;
        val11 /= other.val11;
        val12 /= other.val12;
        val13 /= other.val13;
        val14 /= other.val14;
        val15 /= other.val15;
        val16 /= other.val16;
    }

    inline
    short_vec<double, 16> operator/(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            val1  / other.val1,
            val2  / other.val2,
            val3  / other.val3,
            val4  / other.val4,
            val5  / other.val5,
            val6  / other.val6,
            val7  / other.val7,
            val8  / other.val8,
            val9  / other.val9,
            val10 / other.val10,
            val11 / other.val11,
            val12 / other.val12,
            val13 / other.val13,
            val14 / other.val14,
            val15 / other.val15,
            val16 / other.val16);
    }

#define LFA_SHORTVEC_COMPARE_HELPER(V1, V2, OP) (((V1) OP (V2)) ? 1 : 0)
    inline
    short_vec<double, 16> operator<(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, <),
            LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, <),
            LFA_SHORTVEC_COMPARE_HELPER(val3, other.val3, <),
            LFA_SHORTVEC_COMPARE_HELPER(val4, other.val4, <),
            LFA_SHORTVEC_COMPARE_HELPER(val5, other.val5, <),
            LFA_SHORTVEC_COMPARE_HELPER(val6, other.val6, <),
            LFA_SHORTVEC_COMPARE_HELPER(val7, other.val7, <),
            LFA_SHORTVEC_COMPARE_HELPER(val8, other.val8, <),
            LFA_SHORTVEC_COMPARE_HELPER(val9, other.val9, <),
            LFA_SHORTVEC_COMPARE_HELPER(val10, other.val10, <),
            LFA_SHORTVEC_COMPARE_HELPER(val11, other.val11, <),
            LFA_SHORTVEC_COMPARE_HELPER(val12, other.val12, <),
            LFA_SHORTVEC_COMPARE_HELPER(val13, other.val13, <),
            LFA_SHORTVEC_COMPARE_HELPER(val14, other.val14, <),
            LFA_SHORTVEC_COMPARE_HELPER(val15, other.val15, <),
            LFA_SHORTVEC_COMPARE_HELPER(val16, other.val16, <));
    }

    inline
    short_vec<double, 16> operator<=(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val3, other.val3, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val4, other.val4, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val5, other.val5, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val6, other.val6, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val7, other.val7, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val8, other.val8, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val9, other.val9, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val10, other.val10, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val11, other.val11, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val12, other.val12, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val13, other.val13, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val14, other.val14, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val15, other.val15, <=),
            LFA_SHORTVEC_COMPARE_HELPER(val16, other.val16, <=));
    }

    inline
    short_vec<double, 16> operator==(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val3, other.val3, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val4, other.val4, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val5, other.val5, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val6, other.val6, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val7, other.val7, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val8, other.val8, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val9, other.val9, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val10, other.val10, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val11, other.val11, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val12, other.val12, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val13, other.val13, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val14, other.val14, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val15, other.val15, ==),
            LFA_SHORTVEC_COMPARE_HELPER(val16, other.val16, ==));
    }

    inline
    short_vec<double, 16> operator>(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, >),
            LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, >),
            LFA_SHORTVEC_COMPARE_HELPER(val3, other.val3, >),
            LFA_SHORTVEC_COMPARE_HELPER(val4, other.val4, >),
            LFA_SHORTVEC_COMPARE_HELPER(val5, other.val5, >),
            LFA_SHORTVEC_COMPARE_HELPER(val6, other.val6, >),
            LFA_SHORTVEC_COMPARE_HELPER(val7, other.val7, >),
            LFA_SHORTVEC_COMPARE_HELPER(val8, other.val8, >),
            LFA_SHORTVEC_COMPARE_HELPER(val9, other.val9, >),
            LFA_SHORTVEC_COMPARE_HELPER(val10, other.val10, >),
            LFA_SHORTVEC_COMPARE_HELPER(val11, other.val11, >),
            LFA_SHORTVEC_COMPARE_HELPER(val12, other.val12, >),
            LFA_SHORTVEC_COMPARE_HELPER(val13, other.val13, >),
            LFA_SHORTVEC_COMPARE_HELPER(val14, other.val14, >),
            LFA_SHORTVEC_COMPARE_HELPER(val15, other.val15, >),
            LFA_SHORTVEC_COMPARE_HELPER(val16, other.val16, >));
    }

    inline
    short_vec<double, 16> operator>=(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            LFA_SHORTVEC_COMPARE_HELPER(val1, other.val1, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val2, other.val2, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val3, other.val3, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val4, other.val4, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val5, other.val5, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val6, other.val6, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val7, other.val7, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val8, other.val8, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val9, other.val9, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val10, other.val10, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val11, other.val11, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val12, other.val12, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val13, other.val13, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val14, other.val14, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val15, other.val15, >=),
            LFA_SHORTVEC_COMPARE_HELPER(val16, other.val16, >=));
    }
#undef LFA_SHORTVEC_COMPARE_HELPER

    inline
    short_vec<double, 16> sqrt() const
    {
        return short_vec<double, 16>(
            std::sqrt(val1),
            std::sqrt(val2),
            std::sqrt(val3),
            std::sqrt(val4),
            std::sqrt(val5),
            std::sqrt(val6),
            std::sqrt(val7),
            std::sqrt(val8),
            std::sqrt(val9),
            std::sqrt(val10),
            std::sqrt(val11),
            std::sqrt(val12),
            std::sqrt(val13),
            std::sqrt(val14),
            std::sqrt(val15),
            std::sqrt(val16));
    }

    inline
    void load(const double *data)
    {
        val1  = data[ 0];
        val2  = data[ 1];
        val3  = data[ 2];
        val4  = data[ 3];
        val5  = data[ 4];
        val6  = data[ 5];
        val7  = data[ 6];
        val8  = data[ 7];
        val9  = data[ 8];
        val10 = data[ 9];
        val11 = data[10];
        val12 = data[11];
        val13 = data[12];
        val14 = data[13];
        val15 = data[14];
        val16 = data[15];
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
        *(data +  4) = val5;
        *(data +  5) = val6;
        *(data +  6) = val7;
        *(data +  7) = val8;
        *(data +  8) = val9;
        *(data +  9) = val10;
        *(data + 10) = val11;
        *(data + 11) = val12;
        *(data + 12) = val13;
        *(data + 13) = val14;
        *(data + 14) = val15;
        *(data + 15) = val16;
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
        val1  = ptr[offsets[ 0]];
        val2  = ptr[offsets[ 1]];
        val3  = ptr[offsets[ 2]];
        val4  = ptr[offsets[ 3]];
        val5  = ptr[offsets[ 4]];
        val6  = ptr[offsets[ 5]];
        val7  = ptr[offsets[ 6]];
        val8  = ptr[offsets[ 7]];
        val9  = ptr[offsets[ 8]];
        val10 = ptr[offsets[ 9]];
        val11 = ptr[offsets[10]];
        val12 = ptr[offsets[11]];
        val13 = ptr[offsets[12]];
        val14 = ptr[offsets[13]];
        val15 = ptr[offsets[14]];
        val16 = ptr[offsets[15]];
    }

    inline
    void scatter(double *ptr, const int *offsets) const
    {
        ptr[offsets[0]]  = val1;
        ptr[offsets[1]]  = val2;
        ptr[offsets[2]]  = val3;
        ptr[offsets[3]]  = val4;
        ptr[offsets[4]]  = val5;
        ptr[offsets[5]]  = val6;
        ptr[offsets[6]]  = val7;
        ptr[offsets[7]]  = val8;
        ptr[offsets[8]]  = val9;
        ptr[offsets[9]]  = val10;
        ptr[offsets[10]] = val11;
        ptr[offsets[11]] = val12;
        ptr[offsets[12]] = val13;
        ptr[offsets[13]] = val14;
        ptr[offsets[14]] = val15;
        ptr[offsets[15]] = val16;
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
    double val9;
    double val10;
    double val11;
    double val12;
    double val13;
    double val14;
    double val15;
    double val16;
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

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 16>& vec)
{
    __os << "["  << vec.val1  << ", " << vec.val2  << ", " << vec.val3  << ", " << vec.val4
         << ", " << vec.val5  << ", " << vec.val6  << ", " << vec.val7  << ", " << vec.val8
         << ", " << vec.val9  << ", " << vec.val10 << ", " << vec.val11 << ", " << vec.val12
         << ", " << vec.val13 << ", " << vec.val14 << ", " << vec.val15 << ", " << vec.val16
         << "]";
    return __os;
}

}

#endif

#endif
