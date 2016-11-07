/**
 * Copyright 2015 Kurt Kanzenbach
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_INT_32_HPP

#include <libflatarray/config.h>
#include <libflatarray/short_vec_base.hpp>

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SCALAR) ||          \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_ARM_NEON) ||        \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_MIC) ||             \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_QPX)

namespace LibFlatArray {

template<typename CARGO, std::size_t ARITY>
class short_vec;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<int, 32> : public short_vec_base<int, 32>
{
public:
    static const std::size_t ARITY = 32;

    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<int, 32>& vec);

    inline
    short_vec(const int data = 0) :
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
        val16(data),
        val17(data),
        val18(data),
        val19(data),
        val20(data),
        val21(data),
        val22(data),
        val23(data),
        val24(data),
        val25(data),
        val26(data),
        val27(data),
        val28(data),
        val29(data),
        val30(data),
        val31(data),
        val32(data)
    {}

    inline
    short_vec(const int *data)
    {
        load(data);
    }

    inline
    short_vec(
        const int val1,
        const int val2,
        const int val3,
        const int val4,
        const int val5,
        const int val6,
        const int val7,
        const int val8,
        const int val9,
        const int val10,
        const int val11,
        const int val12,
        const int val13,
        const int val14,
        const int val15,
        const int val16,
        const int val17,
        const int val18,
        const int val19,
        const int val20,
        const int val21,
        const int val22,
        const int val23,
        const int val24,
        const int val25,
        const int val26,
        const int val27,
        const int val28,
        const int val29,
        const int val30,
        const int val31,
        const int val32) :
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
        val16(val16),
        val17(val17),
        val18(val18),
        val19(val19),
        val20(val20),
        val21(val21),
        val22(val22),
        val23(val23),
        val24(val24),
        val25(val25),
        val26(val26),
        val27(val27),
        val28(val28),
        val29(val29),
        val30(val30),
        val31(val31),
        val32(val32)
    {}

#ifdef LIBFLATARRAY_WITH_CPP14
    inline
    short_vec(const std::initializer_list<int>& il)
    {
        const int *ptr = static_cast<const int *>(&(*il.begin()));
        load(ptr);
    }
#endif

    inline
    void operator-=(const short_vec<int, 32>& other)
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
        val17 -= other.val17;
        val18 -= other.val18;
        val19 -= other.val19;
        val20 -= other.val20;
        val21 -= other.val21;
        val22 -= other.val22;
        val23 -= other.val23;
        val24 -= other.val24;
        val25 -= other.val25;
        val26 -= other.val26;
        val27 -= other.val27;
        val28 -= other.val28;
        val29 -= other.val29;
        val30 -= other.val30;
        val31 -= other.val31;
        val32 -= other.val32;
    }

    inline
    short_vec<int, 32> operator-(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
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
            val16 - other.val16,
            val17 - other.val17,
            val18 - other.val18,
            val19 - other.val19,
            val20 - other.val20,
            val21 - other.val21,
            val22 - other.val22,
            val23 - other.val23,
            val24 - other.val24,
            val25 - other.val25,
            val26 - other.val26,
            val27 - other.val27,
            val28 - other.val28,
            val29 - other.val29,
            val30 - other.val30,
            val31 - other.val31,
            val32 - other.val32);
    }

    inline
    void operator+=(const short_vec<int, 32>& other)
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
        val17 += other.val17;
        val18 += other.val18;
        val19 += other.val19;
        val20 += other.val20;
        val21 += other.val21;
        val22 += other.val22;
        val23 += other.val23;
        val24 += other.val24;
        val25 += other.val25;
        val26 += other.val26;
        val27 += other.val27;
        val28 += other.val28;
        val29 += other.val29;
        val30 += other.val30;
        val31 += other.val31;
        val32 += other.val32;
    }

    inline
    short_vec<int, 32> operator+(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
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
            val16 + other.val16,
            val17 + other.val17,
            val18 + other.val18,
            val19 + other.val19,
            val20 + other.val20,
            val21 + other.val21,
            val22 + other.val22,
            val23 + other.val23,
            val24 + other.val24,
            val25 + other.val25,
            val26 + other.val26,
            val27 + other.val27,
            val28 + other.val28,
            val29 + other.val29,
            val30 + other.val30,
            val31 + other.val31,
            val32 + other.val32);
    }

    inline
    void operator*=(const short_vec<int, 32>& other)
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
        val17 *= other.val17;
        val18 *= other.val18;
        val19 *= other.val19;
        val20 *= other.val20;
        val21 *= other.val21;
        val22 *= other.val22;
        val23 *= other.val23;
        val24 *= other.val24;
        val25 *= other.val25;
        val26 *= other.val26;
        val27 *= other.val27;
        val28 *= other.val28;
        val29 *= other.val29;
        val30 *= other.val30;
        val31 *= other.val31;
        val32 *= other.val32;
    }

    inline
    short_vec<int, 32> operator*(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
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
            val16 * other.val16,
            val17 * other.val17,
            val18 * other.val18,
            val19 * other.val19,
            val20 * other.val20,
            val21 * other.val21,
            val22 * other.val22,
            val23 * other.val23,
            val24 * other.val24,
            val25 * other.val25,
            val26 * other.val26,
            val27 * other.val27,
            val28 * other.val28,
            val29 * other.val29,
            val30 * other.val30,
            val31 * other.val31,
            val32 * other.val32);
    }

    inline
    void operator/=(const short_vec<int, 32>& other)
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
        val17 /= other.val17;
        val18 /= other.val18;
        val19 /= other.val19;
        val20 /= other.val20;
        val21 /= other.val21;
        val22 /= other.val22;
        val23 /= other.val23;
        val24 /= other.val24;
        val25 /= other.val25;
        val26 /= other.val26;
        val27 /= other.val27;
        val28 /= other.val28;
        val29 /= other.val29;
        val30 /= other.val30;
        val31 /= other.val31;
        val32 /= other.val32;
    }

    inline
    short_vec<int, 32> operator/(const short_vec<int, 32>& other) const
    {
        return short_vec<int, 32>(
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
            val16 / other.val16,
            val17 / other.val17,
            val18 / other.val18,
            val19 / other.val19,
            val20 / other.val20,
            val21 / other.val21,
            val22 / other.val22,
            val23 / other.val23,
            val24 / other.val24,
            val25 / other.val25,
            val26 / other.val26,
            val27 / other.val27,
            val28 / other.val28,
            val29 / other.val29,
            val30 / other.val30,
            val31 / other.val31,
            val32 / other.val32);
    }

    inline
    short_vec<int, 32> sqrt() const
    {
        return short_vec<int, 32>(
            static_cast<int>(std::sqrt(val1)),
            static_cast<int>(std::sqrt(val2)),
            static_cast<int>(std::sqrt(val3)),
            static_cast<int>(std::sqrt(val4)),
            static_cast<int>(std::sqrt(val5)),
            static_cast<int>(std::sqrt(val6)),
            static_cast<int>(std::sqrt(val7)),
            static_cast<int>(std::sqrt(val8)),
            static_cast<int>(std::sqrt(val9)),
            static_cast<int>(std::sqrt(val10)),
            static_cast<int>(std::sqrt(val11)),
            static_cast<int>(std::sqrt(val12)),
            static_cast<int>(std::sqrt(val13)),
            static_cast<int>(std::sqrt(val14)),
            static_cast<int>(std::sqrt(val15)),
            static_cast<int>(std::sqrt(val16)),
            static_cast<int>(std::sqrt(val17)),
            static_cast<int>(std::sqrt(val18)),
            static_cast<int>(std::sqrt(val19)),
            static_cast<int>(std::sqrt(val20)),
            static_cast<int>(std::sqrt(val21)),
            static_cast<int>(std::sqrt(val22)),
            static_cast<int>(std::sqrt(val23)),
            static_cast<int>(std::sqrt(val24)),
            static_cast<int>(std::sqrt(val25)),
            static_cast<int>(std::sqrt(val26)),
            static_cast<int>(std::sqrt(val27)),
            static_cast<int>(std::sqrt(val28)),
            static_cast<int>(std::sqrt(val29)),
            static_cast<int>(std::sqrt(val30)),
            static_cast<int>(std::sqrt(val31)),
            static_cast<int>(std::sqrt(val32)));
    }

    inline
    void load(const int *data)
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
        val17 = data[16];
        val18 = data[17];
        val19 = data[18];
        val20 = data[19];
        val21 = data[20];
        val22 = data[21];
        val23 = data[22];
        val24 = data[23];
        val25 = data[24];
        val26 = data[25];
        val27 = data[26];
        val28 = data[27];
        val29 = data[28];
        val30 = data[29];
        val31 = data[30];
        val32 = data[31];
    }

    inline
    void load_aligned(const int *data)
    {
        load(data);
    }

    inline
    void store(int *data) const
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
        *(data + 16) = val17;
        *(data + 17) = val18;
        *(data + 18) = val19;
        *(data + 19) = val20;
        *(data + 20) = val21;
        *(data + 21) = val22;
        *(data + 22) = val23;
        *(data + 23) = val24;
        *(data + 24) = val25;
        *(data + 25) = val26;
        *(data + 26) = val27;
        *(data + 27) = val28;
        *(data + 28) = val29;
        *(data + 29) = val30;
        *(data + 30) = val31;
        *(data + 31) = val32;
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
        val17 = ptr[offsets[16]];
        val18 = ptr[offsets[17]];
        val19 = ptr[offsets[18]];
        val20 = ptr[offsets[19]];
        val21 = ptr[offsets[20]];
        val22 = ptr[offsets[21]];
        val23 = ptr[offsets[22]];
        val24 = ptr[offsets[23]];
        val25 = ptr[offsets[24]];
        val26 = ptr[offsets[25]];
        val27 = ptr[offsets[26]];
        val28 = ptr[offsets[27]];
        val29 = ptr[offsets[28]];
        val30 = ptr[offsets[29]];
        val31 = ptr[offsets[30]];
        val32 = ptr[offsets[31]];
    }

    inline
    void scatter(int *ptr, const int *offsets) const
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
        ptr[offsets[16]] = val17;
        ptr[offsets[17]] = val18;
        ptr[offsets[18]] = val19;
        ptr[offsets[19]] = val20;
        ptr[offsets[20]] = val21;
        ptr[offsets[21]] = val22;
        ptr[offsets[22]] = val23;
        ptr[offsets[23]] = val24;
        ptr[offsets[24]] = val25;
        ptr[offsets[25]] = val26;
        ptr[offsets[26]] = val27;
        ptr[offsets[27]] = val28;
        ptr[offsets[28]] = val29;
        ptr[offsets[29]] = val30;
        ptr[offsets[30]] = val31;
        ptr[offsets[31]] = val32;
    }

private:
    int val1;
    int val2;
    int val3;
    int val4;
    int val5;
    int val6;
    int val7;
    int val8;
    int val9;
    int val10;
    int val11;
    int val12;
    int val13;
    int val14;
    int val15;
    int val16;
    int val17;
    int val18;
    int val19;
    int val20;
    int val21;
    int val22;
    int val23;
    int val24;
    int val25;
    int val26;
    int val27;
    int val28;
    int val29;
    int val30;
    int val31;
    int val32;
};

inline
void operator<<(int *data, const short_vec<int, 32>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<int, 32> sqrt(const short_vec<int, 32>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<int, 32>& vec)
{
    __os << "["  << vec.val1  << ", " << vec.val2  << ", " << vec.val3  << ", " << vec.val4
         << ", " << vec.val5  << ", " << vec.val6  << ", " << vec.val7  << ", " << vec.val8
         << ", " << vec.val9  << ", " << vec.val10 << ", " << vec.val11 << ", " << vec.val12
         << ", " << vec.val13 << ", " << vec.val14 << ", " << vec.val15 << ", " << vec.val16
         << ", " << vec.val17 << ", " << vec.val18 << ", " << vec.val19 << ", " << vec.val20
         << ", " << vec.val21 << ", " << vec.val22 << ", " << vec.val23 << ", " << vec.val24
         << ", " << vec.val25 << ", " << vec.val26 << ", " << vec.val27 << ", " << vec.val28
         << ", " << vec.val29 << ", " << vec.val30 << ", " << vec.val31 << ", " << vec.val32
         << "]";
    return __os;
}

}

#endif

#endif
