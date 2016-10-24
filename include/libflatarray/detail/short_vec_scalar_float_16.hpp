/**
 * Copyright 2014-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_FLOAT_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_SCALAR_FLOAT_16_HPP

#if (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_SCALAR) ||          \
    (LIBFLATARRAY_WIDEST_VECTOR_ISA == LIBFLATARRAY_QPX)

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
class short_vec<float, 16>
{
public:
    static const int ARITY = 16;
    typedef unsigned short mask_type;
    typedef short_vec_strategy::scalar strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 16>& vec);

    inline
    short_vec(const float data = 0) :
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
    short_vec(const float *data)
    {
        load(data);
    }

    inline
    short_vec(
        const float val1,
        const float val2,
        const float val3,
        const float val4,
        const float val5,
        const float val6,
        const float val7,
        const float val8,
        const float val9,
        const float val10,
        const float val11,
        const float val12,
        const float val13,
        const float val14,
        const float val15,
        const float val16) :
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
    float get(const int i) const
    {
        switch (i) {
        case 0:
            return val1;
        case 1:
            return val2;
        case 2:
            return val3;
        case 3:
            return val4;
        case 4:
            return val5;
        case 5:
            return val6;
        case 6:
            return val7;
        case 7:
            return val8;
        case 8:
            return val9;
        case 9:
            return val10;
        case 10:
            return val11;
        case 11:
            return val12;
        case 12:
            return val13;
        case 13:
            return val14;
        case 14:
            return val15;
        default:
            return val16;
        }
    }

    inline
    void operator-=(const short_vec<float, 16>& other)
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
    short_vec<float, 16> operator-(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
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
    void operator+=(const short_vec<float, 16>& other)
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
    short_vec<float, 16> operator+(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
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
    void operator*=(const short_vec<float, 16>& other)
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
    short_vec<float, 16> operator*(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
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
    void operator/=(const short_vec<float, 16>& other)
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
    short_vec<float, 16> operator/(const short_vec<float, 16>& other) const
    {
        return short_vec<float, 16>(
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

#define LFA_SHORTVEC_COMPARE_HELPER(V1, V2, OP) ((V1) OP (V2))
    inline
    mask_type operator<(const short_vec<float, 16>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val1,  other.val1,  <) <<  0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val2,  other.val2,  <) <<  1) +
            (LFA_SHORTVEC_COMPARE_HELPER(val3,  other.val3,  <) <<  2) +
            (LFA_SHORTVEC_COMPARE_HELPER(val4,  other.val4,  <) <<  3) +
            (LFA_SHORTVEC_COMPARE_HELPER(val5,  other.val5,  <) <<  4) +
            (LFA_SHORTVEC_COMPARE_HELPER(val6,  other.val6,  <) <<  5) +
            (LFA_SHORTVEC_COMPARE_HELPER(val7,  other.val7,  <) <<  6) +
            (LFA_SHORTVEC_COMPARE_HELPER(val8,  other.val8,  <) <<  7) +
            (LFA_SHORTVEC_COMPARE_HELPER(val9,  other.val9,  <) <<  8) +
            (LFA_SHORTVEC_COMPARE_HELPER(val10, other.val10, <) <<  9) +
            (LFA_SHORTVEC_COMPARE_HELPER(val11, other.val11, <) << 10) +
            (LFA_SHORTVEC_COMPARE_HELPER(val12, other.val12, <) << 11) +
            (LFA_SHORTVEC_COMPARE_HELPER(val13, other.val13, <) << 12) +
            (LFA_SHORTVEC_COMPARE_HELPER(val14, other.val14, <) << 13) +
            (LFA_SHORTVEC_COMPARE_HELPER(val15, other.val15, <) << 14) +
            (LFA_SHORTVEC_COMPARE_HELPER(val16, other.val16, <) << 15);
    }

    inline
    mask_type operator<=(const short_vec<float, 16>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val1,  other.val1,  <=) <<  0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val2,  other.val2,  <=) <<  1) +
            (LFA_SHORTVEC_COMPARE_HELPER(val3,  other.val3,  <=) <<  2) +
            (LFA_SHORTVEC_COMPARE_HELPER(val4,  other.val4,  <=) <<  3) +
            (LFA_SHORTVEC_COMPARE_HELPER(val5,  other.val5,  <=) <<  4) +
            (LFA_SHORTVEC_COMPARE_HELPER(val6,  other.val6,  <=) <<  5) +
            (LFA_SHORTVEC_COMPARE_HELPER(val7,  other.val7,  <=) <<  6) +
            (LFA_SHORTVEC_COMPARE_HELPER(val8,  other.val8,  <=) <<  7) +
            (LFA_SHORTVEC_COMPARE_HELPER(val9,  other.val9,  <=) <<  8) +
            (LFA_SHORTVEC_COMPARE_HELPER(val10, other.val10, <=) <<  9) +
            (LFA_SHORTVEC_COMPARE_HELPER(val11, other.val11, <=) << 10) +
            (LFA_SHORTVEC_COMPARE_HELPER(val12, other.val12, <=) << 11) +
            (LFA_SHORTVEC_COMPARE_HELPER(val13, other.val13, <=) << 12) +
            (LFA_SHORTVEC_COMPARE_HELPER(val14, other.val14, <=) << 13) +
            (LFA_SHORTVEC_COMPARE_HELPER(val15, other.val15, <=) << 14) +
            (LFA_SHORTVEC_COMPARE_HELPER(val16, other.val16, <=) << 15);
    }

    inline
    mask_type operator==(const short_vec<float, 16>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val1,  other.val1,  ==) <<  0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val2,  other.val2,  ==) <<  1) +
            (LFA_SHORTVEC_COMPARE_HELPER(val3,  other.val3,  ==) <<  2) +
            (LFA_SHORTVEC_COMPARE_HELPER(val4,  other.val4,  ==) <<  3) +
            (LFA_SHORTVEC_COMPARE_HELPER(val5,  other.val5,  ==) <<  4) +
            (LFA_SHORTVEC_COMPARE_HELPER(val6,  other.val6,  ==) <<  5) +
            (LFA_SHORTVEC_COMPARE_HELPER(val7,  other.val7,  ==) <<  6) +
            (LFA_SHORTVEC_COMPARE_HELPER(val8,  other.val8,  ==) <<  7) +
            (LFA_SHORTVEC_COMPARE_HELPER(val9,  other.val9,  ==) <<  8) +
            (LFA_SHORTVEC_COMPARE_HELPER(val10, other.val10, ==) <<  9) +
            (LFA_SHORTVEC_COMPARE_HELPER(val11, other.val11, ==) << 10) +
            (LFA_SHORTVEC_COMPARE_HELPER(val12, other.val12, ==) << 11) +
            (LFA_SHORTVEC_COMPARE_HELPER(val13, other.val13, ==) << 12) +
            (LFA_SHORTVEC_COMPARE_HELPER(val14, other.val14, ==) << 13) +
            (LFA_SHORTVEC_COMPARE_HELPER(val15, other.val15, ==) << 14) +
            (LFA_SHORTVEC_COMPARE_HELPER(val16, other.val16, ==) << 15);
    }

    inline
    mask_type operator>(const short_vec<float, 16>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val1,  other.val1,  >) <<  0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val2,  other.val2,  >) <<  1) +
            (LFA_SHORTVEC_COMPARE_HELPER(val3,  other.val3,  >) <<  2) +
            (LFA_SHORTVEC_COMPARE_HELPER(val4,  other.val4,  >) <<  3) +
            (LFA_SHORTVEC_COMPARE_HELPER(val5,  other.val5,  >) <<  4) +
            (LFA_SHORTVEC_COMPARE_HELPER(val6,  other.val6,  >) <<  5) +
            (LFA_SHORTVEC_COMPARE_HELPER(val7,  other.val7,  >) <<  6) +
            (LFA_SHORTVEC_COMPARE_HELPER(val8,  other.val8,  >) <<  7) +
            (LFA_SHORTVEC_COMPARE_HELPER(val9,  other.val9,  >) <<  8) +
            (LFA_SHORTVEC_COMPARE_HELPER(val10, other.val10, >) <<  9) +
            (LFA_SHORTVEC_COMPARE_HELPER(val11, other.val11, >) << 10) +
            (LFA_SHORTVEC_COMPARE_HELPER(val12, other.val12, >) << 11) +
            (LFA_SHORTVEC_COMPARE_HELPER(val13, other.val13, >) << 12) +
            (LFA_SHORTVEC_COMPARE_HELPER(val14, other.val14, >) << 13) +
            (LFA_SHORTVEC_COMPARE_HELPER(val15, other.val15, >) << 14) +
            (LFA_SHORTVEC_COMPARE_HELPER(val16, other.val16, >) << 15);
    }

    inline
    mask_type operator>=(const short_vec<float, 16>& other) const
    {
        return
            (LFA_SHORTVEC_COMPARE_HELPER(val1,  other.val1,  >=) <<  0) +
            (LFA_SHORTVEC_COMPARE_HELPER(val2,  other.val2,  >=) <<  1) +
            (LFA_SHORTVEC_COMPARE_HELPER(val3,  other.val3,  >=) <<  2) +
            (LFA_SHORTVEC_COMPARE_HELPER(val4,  other.val4,  >=) <<  3) +
            (LFA_SHORTVEC_COMPARE_HELPER(val5,  other.val5,  >=) <<  4) +
            (LFA_SHORTVEC_COMPARE_HELPER(val6,  other.val6,  >=) <<  5) +
            (LFA_SHORTVEC_COMPARE_HELPER(val7,  other.val7,  >=) <<  6) +
            (LFA_SHORTVEC_COMPARE_HELPER(val8,  other.val8,  >=) <<  7) +
            (LFA_SHORTVEC_COMPARE_HELPER(val9,  other.val9,  >=) <<  8) +
            (LFA_SHORTVEC_COMPARE_HELPER(val10, other.val10, >=) <<  9) +
            (LFA_SHORTVEC_COMPARE_HELPER(val11, other.val11, >=) << 10) +
            (LFA_SHORTVEC_COMPARE_HELPER(val12, other.val12, >=) << 11) +
            (LFA_SHORTVEC_COMPARE_HELPER(val13, other.val13, >=) << 12) +
            (LFA_SHORTVEC_COMPARE_HELPER(val14, other.val14, >=) << 13) +
            (LFA_SHORTVEC_COMPARE_HELPER(val15, other.val15, >=) << 14) +
            (LFA_SHORTVEC_COMPARE_HELPER(val16, other.val16, >=) << 15);
    }
#undef LFA_SHORTVEC_COMPARE_HELPER

    inline
    short_vec<float, 16> sqrt() const
    {
        return short_vec<float, 16>(
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
    void load(const float *data)
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
    void load_aligned(const float *data)
    {
        load(data);
    }

    inline
    void store(float *data) const
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
    void scatter(float *ptr, const int *offsets) const
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

    inline
    void blend(const mask_type& mask, const short_vec<float, 1>& other)
    {
        if (mask & (1 << 0)) {
            val1 = other.val1;
        }
        if (mask & (1 << 1)) {
            val2 = other.val2;
        }
        if (mask & (1 << 2)) {
            val3 = other.val3;
        }
        if (mask & (1 << 3)) {
            val4 = other.val4;
        }
        if (mask & (1 << 4)) {
            val5 = other.val5;
        }
        if (mask & (1 << 5)) {
            val6 = other.val6;
        }
        if (mask & (1 << 6)) {
            val7 = other.val7;
        }
        if (mask & (1 << 7)) {
            val8 = other.val8;
        }
        if (mask & (1 << 8)) {
            val9 = other.val9;
        }
        if (mask & (1 << 9)) {
            val10 = other.val10;
        }
        if (mask & (1 << 10)) {
            val11 = other.val11;
        }
        if (mask & (1 << 11)) {
            val12 = other.val12;
        }
        if (mask & (1 << 12)) {
            val13 = other.val13;
        }
        if (mask & (1 << 13)) {
            val14 = other.val14;
        }
        if (mask & (1 << 14)) {
            val15 = other.val15;
        }
        if (mask & (1 << 15)) {
            val16 = other.val16;
        }
    }

private:
    float val1;
    float val2;
    float val3;
    float val4;
    float val5;
    float val6;
    float val7;
    float val8;
    float val9;
    float val10;
    float val11;
    float val12;
    float val13;
    float val14;
    float val15;
    float val16;
};

inline
void operator<<(float *data, const short_vec<float, 16>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<float, 16> sqrt(const short_vec<float, 16>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 16>& vec)
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
