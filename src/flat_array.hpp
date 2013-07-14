/**
 * Copyright 2012 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef _FLAT_ARRAY_HPP_

#include "stdio.h"
#include <boost/preprocessor/seq.hpp>

// fix compilation for non-cuda builds
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace LibFlatArray {

namespace detail {

namespace flat_array {

template<typename CELL, int I>
class offset;

template<typename CELL>
class offset<CELL, 0>
{
public:
    static const std::size_t OFFSET = 0;
};

template<template<int D> class CARGO>
class bind
{
public:
    template<typename T1, typename T2, typename T3, typename T4>
    void operator()(int size, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
    {
#define CASE(SIZE)                                                      \
        if (size <= SIZE) {                                             \
            CARGO<SIZE>()(size, arg1, arg2, arg3, arg4);                \
            return;                                                     \
        }

        CASE( 32);
        CASE( 64);
        CASE( 96);
        CASE(128);
        CASE(160);
        CASE(192);
    }
#undef CASE
};

}

}

#define DEFINE_FIELD_OFFSET(r, CELL_TYPE, t)                            \
    namespace detail {                                                  \
    namespace flat_array {                                              \
    template<>                                                          \
    class offset<CELL_TYPE, r - 1>                                      \
    {                                                                   \
    public:                                                             \
        static const std::size_t OFFSET = offset<CELL_TYPE, r - 2>::OFFSET +  \
            sizeof(BOOST_PP_SEQ_ELEM(0, t));                            \
    };                                                                  \
    }                                                                   \
    }

        // std::cout << "DIM_X = " << DIM_X << "\n"                        \
        //           << "DIM_Y = " << DIM_Y << "\n"                        \
        //           << "DIM_Z = " << DIM_Z << "\n"                        \
        //           << "field offset = " << detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET << "\n" \
        //           << "sizeof = " << sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)) << "\n" \
        //           << "index = " << *index << "\n"                       \
        //           << "INDEX = " << INDEX << "\n\n";                     \

        // return *(BOOST_PP_SEQ_ELEM(0, MEMBER)*)(data);                  \


        // printf("  actual offset is %d\n", ((DIM_X * DIM_Y * DIM_Z) * detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET + \
        //                                    *index * sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)) + \
        //                                    INDEX  * sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)))); \
        // printf("  d1 %d\n", (DIM_X * DIM_Y * DIM_Z)); \
        // printf("  d2 %d\n", detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET); \
        // printf("  d3 %d\n", *index); \
        // printf("  d4 %d\n", INDEX); \
        // printf("  d5 %d\n", sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)));      \

#define DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, CONST)           \
    inline                                                              \
    __host__ __device__                                                 \
    CONST BOOST_PP_SEQ_ELEM(0, MEMBER)& BOOST_PP_SEQ_ELEM(1, MEMBER)() CONST \
    {                                                                   \
        return *(BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                        \
            data +                                                      \
            (DIM_X * DIM_Y * DIM_Z) * detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET + \
            index * sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)) +              \
            INDEX  * sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)));             \
    }

#define DECLARE_SOA_MEMBER_CONST(MEMBER_INDEX, CELL, MEMBER)    \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, const)

#define DECLARE_SOA_MEMBER_NORMAL(MEMBER_INDEX, CELL, MEMBER)   \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, )

#define COPY_SOA_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER)                  \
    BOOST_PP_SEQ_ELEM(1, MEMBER)() = cell.BOOST_PP_SEQ_ELEM(1, MEMBER);

#define COPY_SOA_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER)                 \
    cell.BOOST_PP_SEQ_ELEM(1, MEMBER) = soa.BOOST_PP_SEQ_ELEM(1, MEMBER)();

template<int X, int Y, int Z> class FixedCoord {};

/**
 * This class provides an object-oriented view to a "Struct of
 * Arrays"-style grid. It requires the user to register the type CELL
 * using the macro LIBFLATARRAY_REGISTER_SOA.
 */
template<typename CELL, int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
class soa_accessor;

#define LIBFLATARRAY_REGISTER_SOA(CELL_TYPE, CELL_MEMBERS)              \
    namespace LibFlatArray {                                            \
    BOOST_PP_SEQ_FOR_EACH(                                              \
        DEFINE_FIELD_OFFSET,                                            \
        CELL_TYPE,                                                      \
        CELL_MEMBERS)                                                   \
                                                                        \
    template<int MY_DIM_X, int MY_DIM_Y, int MY_DIM_Z, int INDEX>       \
    class soa_accessor<CELL_TYPE, MY_DIM_X, MY_DIM_Y, MY_DIM_Z, INDEX>  \
    {                                                                   \
    public:                                                             \
        typedef CELL_TYPE MyCell;                                       \
                                                                        \
        static const int DIM_X = MY_DIM_X;                              \
        static const int DIM_Y = MY_DIM_Y;                              \
        static const int DIM_Z = MY_DIM_Z;                              \
                                                                        \
        __host__ __device__                                             \
            soa_accessor(char *data=0, int index=0) :                   \
            data(data),                                                 \
            index(index)                                                \
            {}                                                          \
                                                                        \
        template<int X, int Y, int Z>                                   \
            inline                                                      \
            __host__ __device__                                         \
            soa_accessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX + Z * (DIM_X * DIM_Y) + Y * DIM_X + X> operator[](FixedCoord<X, Y, Z>) const \
            {                                                           \
                return soa_accessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX + Z * (DIM_X * DIM_Y) + Y * DIM_X + X>(data, index); \
            }                                                           \
                                                                        \
        inline                                                          \
            __host__ __device__                                         \
            soa_accessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX> operator[](int offset) const \
        {                                                               \
            return soa_accessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX>(data, index + offset); \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
            inline                                                      \
            void operator=(const CELL_TYPE& cell)                       \
            {                                                           \
                BOOST_PP_SEQ_FOR_EACH(                                  \
                    COPY_SOA_MEMBER_IN,                                 \
                    CELL_TYPE,                                          \
                    CELL_MEMBERS);                                      \
            }                                                           \
                                                                        \
        __host__ __device__                                             \
            inline                                                      \
            void operator<<(const CELL_TYPE& cell)                      \
            {                                                           \
                (*this) = cell;                                         \
            }                                                           \
                                                                        \
        BOOST_PP_SEQ_FOR_EACH(                                          \
            DECLARE_SOA_MEMBER_NORMAL,                                  \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
                                                                        \
        BOOST_PP_SEQ_FOR_EACH(                                          \
            DECLARE_SOA_MEMBER_CONST,                                   \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
                                                                        \
    private:                                                            \
        char *data;                                                     \
        int index;                                                      \
    };                                                                  \
    }                                                                   \
                                                                        \
    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>                \
    __host__ __device__                                                 \
    inline                                                              \
    void operator<<(                                                    \
        CELL_TYPE& cell,                                                \
        const LibFlatArray::soa_accessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX> soa) \
    {                                                                   \
        BOOST_PP_SEQ_FOR_EACH(                                          \
            COPY_SOA_MEMBER_OUT,                                        \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
    }

template<typename CELL_TYPE>
class soa_grid
{
public:
    const static int DIM_X = 256;
    const static int DIM_Y = 256;
    const static int DIM_Z = 256;

    soa_grid(size_t dimX, size_t dimY, size_t dimZ) :
        dimX(dimX),
        dimY(dimY),
        dimZ(dimZ),
        byteSize(DIM_X * DIM_Y * DIM_Z * sizeof(CELL_TYPE))
    {
        data = new char[byteSize];
    }

    soa_grid(const soa_grid& other) :
        dimX(other.dimX),
        dimY(other.dimY),
        dimZ(other.dimZ),
        byteSize(DIM_X * DIM_Y * DIM_Z * sizeof(CELL_TYPE))
    {
        data = new char[byteSize];
        std::copy(other.data, other.data + byteSize, data);
    }

    template<typename  CALLBACK>
    void iterate(size_t x, size_t y, size_t z)
    {
        size_t indexStart = z * DIM_X * DIM_Y + y * DIM_X + x;
        iterate<CALLBACK>(indexStart);
    }

    template<typename  FUNCTOR>
    void callback(const FUNCTOR& functor) const
    {
        functor(soa_accessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, 0>(data));
    }

    void set(size_t x, size_t y, size_t z, const CELL_TYPE& cell)
    {
        int index = z * DIM_X * DIM_Y + y * DIM_X + x;
        soa_accessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, 0> accessor(data, index);
        accessor << cell;
    }

    CELL_TYPE get(size_t x, size_t y, size_t z) const
    {
        int index = z * DIM_X * DIM_Y + y * DIM_X + x;
        soa_accessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, 0> accessor(data, index);
        CELL_TYPE cell;
        cell << accessor;
        return cell;
    }

    ~soa_grid()
    {
        delete data;
    }

private:
    size_t dimX;
    size_t dimY;
    size_t dimZ;
    size_t byteSize;
    char *data;

};

}

#endif
