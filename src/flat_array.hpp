/**
 * Copyright 2012 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef _FLAT_ARRAY_HPP_

#include <stdexcept>
#include <boost/preprocessor/seq.hpp>

// fix compilation for non-cuda builds
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace LibFlatArray {

/**
 * This class provides an object-oriented view to a "Struct of
 * Arrays"-style grid. It requires the user to register the type CELL
 * using the macro LIBFLATARRAY_REGISTER_SOA.
 */
template<typename CELL, int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
class soa_accessor;

namespace detail {

namespace flat_array {

/**
 * This helper class is used to retrieve objects from the SoA storage
 * via an accessor.
 */
template<typename CELL>
class get_instance_functor
{
public:
    get_instance_functor(CELL *target) :
        target(target)
    {}

    template<typename ACCESSOR>
    void operator()(const ACCESSOR& accessor) const
    {
        *target << accessor;
    }

private:
    CELL *target;
};

/**
 * This helper class uses an accessor to push an object's members into
 * the SoA storage.
 */
template<typename CELL>
class set_instance_functor
{
public:
    set_instance_functor(const CELL *source) :
        source(source)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR accessor) const
    {
        accessor << *source;
    }

private:
    const CELL *source;
};

/**
 * This helper class uses the dimension specified in the accessor to
 * compute how many bytes a grid needs to allocate im memory.
 */
template<typename CELL>
class set_byte_size_functor
{
public:
    set_byte_size_functor(size_t *byte_size) :
        byte_size(byte_size)
    {}

    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void operator()(const soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        *byte_size = sizeof(CELL) * DIM_X * DIM_Y * DIM_Z;
    }

private:
    size_t *byte_size;
};


template<typename CELL, int I>
class offset;

template<typename CELL>
class offset<CELL, 0>
{
public:
    static const std::size_t OFFSET = 0;
};

// fixme: kill this
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
        CASE(224);
        CASE(256);
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

#define DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, CONST)           \
    inline                                                              \
    __host__ __device__                                                 \
    CONST BOOST_PP_SEQ_ELEM(0, MEMBER)& BOOST_PP_SEQ_ELEM(1, MEMBER)() CONST \
    {                                                                   \
        return *(BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                        \
            data + (DIM_X * DIM_Y * DIM_Z) *                            \
            detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET + \
            *index * sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)) +             \
            INDEX  * sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)));             \
    }

#define DECLARE_SOA_MEMBER_CONST(MEMBER_INDEX, CELL, MEMBER)    \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, const)

#define DECLARE_SOA_MEMBER_NORMAL(MEMBER_INDEX, CELL, MEMBER)   \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER,      )

#define COPY_SOA_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER)                  \
    BOOST_PP_SEQ_ELEM(1, MEMBER)() = cell.BOOST_PP_SEQ_ELEM(1, MEMBER);

#define COPY_SOA_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER)                 \
    cell.BOOST_PP_SEQ_ELEM(1, MEMBER) = soa.BOOST_PP_SEQ_ELEM(1, MEMBER)();

template<int X, int Y, int Z> class FixedCoord {};

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
            soa_accessor(char *data=0, int *index=0) :                  \
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
        __host__ __device__                                             \
            const char *getData() const                                 \
        {                                                               \
            return data;                                                \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
            char *getData()                                             \
        {                                                               \
            return data;                                                \
        }                                                               \
                                                                        \
    private:                                                            \
        char *data;                                                     \
        int *index;                                                     \
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
    soa_grid(size_t dim_x, size_t dim_y, size_t dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z),
        data(0)
    {
        // we need callback() to round up our grid size
        callback(detail::flat_array::set_byte_size_functor<CELL_TYPE>(&byte_size), 0);
        // FIXME: make external allocators work here (e.g. for CUDA)
        data = new char[byte_size];
    }

    soa_grid(const soa_grid& other) :
        dim_x(other.dim_x),
        dim_y(other.dim_y),
        dim_z(other.dim_z),
        byte_size(other.byte_size)
    {
        data = new char[byte_size];
        std::copy(other.data, other.data + byte_size, data);
    }

    ~soa_grid()
    {
        delete data;
    }

    template<typename FUNCTOR>
    void callback(FUNCTOR functor, int *index = 0) const
    {
        size_t size = std::max(dim_x, dim_y);
        size = std::max(size, dim_z);

#define CASE(SIZE)                                                      \
        if (size <= SIZE) {                                             \
            functor(soa_accessor<CELL_TYPE, SIZE, SIZE, SIZE, 0>(       \
                        data, index));                                  \
            return;                                                     \
        }

        CASE( 32);
        CASE( 64);
        CASE( 96);
        CASE(128);
        CASE(160);
        CASE(192);
        CASE(224);
        CASE(256);
        CASE(288);
        throw std::logic_error("grid size too large");

#undef CASE

    // fixme: kill this
    // functor(soa_accessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, 0>(data, index));
    }

    // fixme: add operator<< and operator>>
    void set(size_t x, size_t y, size_t z, const CELL_TYPE& cell)
    {
        int index = 0;
        callback(detail::flat_array::set_instance_functor<CELL_TYPE>(&cell), &index);
        // soa_accessor<CELL_TYPE, 32, 32, 32, 0> accessor(data, &index);
        // std::cout << "    gringo1\n";
        // accessor << cell;
        // std::cout << "    gringo2\n";
    }

    CELL_TYPE get(size_t x, size_t y, size_t z) const
    {
        // int index = z * 32 * 32 + y * 32 + x;
        // soa_accessor<CELL_TYPE, 32, 32, 32, 0> accessor(data, &index);
        CELL_TYPE cell;
        int index = 0;
        callback(detail::flat_array::get_instance_functor<CELL_TYPE>(&cell), &index);

        // cell << accessor;
        return cell;
    }

private:
    size_t dim_x;
    size_t dim_y;
    size_t dim_z;
    size_t byte_size;
    // We can't use std::vector here since the code needs to work with CUDA, too.
    char *data;
};

}

#endif
