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
    get_instance_functor(
        CELL *target,
        int x,
        int y,
        int z) :
        target(target),
        x(x),
        y(y),
        z(z)
    {}

    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void operator()(const soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor, int *index) const
    {
        *index =
            z * DIM_X * DIM_Y +
            y * DIM_X +
            x;
        *target << accessor;
    }

private:
    CELL *target;
    int x;
    int y;
    int z;
};

/**
 * This helper class uses an accessor to push an object's members into
 * the SoA storage.
 */
template<typename CELL>
class set_instance_functor
{
public:
    set_instance_functor(
        const CELL *source,
        int x,
        int y,
        int z) :
        source(source),
        x(x),
        y(y),
        z(z)
    {}

    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor, int *index) const
    {
        *index =
            z * DIM_X * DIM_Y +
            y * DIM_X +
            x;
        accessor << *source;
    }

private:
    const CELL *source;
    int x;
    int y;
    int z;
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
    void operator()(const soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor, int *index) const
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

template<typename ACCESSOR1, typename FUNCTOR>
class dual_callback_helper2
{
public:
    dual_callback_helper2(ACCESSOR1 accessor1, int *index1, FUNCTOR functor) :
        accessor1(accessor1),
        index1(index1),
        functor(functor)
    {}

    template<typename ACCESSOR2>
    void operator()(ACCESSOR2 accessor2, int *index2) const
    {
        functor(accessor1, index1, accessor2, index2);
    }

private:
    ACCESSOR1 accessor1;
    int *index1;
    FUNCTOR functor;
};

template<typename GRID2, typename FUNCTOR>
class dual_callback_helper1
{
public:

    dual_callback_helper1(GRID2 *grid2, FUNCTOR functor) :
        grid2(grid2),
        functor(functor)
    {}

    template<typename ACCESSOR1>
    void operator()(ACCESSOR1 accessor1, int *index1)
    {
        dual_callback_helper2<ACCESSOR1, FUNCTOR> helper(accessor1, index1, functor);
        int index2;
        grid2->callback(helper, &index2);
    }

private:
    GRID2 *grid2;
    FUNCTOR functor;
};

class dual_callback_helper
{
public:
    template<typename GRID1, typename GRID2, typename FUNCTOR>
    void operator()(GRID1 *gridOld, GRID2 *gridNew, FUNCTOR functor)
    {
        dual_callback_helper1<GRID2, FUNCTOR> helper(gridNew, functor);
        int index1;
        gridOld->callback(helper, &index1);
    }
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

template<int X, int Y, int Z> class coord {};

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
            soa_accessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX + Z * (DIM_X * DIM_Y) + Y * DIM_X + X> operator[](coord<X, Y, Z>) const \
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
            const char *get_data() const                                \
        {                                                               \
            return data;                                                \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
            char *get_data()                                            \
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
    soa_grid(size_t dim_x = 0, size_t dim_y = 0, size_t dim_z = 0) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z),
        data(0)
    {
        resize();
    }

    soa_grid(const soa_grid& other) :
        dim_x(other.dim_x),
        dim_y(other.dim_y),
        dim_z(other.dim_z),
        my_byte_size(other.my_byte_size)
    {
        data = new char[byte_size()];
        std::copy(other.data, other.data + byte_size(), data);
    }

    ~soa_grid()
    {
        delete data;
    }

    void resize(size_t new_dim_x, size_t new_dim_y, size_t new_dim_z)
    {
        dim_x = new_dim_x;
        dim_y = new_dim_y;
        dim_z = new_dim_z;
        resize();
    }

    template<typename FUNCTOR>
    void callback(FUNCTOR functor, int *index = 0) const
    {
        bind_parameters0(functor, index);
    }

    template<typename FUNCTOR>
    void callback(soa_grid<CELL_TYPE> *otherGrid, FUNCTOR functor) const
    {
        detail::flat_array::dual_callback_helper()(this, otherGrid, functor);
    }

    // fixme: add operator<< and operator>>
    void set(size_t x, size_t y, size_t z, const CELL_TYPE& cell)
    {
        int index = 0;
        callback(detail::flat_array::set_instance_functor<CELL_TYPE>(&cell, x, y, z), &index);
    }

    CELL_TYPE get(size_t x, size_t y, size_t z) const
    {
        CELL_TYPE cell;
        int index = 0;
        callback(detail::flat_array::get_instance_functor<CELL_TYPE>(&cell, x, y, z), &index);

        return cell;
    }

    size_t byte_size() const
    {
        return my_byte_size;
    }

    // fixme: use configurable allocator instead
    char *get_data()
    {
        return data;
    }

    // fixme: use configurable allocator instead
    void set_data(char *new_data)
    {
        data = new_data;
    }

private:
    size_t dim_x;
    size_t dim_y;
    size_t dim_z;
    size_t my_byte_size;
    // We can't use std::vector here since the code needs to work with CUDA, too.
    char *data;

    /**
     * Adapt size of allocated memory to dim_[x-z]
     */
    void resize()
    {
        // we need callback() to round up our grid size
        callback(detail::flat_array::set_byte_size_functor<CELL_TYPE>(&my_byte_size), 0);
        // FIXME: make external allocators work here (e.g. for CUDA)
        if (data) {
            delete data;
        }
        data = new char[byte_size()];
    }

    template<int DIM_X, int DIM_Y, typename FUNCTOR>
    void bind_parameters2(FUNCTOR functor, int *index) const
    {
        size_t size = dim_z;

#define CASE(SIZE)                                                      \
        if (size <= SIZE) {                                             \
            functor(soa_accessor<CELL_TYPE, DIM_X, DIM_Y, SIZE, 0>(     \
                        data, index),                                   \
                    index);                                             \
            return;                                                     \
        }

        // CASE( 32);
        // CASE( 64);
        // CASE( 96);
        // CASE(128);
        // CASE(160);
        // CASE(192);
        // CASE(224);
        CASE(256);
        // CASE(288);
        throw std::logic_error("grid dimension Z too large");

#undef CASE
    }

    template<int DIM_X, typename FUNCTOR>
    void bind_parameters1(FUNCTOR functor, int *index) const
    {
        size_t size = dim_y;

#define CASE(SIZE)                                                      \
        if (size <= SIZE) {                                             \
            bind_parameters2<DIM_X, SIZE>(functor, index);              \
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
        throw std::logic_error("grid dimension Y too large");

#undef CASE
    }

    template<typename FUNCTOR>
    void bind_parameters0(FUNCTOR functor, int *index) const
    {
        size_t size = dim_x;
        // fixme: this would be superfluous if we'd call bind_parameters1
        if (dim_y > size) {
            size = dim_y;
        }

#define CASE(SIZE)                                                      \
        if (size <= SIZE) {                                             \
            /* fixme: */                                                \
            bind_parameters2<SIZE, SIZE>(functor, index);               \
            /* bind_parameters1<SIZE>(functor, index); */               \
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
        throw std::logic_error("grid dimension X too large");

#undef CASE
    }
};

}

#endif
