/**
 * Copyright 2012-2013 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_HPP
#define FLAT_ARRAY_HPP

#include <libflatarray/aligned_allocator.hpp>
#include <stdexcept>
#include <boost/preprocessor/seq.hpp>

// fix compilation for non-cuda builds
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#define LIBFLATARRAY_INDEX(X, Y, Z, DIM_X, DIM_Y, DIM_Z, INDEX) \
    (INDEX + Z * (DIM_X * DIM_Y) + Y * DIM_X + X)

#define LIBFLATARRAY_PARAMS_FULL(X, Y, Z, DIM_X, DIM_Y, DIM_Z, INDEX)	\
    DIM_X, DIM_Y, DIM_Z, LIBFLATARRAY_INDEX(X, Y, Z, DIM_X, DIM_Y, DIM_Z, INDEX)

/**
 * This macro is convenient when you need to return instances of the
 * soa_accessor from your own functions.
 */
#define LIBFLATARRAY_PARAMS						\
    LIBFLATARRAY_PARAMS_FULL(X, Y, Z, DIM_X, DIM_Y, DIM_Z, INDEX)

/**
 * Use this macro to give LibFlatArray access to your class' private members.
 */
#define LIBFLATARRAY_ACCESS(CELL)                                       \
    template<typename CELL_TYPE, int MY_DIM_X, int MY_DIM_Y, int MY_DIM_Z, int INDEX> \
    friend class LibFlatArray::soa_accessor;

namespace LibFlatArray {

/**
 * Allow the user to access the number of data members of the SoA type.
 */
template<typename CELL_TYPE>
class number_of_members;

/**
 * Accumulate the sizes of the individual data members. This may be
 * lower than sizeof(CELL_TYPE) as structs/objects in C++ may need
 * padding. We can avoid the padding of individual members in a SoA
 * memory layout.
 */
template<typename CELL_TYPE>
class aggregated_member_size;

/**
 * This class provides an object-oriented view to a "Struct of
 * Arrays"-style grid. It requires the user to register the type CELL
 * using the macro LIBFLATARRAY_REGISTER_SOA. It provides an
 * operator[] which can be used to access neighboring cells.
 */
template<typename CELL, int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
class soa_accessor;

/**
 * Instances of this class will be returned by
 * soa_accessor::operator[]. Separation of both is done to reduce
 * compilation times.
 */
template<typename CELL, int GRID_DIM, int INDEX>
class soa_accessor_final;

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
        int z,
	int count) :
        target(target),
        x(x),
        y(y),
        z(z),
	count(count)
    {}

    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void operator()(const soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor, int *index) const
    {
        *index =
            z * DIM_X * DIM_Y +
            y * DIM_X +
            x;
	CELL *cursor = target;

	for (int i = 0; i < count; ++i) {
            accessor >> *cursor;
	    ++cursor;
	    ++*index;
	}
    }

private:
    CELL *target;
    int x;
    int y;
    int z;
    int count;
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
        int z,
        int count) :
        source(source),
        x(x),
        y(y),
        z(z),
        count(count)
    {}

    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor, int *index) const
    {
        *index =
            z * DIM_X * DIM_Y +
            y * DIM_X +
            x;
        const CELL *cursor = source;
        for (int i = 0; i < count; ++i) {
            accessor << *cursor;
            ++cursor;
            ++(*index);
        }
    }

private:
    const CELL *source;
    int x;
    int y;
    int z;
    int count;
};

/**
 * The purpose of this functor is to load a row of cells which are
 * already prepackaged (in SoA form) in a raw data segment (i.e. all
 * members are stored in a consecutive array of the given length and
 * all arrays are concatenated).
 */
template<typename CELL>
class load_functor
{
public:
    load_functor(
        size_t x,
        size_t y,
        size_t z,
        const char *source,
        int count) :
        source(source),
        count(count),
        x(x),
        y(y),
        z(z)
    {}

    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor, int *index) const
    {
        *index = x + y * DIM_X + z * DIM_X * DIM_Y;
        accessor.load(source, count);
    }

private:
    const char *source;
    int count;
    int x;
    int y;
    int z;
};

/**
 * Same as save_functor, but the other way around.
 */
template<typename CELL>
class save_functor
{
public:
    save_functor(
        size_t x,
        size_t y,
        size_t z,
        char *target,
        int count) :
        target(target),
        count(count),
        x(x),
        y(y),
        z(z)
    {}

    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor, int *index)
    {
        *index = x + y * DIM_X + z * DIM_X * DIM_Y;
        accessor.save(target, count);
    }

private:
    char *target;
    int count;
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
        *byte_size = aggregated_member_size<CELL>::VALUE * DIM_X * DIM_Y * DIM_Z;
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
    cell.BOOST_PP_SEQ_ELEM(1, MEMBER) = this->BOOST_PP_SEQ_ELEM(1, MEMBER)();

#define COPY_SOA_MEMBER_ARRAY_IN(MEMBER_INDEX, CELL, MEMBER)            \
    std::copy(                                                          \
        (const BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                          \
            source + detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET * count), \
        (const BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                          \
            source + detail::flat_array::offset<CELL, MEMBER_INDEX - 1>::OFFSET * count), \
        &this->BOOST_PP_SEQ_ELEM(1, MEMBER)());

#define COPY_SOA_MEMBER_ARRAY_OUT(MEMBER_INDEX, CELL, MEMBER)           \
    std::copy(                                                          \
        &this->BOOST_PP_SEQ_ELEM(1, MEMBER)(),                          \
        &this->BOOST_PP_SEQ_ELEM(1, MEMBER)() + count,                  \
        (BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                                \
            target + detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET * count));

template<int X, int Y, int Z>
class coord
{};

#define LIBFLATARRAY_REGISTER_SOA(CELL_TYPE, CELL_MEMBERS)              \
    namespace LibFlatArray {                                            \
    BOOST_PP_SEQ_FOR_EACH(                                              \
        DEFINE_FIELD_OFFSET,                                            \
        CELL_TYPE,                                                      \
        CELL_MEMBERS)                                                   \
                                                                        \
    template<>                                                          \
    class number_of_members<CELL_TYPE>                                  \
    {                                                                   \
    public:                                                             \
        static const size_t VALUE = BOOST_PP_SEQ_SIZE(CELL_MEMBERS);    \
    };                                                                  \
                                                                        \
    template<>                                                          \
    class aggregated_member_size<CELL_TYPE>                             \
    {                                                                   \
    private:                                                            \
        static const size_t INDEX = number_of_members<CELL_TYPE>::VALUE; \
                                                                        \
    public:                                                             \
        static const size_t VALUE =                                     \
            detail::flat_array::offset<CELL_TYPE, INDEX>::OFFSET;       \
    };                                                                  \
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
        soa_accessor(char *data=0, int *index=0) :                      \
            data(data),                                                 \
            index(index)                                                \
        {}                                                              \
                                                                        \
        template<int X, int Y, int Z>                                   \
        inline                                                          \
        __host__ __device__                                             \
        soa_accessor<CELL_TYPE, LIBFLATARRAY_PARAMS> operator[](coord<X, Y, Z>) const \
        {                                                               \
            return soa_accessor<CELL_TYPE, LIBFLATARRAY_PARAMS>(data, index); \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void operator=(const CELL_TYPE& cell)                           \
        {                                                               \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                COPY_SOA_MEMBER_IN,                                     \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void operator<<(const CELL_TYPE& cell)                          \
        {                                                               \
            (*this) = cell;                                             \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void operator>>(CELL_TYPE& cell) const                          \
        {                                                               \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                COPY_SOA_MEMBER_OUT,                                    \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void load(const char *source, size_t count)                     \
        {                                                               \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                COPY_SOA_MEMBER_ARRAY_IN,                               \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void save(char *target, size_t count) const                     \
        {                                                               \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                COPY_SOA_MEMBER_ARRAY_OUT,                              \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
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
                                                                        \
    }                                                                   \
                                                                        \
    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>                \
    __host__ __device__                                                 \
    inline                                                              \
    void operator<<(                                                    \
        CELL_TYPE& cell,                                                \
        const LibFlatArray::soa_accessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX> soa) \
    {                                                                   \
        soa >> cell;                                                    \
    }

template<typename CELL, int MY_SIZE>
class soa_array
{
public:
    typedef CELL Cell;
    static const int SIZE = MY_SIZE;
    static const int BYTE_SIZE = aggregated_member_size<CELL>::VALUE * SIZE;

    inline
    __host__ __device__
    soa_array(int elements = 0, const CELL& value = CELL()) :
        elements(elements),
        index(0)
    {
        int i = 0;
        soa_accessor<CELL, SIZE, 0, 0, 0> accessor(data, &i);
        for (; i < elements; ++i) {
            accessor << value;
        }
    }

    inline
    __host__ __device__
    soa_accessor<CELL, SIZE, 1, 1, 0> operator[](int& index)
    {
        return soa_accessor<CELL, SIZE, 1, 1, 0>(data, &index);
    }

    inline
    __host__ __device__
    const soa_accessor<CELL, SIZE, 1, 1, 0> operator[](int& index) const
    {
        return soa_accessor<CELL, SIZE, 1, 1, 0>(data, &index);
    }

    inline
    __host__ __device__
    void operator<<(const CELL& cell)
    {
        if (elements >= SIZE) {
            throw std::out_of_range("capacity exceeded");
        }

        (*this)[elements] = cell;
        ++elements;
    }

    inline
    __host__ __device__
    size_t size() const
    {
        return elements;
    }

private:
    char data[BYTE_SIZE];
    int elements;
    int index;
};

template<typename CELL_TYPE, typename ALLOCATOR = aligned_allocator<char, 4096> >
class soa_grid
{
public:
    friend class TestAssignment;

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
        data = ALLOCATOR().allocate(byte_size());
        std::copy(other.data, other.data + byte_size(), data);
    }

    ~soa_grid()
    {
        ALLOCATOR().deallocate(data, byte_size());
    }

    soa_grid& operator=(const soa_grid& other)
    {
        ALLOCATOR().deallocate(data, byte_size());

        dim_x = other.dim_x;
        dim_y = other.dim_y;
        dim_z = other.dim_z;
        my_byte_size = other.my_byte_size;

        data = ALLOCATOR().allocate(byte_size());
        std::copy(other.data, other.data + byte_size(), data);

        return *this;
    }

    void swap(soa_grid& other)
    {
        std::swap(dim_x, other.dim_x);
        std::swap(dim_x, other.dim_x);
        std::swap(dim_x, other.dim_x);
        std::swap(my_byte_size, other.my_byte_size);
        std::swap(data, other.data);
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

    void set(size_t x, size_t y, size_t z, const CELL_TYPE& cell)
    {
        int index = 0;
        callback(detail::flat_array::set_instance_functor<CELL_TYPE>(&cell, x, y, z, 1), &index);
    }

    void set(size_t x, size_t y, size_t z, const CELL_TYPE *cells, size_t count)
    {
        int index = 0;
        callback(detail::flat_array::set_instance_functor<CELL_TYPE>(cells, x, y, z, count), &index);
    }

    CELL_TYPE get(size_t x, size_t y, size_t z) const
    {
        CELL_TYPE cell;
        int index = 0;
        callback(detail::flat_array::get_instance_functor<CELL_TYPE>(&cell, x, y, z, 1), &index);

        return cell;
    }

    void get(size_t x, size_t y, size_t z, CELL_TYPE *cells, size_t count) const
    {
        int index = 0;
        callback(detail::flat_array::get_instance_functor<CELL_TYPE>(cells, x, y, z, count), &index);
    }

    void load(size_t x, size_t y, size_t z, const char *data, size_t count)
    {
        int index = 0;
        callback(detail::flat_array::load_functor<CELL_TYPE>(x, y, z, data, count), &index);
    }

    void save(size_t x, size_t y, size_t z, char *data, size_t count) const
    {
        int index = 0;
        callback(detail::flat_array::save_functor<CELL_TYPE>(x, y, z, data, count), &index);
    }

    size_t byte_size() const
    {
        return my_byte_size;
    }

    char *get_data()
    {
        return data;
    }

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
        ALLOCATOR().deallocate(data, byte_size());

        // we need callback() to round up our grid size
        callback(detail::flat_array::set_byte_size_functor<CELL_TYPE>(&my_byte_size), 0);
        data = ALLOCATOR().allocate(byte_size());
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

        // CASE(  1);
        CASE( 32);
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

        // CASE(  1);
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

namespace std
{
    template<typename CELL_TYPE>
    void swap(LibFlatArray::soa_grid<CELL_TYPE>& a, LibFlatArray::soa_grid<CELL_TYPE>& b)
    {
        a.swap(b);
    }
}

#endif
