/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_MACROS_HPP
#define FLAT_ARRAY_MACROS_HPP

#include <libflatarray/detail/macros.hpp>

/**
 * This macro is convenient when you need to return instances of the
 * soa_accessor from your own functions.
 */
#define LIBFLATARRAY_PARAMS						\
    LIBFLATARRAY_PARAMS_FULL(X, Y, Z, DIM_X, DIM_Y, DIM_Z, INDEX)

/**
 * Use this macro to give LibFlatArray access to your class' private
 * members.
 */
#define LIBFLATARRAY_ACCESS(CELL)                                       \
    template<typename CELL_TYPE,                                        \
             int MY_DIM_X, int MY_DIM_Y, int MY_DIM_Z, int INDEX>       \
    friend class LibFlatArray::soa_accessor;                            \
                                                                        \
    template<typename CELL_TYPE,                                        \
             int MY_DIM_X, int MY_DIM_Y, int MY_DIM_Z, int INDEX>       \
    friend class LibFlatArray::const_soa_accessor;

/**
 * This macros registers a type with LibFlatArray so that it can be
 * used with soa_grid, soa_array and friends. It will instantiate all
 * templates required for the "Struct of Arrays" (SoA) storage and
 * adds utilities so that user code can also discover properties of
 * the SoA layout.
 */
#define LIBFLATARRAY_REGISTER_SOA(CELL_TYPE, CELL_MEMBERS)              \
    namespace LibFlatArray {                                            \
                                                                        \
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
        static const size_t INDEX =                                     \
            number_of_members<CELL_TYPE>::VALUE;                        \
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
        soa_accessor<CELL_TYPE, LIBFLATARRAY_PARAMS> operator[](        \
            coord<X, Y, Z>) const                                       \
        {                                                               \
            return soa_accessor<CELL_TYPE, LIBFLATARRAY_PARAMS>(        \
                data, index);                                           \
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
        template<typename MEMBER_TYPE, int OFFSET>                      \
        inline                                                          \
        __host__ __device__                                             \
        MEMBER_TYPE& access_member()                                    \
        {                                                               \
            return *(MEMBER_TYPE*)(                                     \
                data + (DIM_X * DIM_Y * DIM_Z) *                        \
                detail::flat_array::offset<CELL_TYPE, OFFSET>::OFFSET + \
                *index * sizeof(MEMBER_TYPE) +                          \
                INDEX  * sizeof(MEMBER_TYPE));                          \
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
        const char *get_data() const                                    \
        {                                                               \
            return data;                                                \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        char *get_data()                                                \
        {                                                               \
            return data;                                                \
        }                                                               \
                                                                        \
    private:                                                            \
        char *data;                                                     \
        int *index;                                                     \
    };                                                                  \
                                                                        \
    template<int MY_DIM_X, int MY_DIM_Y, int MY_DIM_Z, int INDEX>       \
    class const_soa_accessor<                                           \
        CELL_TYPE, MY_DIM_X, MY_DIM_Y, MY_DIM_Z, INDEX>                 \
    {                                                                   \
    public:                                                             \
        typedef CELL_TYPE MyCell;                                       \
                                                                        \
        static const int DIM_X = MY_DIM_X;                              \
        static const int DIM_Y = MY_DIM_Y;                              \
        static const int DIM_Z = MY_DIM_Z;                              \
                                                                        \
        __host__ __device__                                             \
        const_soa_accessor(const char *data=0, int *index=0) :          \
            data(data),                                                 \
            index(index)                                                \
        {}                                                              \
                                                                        \
        template<int X, int Y, int Z>                                   \
        inline                                                          \
        __host__ __device__                                             \
        const_soa_accessor<CELL_TYPE, LIBFLATARRAY_PARAMS> operator[](  \
            coord<X, Y, Z>) const                                       \
        {                                                               \
            return const_soa_accessor<CELL_TYPE, LIBFLATARRAY_PARAMS>(  \
                data, index);                                           \
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
        void save(char *target, size_t count) const                     \
        {                                                               \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                COPY_SOA_MEMBER_ARRAY_OUT,                              \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        BOOST_PP_SEQ_FOR_EACH(                                          \
            DECLARE_SOA_MEMBER_CONST,                                   \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
                                                                        \
        __host__ __device__                                             \
        const char *get_data() const                                    \
        {                                                               \
            return data;                                                \
        }                                                               \
                                                                        \
    private:                                                            \
        const char *data;                                               \
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
        const LibFlatArray::soa_accessor<                               \
            CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX> soa)                 \
    {                                                                   \
        soa >> cell;                                                    \
    }

#endif
