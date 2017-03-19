/**
 * Copyright 2014-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_MACROS_HPP
#define FLAT_ARRAY_MACROS_HPP

#include <libflatarray/coord.hpp>
#include <libflatarray/number_of_members.hpp>
#include <libflatarray/detail/macros.hpp>
#include <libflatarray/detail/offset.hpp>
#include <libflatarray/detail/sibling_short_vec_switch.hpp>

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
#define LIBFLATARRAY_ACCESS                                             \
    template<typename CELL_TYPE,                                        \
             long MY_DIM_X, long MY_DIM_Y, long MY_DIM_Z, long INDEX>   \
    friend class LibFlatArray::soa_accessor;                            \
                                                                        \
    template<typename CELL_TYPE,                                        \
             long MY_DIM_X, long MY_DIM_Y, long MY_DIM_Z, long INDEX>   \
    friend class LibFlatArray::const_soa_accessor;                      \
                                                                        \
    template<typename CELL_TYPE,                                        \
             long MY_DIM_X, long MY_DIM_Y, long MY_DIM_Z, long INDEX>   \
    friend class LibFlatArray::soa_accessor_light;                      \
                                                                        \
    template<typename CELL_TYPE,                                        \
             long MY_DIM_X, long MY_DIM_Y, long MY_DIM_Z, long INDEX>   \
    friend class LibFlatArray::const_soa_accessor_light;                \
                                                                        \
    template<typename CELL_TYPE, long R>                                \
    friend class LibFlatArray::detail::flat_array::offset;



#ifdef _MSC_BUILD
/**
 * This macros registers a type with LibFlatArray so that it can be
 * used with soa_grid, soa_array and friends. It will instantiate all
 * templates required for the "Struct of Arrays" (SoA) storage and
 * adds utilities so that user code can also discover properties of
 * the SoA layout.
 */
#  define LIBFLATARRAY_REGISTER_SOA(CELL_TYPE, CELL_MEMBERS)    \
    __pragma( warning( push ) )                                 \
    __pragma( warning( disable : 4307 4514 4626 ) )             \
    LIBFLATARRAY_REGISTER_SOA_MAIN(CELL_TYPE, CELL_MEMBERS)     \
    __pragma( warning( pop ) )                                  \

#else

/**
 * This macros registers a type with LibFlatArray so that it can be
 * used with soa_grid, soa_array and friends. It will instantiate all
 * templates required for the "Struct of Arrays" (SoA) storage and
 * adds utilities so that user code can also discover properties of
 * the SoA layout.
 */
#  define LIBFLATARRAY_REGISTER_SOA(CELL_TYPE, CELL_MEMBERS)    \
    LIBFLATARRAY_REGISTER_SOA_MAIN(CELL_TYPE, CELL_MEMBERS)     \

#endif

#define LIBFLATARRAY_REGISTER_SOA_MAIN(CELL_TYPE, CELL_MEMBERS)         \
    namespace LibFlatArray {                                            \
                                                                        \
    LIBFLATARRAY_FOR_EACH(                                              \
        LIBFLATARRAY_DEFINE_FIELD_OFFSET,                               \
        CELL_TYPE,                                                      \
        CELL_MEMBERS)                                                   \
                                                                        \
    template<>                                                          \
    class number_of_members<CELL_TYPE>                                  \
    {                                                                   \
    public:                                                             \
        static const std::size_t VALUE =                                \
            LIBFLATARRAY_SIZE(CELL_MEMBERS);                            \
    };                                                                  \
                                                                        \
    template<>                                                          \
    class aggregated_member_size<CELL_TYPE>                             \
    {                                                                   \
    private:                                                            \
        static const std::size_t INDEX =                                \
            number_of_members<CELL_TYPE>::VALUE;                        \
                                                                        \
    public:                                                             \
        static const std::size_t VALUE =                                \
            detail::flat_array::offset<CELL_TYPE, INDEX>::OFFSET;       \
    };                                                                  \
                                                                        \
    template<long MY_DIM_X, long MY_DIM_Y, long MY_DIM_Z, long INDEX>   \
    class soa_accessor<CELL_TYPE, MY_DIM_X, MY_DIM_Y, MY_DIM_Z, INDEX>  \
    {                                                                   \
    public:                                                             \
        typedef CELL_TYPE element_type;                                 \
                                                                        \
        static const long DIM_X = MY_DIM_X;                             \
        static const long DIM_Y = MY_DIM_Y;                             \
        static const long DIM_Z = MY_DIM_Z;                             \
        static const long DIM_PROD = DIM_X * DIM_Y * DIM_Z;             \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        static                                                          \
        long gen_index(const long x, const long y, const long z)        \
        {                                                               \
            return z * DIM_X * DIM_Y + y * DIM_X + x;                   \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        static                                                          \
        long gen_index(const std::size_t x, const std::size_t y, const std::size_t z) \
        {                                                               \
            return                                                      \
                static_cast<long>(z) * DIM_X * DIM_Y +                  \
                static_cast<long>(y) * DIM_X +                          \
                static_cast<long>(x);                                   \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        soa_accessor(char *my_data, const long my_index) :              \
            my_data(my_data),                                           \
            my_index(my_index)                                          \
        {}                                                              \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        operator CELL_TYPE() const                                      \
        {                                                               \
            CELL_TYPE ret;                                              \
            *this >> ret;                                               \
            return ret;                                                 \
        }                                                               \
                                                                        \
        template<typename SOA_ACCESSOR>                                 \
        inline                                                          \
        __host__ __device__                                             \
        bool operator==(const SOA_ACCESSOR& other) const                \
        {                                                               \
            return                                                      \
                (my_data == other.my_data) &&                           \
                (my_index == other.my_index);                           \
        }                                                               \
                                                                        \
        template<typename SOA_ACCESSOR>                                 \
        inline                                                          \
        __host__ __device__                                             \
        bool operator!=(const SOA_ACCESSOR& other) const                \
        {                                                               \
            return !(*this == other);                                   \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        void operator+=(const long offset)                              \
        {                                                               \
            my_index += offset;                                         \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        void operator++()                                               \
        {                                                               \
            ++my_index;                                                 \
        }                                                               \
                                                                        \
        template<long X, long Y, long Z>                                \
        inline                                                          \
        __host__ __device__                                             \
        soa_accessor_light<CELL_TYPE, LIBFLATARRAY_PARAMS> operator[](  \
            coord<X, Y, Z>)                                             \
        {                                                               \
            return soa_accessor_light<CELL_TYPE, LIBFLATARRAY_PARAMS>(  \
                my_data, my_index);                                     \
        }                                                               \
                                                                        \
        template<long X, long Y, long Z>                                \
        inline                                                          \
        __host__ __device__                                             \
        const_soa_accessor_light<CELL_TYPE, LIBFLATARRAY_PARAMS> operator[]( \
            coord<X, Y, Z>) const                                       \
        {                                                               \
            return const_soa_accessor_light<CELL_TYPE, LIBFLATARRAY_PARAMS>( \
                my_data, my_index);                                     \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void operator=(const CELL_TYPE& cell)                           \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_IN,                \
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
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_OUT,               \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void load(const char *source, std::size_t count)                \
        {                                                               \
            load(source, count, 0, count);                              \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void load(                                                      \
            const char *source,                                         \
            std::size_t count,                                          \
            std::size_t offset,                                         \
            std::size_t stride)                                         \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_ARRAY_IN,          \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void save(char *target, std::size_t count) const                \
        {                                                               \
            save(target, count, 0, count);                              \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void save(                                                      \
            char *target,                                               \
            std::size_t count,                                          \
            std::size_t offset,                                         \
            std::size_t stride) const                                   \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_ARRAY_OUT,         \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void construct_members()                                        \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_INIT_SOA_GENERIC_MEMBER,                   \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void destroy_members()                                          \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_DESTROY_SOA_GENERIC_MEMBER,                \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        template<typename OTHER_ACCESSOR>                               \
        __host__ __device__                                             \
        inline                                                          \
        void copy_members(const OTHER_ACCESSOR& other, std::size_t count) \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER,                   \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        template<typename MEMBER_TYPE, long OFFSET>                     \
        inline                                                          \
        __host__ __device__                                             \
        MEMBER_TYPE& access_member()                                    \
        {                                                               \
            return *(MEMBER_TYPE*)(                                     \
                my_data +                                               \
                DIM_PROD *                                              \
                detail::flat_array::offset<CELL_TYPE, OFFSET>::OFFSET + \
                my_index * sizeof(MEMBER_TYPE) +                        \
                INDEX * sizeof(MEMBER_TYPE));                           \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        char *access_member(const long size_of_member, const long offset) \
        {                                                               \
            return                                                      \
                my_data +                                               \
                DIM_PROD * offset +                                     \
                my_index * size_of_member +                             \
                INDEX * size_of_member;                                 \
        }                                                               \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_DECLARE_SOA_MEMBER_NORMAL,                     \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_DECLARE_SOA_MEMBER_CONST,                      \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
                                                                        \
        __host__ __device__                                             \
        const char *data() const                                        \
        {                                                               \
            return my_data;                                             \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        char *data()                                                    \
        {                                                               \
            return my_data;                                             \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        long& index()                                                   \
        {                                                               \
            return my_index;                                            \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        const long& index() const                                       \
        {                                                               \
            return my_index;                                            \
        }                                                               \
                                                                        \
    private:                                                            \
        char *my_data;                                                  \
        long my_index;                                                  \
    };                                                                  \
                                                                        \
    template<long MY_DIM_X, long MY_DIM_Y, long MY_DIM_Z, long INDEX>   \
    class const_soa_accessor<                                           \
        CELL_TYPE, MY_DIM_X, MY_DIM_Y, MY_DIM_Z, INDEX>                 \
    {                                                                   \
    public:                                                             \
        typedef CELL_TYPE element_type;                                 \
                                                                        \
        static const long DIM_X = MY_DIM_X;                             \
        static const long DIM_Y = MY_DIM_Y;                             \
        static const long DIM_Z = MY_DIM_Z;                             \
        static const long DIM_PROD = DIM_X * DIM_Y * DIM_Z;             \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        static                                                          \
        long gen_index(const long x, const long y, const long z)        \
        {                                                               \
            return z * DIM_X * DIM_Y + y * DIM_X + x;                   \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        static                                                          \
        long gen_index(const std::size_t x, const std::size_t y, const std::size_t z) \
        {                                                               \
            return                                                      \
                static_cast<long>(z) * DIM_X * DIM_Y +                  \
                static_cast<long>(y) * DIM_X +                          \
                static_cast<long>(x);                                   \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        const_soa_accessor(const char *my_data, long my_index) :        \
            my_data(my_data),                                           \
            my_index(my_index)                                          \
        {}                                                              \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        operator CELL_TYPE() const                                      \
        {                                                               \
            CELL_TYPE ret;                                              \
            *this >> ret;                                               \
            return ret;                                                 \
        }                                                               \
                                                                        \
        template<typename SOA_ACCESSOR>                                 \
        inline                                                          \
        __host__ __device__                                             \
        bool operator==(const SOA_ACCESSOR& other) const                \
        {                                                               \
            return                                                      \
                (my_data == other.my_data) &&                           \
                (my_index == other.my_index);                           \
        }                                                               \
                                                                        \
        template<typename SOA_ACCESSOR>                                 \
        inline                                                          \
        __host__ __device__                                             \
        bool operator!=(const SOA_ACCESSOR& other) const                \
        {                                                               \
            return !(*this == other);                                   \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        void operator+=(const long offset)                              \
        {                                                               \
            my_index += offset;                                         \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        void operator++()                                               \
        {                                                               \
            ++my_index;                                                 \
        }                                                               \
                                                                        \
        template<long X, long Y, long Z>                                \
        inline                                                          \
        __host__ __device__                                             \
        const_soa_accessor<CELL_TYPE, LIBFLATARRAY_PARAMS> operator[](  \
            coord<X, Y, Z>) const                                       \
        {                                                               \
            return const_soa_accessor<CELL_TYPE, LIBFLATARRAY_PARAMS>(  \
                my_data, my_index);                                     \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void operator>>(CELL_TYPE& cell) const                          \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_OUT,               \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void save(char *target, std::size_t count) const                \
        {                                                               \
            save(target, count, 0, count);                              \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void save(                                                      \
            char *target,                                               \
            std::size_t count,                                          \
            std::size_t offset,                                         \
            std::size_t stride) const                                   \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_ARRAY_OUT,         \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_DECLARE_SOA_MEMBER_CONST,                      \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
                                                                        \
        __host__ __device__                                             \
        const char *data() const                                        \
        {                                                               \
            return my_data;                                             \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        long& index()                                                   \
        {                                                               \
            return my_index;                                            \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        const long& index() const                                       \
        {                                                               \
            return my_index;                                            \
        }                                                               \
                                                                        \
    private:                                                            \
        const char *my_data;                                            \
        long my_index;                                                  \
    };                                                                  \
                                                                        \
    template<long MY_DIM_X, long MY_DIM_Y, long MY_DIM_Z, long INDEX>   \
    class soa_accessor_light<CELL_TYPE, MY_DIM_X, MY_DIM_Y, MY_DIM_Z, INDEX> \
    {                                                                   \
    public:                                                             \
        typedef CELL_TYPE element_type;                                 \
                                                                        \
        static const long DIM_X = MY_DIM_X;                             \
        static const long DIM_Y = MY_DIM_Y;                             \
        static const long DIM_Z = MY_DIM_Z;                             \
        static const long DIM_PROD = DIM_X * DIM_Y * DIM_Z;             \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        static                                                          \
        long gen_index(const long x, const long y, const long z)        \
        {                                                               \
            return z * DIM_X * DIM_Y + y * DIM_X + x;                   \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        static                                                          \
        long gen_index(const std::size_t x, const std::size_t y, const std::size_t z) \
        {                                                               \
            return                                                      \
                static_cast<long>(z) * DIM_X * DIM_Y +                  \
                static_cast<long>(y) * DIM_X +                          \
                static_cast<long>(x);                                   \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        soa_accessor_light(char *my_data, long& my_index) :             \
            my_data(my_data),                                           \
            my_index(&my_index)                                         \
        {}                                                              \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        operator CELL_TYPE() const                                      \
        {                                                               \
            CELL_TYPE ret;                                              \
            *this >> ret;                                               \
            return ret;                                                 \
        }                                                               \
                                                                        \
        template<typename SOA_ACCESSOR>                                 \
        inline                                                          \
        __host__ __device__                                             \
        bool operator==(const SOA_ACCESSOR& other) const                \
        {                                                               \
            return                                                      \
                (my_data == other.my_data) &&                           \
                (my_index == other.my_index);                           \
        }                                                               \
                                                                        \
        template<typename SOA_ACCESSOR>                                 \
        inline                                                          \
        __host__ __device__                                             \
        bool operator!=(const SOA_ACCESSOR& other) const                \
        {                                                               \
            return !(*this == other);                                   \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
            void operator+=(const long offset)                          \
        {                                                               \
            *my_index += offset;                                        \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        void operator++()                                               \
        {                                                               \
            ++*my_index;                                                \
        }                                                               \
                                                                        \
        template<long X, long Y, long Z>                                \
        inline                                                          \
        __host__ __device__                                             \
        soa_accessor_light<CELL_TYPE, LIBFLATARRAY_PARAMS> operator[](  \
            coord<X, Y, Z>) const                                       \
        {                                                               \
            return soa_accessor_light<CELL_TYPE, LIBFLATARRAY_PARAMS>(  \
                my_data, *my_index);                                    \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void operator=(const CELL_TYPE& cell)                           \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_IN,                \
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
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_OUT,               \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void load(const char *source, std::size_t count)                \
        {                                                               \
            load(source, count, 0, count);                              \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void load(                                                      \
            const char *source,                                         \
            std::size_t count,                                          \
            std::size_t offset,                                         \
            std::size_t stride)                                         \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_ARRAY_IN,          \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void save(char *target, std::size_t count) const                \
        {                                                               \
            save(target, count, 0, count);                              \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void save(                                                      \
            char *target,                                               \
            std::size_t count,                                          \
            std::size_t offset,                                         \
            std::size_t stride) const                                   \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_ARRAY_OUT,         \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void construct_members()                                        \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_INIT_SOA_GENERIC_MEMBER,                   \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void destroy_members()                                          \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_DESTROY_SOA_GENERIC_MEMBER,                \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        template<typename OTHER_ACCESSOR>                               \
        __host__ __device__                                             \
        inline                                                          \
        void copy_members(const OTHER_ACCESSOR& other, std::size_t count) \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER,                   \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        template<typename MEMBER_TYPE, long OFFSET>                     \
        inline                                                          \
        __host__ __device__                                             \
        MEMBER_TYPE& access_member()                                    \
        {                                                               \
            return *(MEMBER_TYPE*)(                                     \
                my_data +                                               \
                DIM_PROD *                                              \
                detail::flat_array::offset<CELL_TYPE, OFFSET>::OFFSET + \
                *my_index * sizeof(MEMBER_TYPE) +                       \
                INDEX  * sizeof(MEMBER_TYPE));                          \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        char *access_member(const long size_of_member, const long offset) \
        {                                                               \
            return                                                      \
                my_data +                                               \
                DIM_PROD * offset +                                     \
                *my_index * size_of_member +                            \
                INDEX  * size_of_member;                                \
        }                                                               \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_DECLARE_SOA_MEMBER_LIGHT_NORMAL,               \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_DECLARE_SOA_MEMBER_LIGHT_CONST,                \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
                                                                        \
        __host__ __device__                                             \
        const char *data() const                                        \
        {                                                               \
            return my_data;                                             \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        char *data()                                                    \
        {                                                               \
            return my_data;                                             \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        long& index()                                                   \
        {                                                               \
            return *my_index;                                           \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        const long& index() const                                       \
        {                                                               \
            return *my_index;                                           \
        }                                                               \
                                                                        \
    private:                                                            \
        char *my_data;                                                  \
        long *my_index;                                                 \
    };                                                                  \
                                                                        \
    template<long MY_DIM_X, long MY_DIM_Y, long MY_DIM_Z, long INDEX>   \
    class const_soa_accessor_light<CELL_TYPE, MY_DIM_X, MY_DIM_Y, MY_DIM_Z, INDEX> \
    {                                                                   \
    public:                                                             \
        typedef CELL_TYPE element_type;                                 \
                                                                        \
        static const long DIM_X = MY_DIM_X;                             \
        static const long DIM_Y = MY_DIM_Y;                             \
        static const long DIM_Z = MY_DIM_Z;                             \
        static const long DIM_PROD = DIM_X * DIM_Y * DIM_Z;             \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        static                                                          \
        long gen_index(const long x, const long y, const long z)        \
        {                                                               \
            return z * DIM_X * DIM_Y + y * DIM_X + x;                   \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        static                                                          \
        long gen_index(const std::size_t x, const std::size_t y, const std::size_t z) \
        {                                                               \
            return                                                      \
                static_cast<long>(z) * DIM_X * DIM_Y +                  \
                static_cast<long>(y) * DIM_X +                          \
                static_cast<long>(x);                                   \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        const_soa_accessor_light(const char *my_data, long& my_index) : \
            my_data(my_data),                                           \
            my_index(&my_index)                                         \
        {}                                                              \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        operator CELL_TYPE() const                                      \
        {                                                               \
            CELL_TYPE ret;                                              \
            *this >> ret;                                               \
            return ret;                                                 \
        }                                                               \
                                                                        \
        template<typename SOA_ACCESSOR>                                 \
        inline                                                          \
        __host__ __device__                                             \
        bool operator==(const SOA_ACCESSOR& other) const                \
        {                                                               \
            return                                                      \
                (my_data == other.my_data) &&                           \
                (my_index == other.my_index);                           \
        }                                                               \
                                                                        \
        template<typename SOA_ACCESSOR>                                 \
        inline                                                          \
        __host__ __device__                                             \
        bool operator!=(const SOA_ACCESSOR& other) const                \
        {                                                               \
            return !(*this == other);                                   \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        void operator+=(const long offset)                              \
        {                                                               \
            *my_index += offset;                                        \
        }                                                               \
                                                                        \
        inline                                                          \
        __host__ __device__                                             \
        void operator++()                                               \
        {                                                               \
            ++*my_index;                                                \
        }                                                               \
                                                                        \
        template<long X, long Y, long Z>                                \
        inline                                                          \
        __host__ __device__                                             \
        const_soa_accessor_light<CELL_TYPE, LIBFLATARRAY_PARAMS> operator[]( \
            coord<X, Y, Z>) const                                       \
        {                                                               \
            return const_soa_accessor_light<CELL_TYPE, LIBFLATARRAY_PARAMS>( \
                my_data, *my_index);                                    \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void operator>>(CELL_TYPE& cell) const                          \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_OUT,               \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void save(char *target, std::size_t count) const                \
        {                                                               \
            save(target, count, 0, count);                              \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        inline                                                          \
        void save(                                                      \
            char *target,                                               \
            std::size_t count,                                          \
            std::size_t offset,                                         \
            std::size_t stride) const                                   \
        {                                                               \
            LIBFLATARRAY_FOR_EACH(                                      \
                LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_ARRAY_OUT,         \
                CELL_TYPE,                                              \
                CELL_MEMBERS);                                          \
        }                                                               \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_DECLARE_SOA_MEMBER_LIGHT_CONST,                \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
                                                                        \
        __host__ __device__                                             \
        const char *data() const                                        \
        {                                                               \
            return my_data;                                             \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        long& index()                                                   \
        {                                                               \
            return *my_index;                                           \
        }                                                               \
                                                                        \
        __host__ __device__                                             \
        const long& index() const                                       \
        {                                                               \
            return *my_index;                                           \
        }                                                               \
                                                                        \
    private:                                                            \
        const char *my_data;                                            \
        long *my_index;                                                 \
    };                                                                  \
                                                                        \
    }                                                                   \
                                                                        \
    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>            \
    __host__ __device__                                                 \
    inline                                                              \
    void operator<<(                                                    \
        CELL_TYPE& cell,                                                \
        const LibFlatArray::soa_accessor<                               \
            CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX> soa)                 \
    {                                                                   \
        soa >> cell;                                                    \
    }

#define LIBFLATARRAY_CUSTOM_SIZES(X_SIZES, Y_SIZES, Z_SIZES)            \
    typedef void has_sizes;                                             \
                                                                        \
    template<typename CELL, typename FUNCTOR>                           \
    void select_size(                                                   \
        char *data,                                                     \
        FUNCTOR& functor,                                               \
        const std::size_t dim_x = 1,                                    \
        const std::size_t dim_y = 1,                                    \
        const std::size_t dim_z = 1)                                    \
    {                                                                   \
        bind_size_x<CELL>(dim_x, dim_y, dim_z, data, functor);          \
    }                                                                   \
                                                                        \
    template<typename CELL, typename FUNCTOR>                           \
    void select_size(                                                   \
        const char *data,                                               \
        FUNCTOR& functor,                                               \
        const std::size_t dim_x = 1,                                    \
        const std::size_t dim_y = 1,                                    \
        const std::size_t dim_z = 1)                                    \
    {                                                                   \
        bind_size_x<CELL>(dim_x, dim_y, dim_z, data, functor);          \
    }                                                                   \
                                                                        \
    template<typename CELL, typename FUNCTOR>                           \
    void bind_size_x(                                                   \
        const std::size_t dim_x,                                        \
        const std::size_t dim_y,                                        \
        const std::size_t dim_z,                                        \
        char *data,                                                     \
        FUNCTOR& functor) const                                         \
    {                                                                   \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_X,                                    \
            unused,                                                     \
            X_SIZES);                                                   \
                                                                        \
        throw std::out_of_range("grid dimension X too large");          \
    }                                                                   \
                                                                        \
    template<typename CELL, typename FUNCTOR>                           \
    void bind_size_x(                                                   \
        const std::size_t dim_x,                                        \
        const std::size_t dim_y,                                        \
        const std::size_t dim_z,                                        \
        const char *data,                                               \
        FUNCTOR& functor) const                                         \
    {                                                                   \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_X,                                    \
            unused,                                                     \
            X_SIZES);                                                   \
                                                                        \
        throw std::out_of_range("grid dimension X too large");          \
    }                                                                   \
                                                                        \
    template<typename CELL, long DIM_X, typename FUNCTOR>               \
    void bind_size_y(                                                   \
        const std::size_t dim_y,                                        \
        const std::size_t dim_z,                                        \
        char *data,                                                     \
        FUNCTOR& functor) const                                         \
    {                                                                   \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_Y,                                    \
            unused,                                                     \
            Y_SIZES);                                                   \
                                                                        \
        throw std::out_of_range("grid dimension Y too large");          \
    }                                                                   \
                                                                        \
    template<typename CELL, long DIM_X, typename FUNCTOR>               \
    void bind_size_y(                                                   \
        const std::size_t dim_y,                                        \
        const std::size_t dim_z,                                        \
        const char *data,                                               \
        FUNCTOR& functor) const                                         \
    {                                                                   \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_Y,                                    \
            unused,                                                     \
            Y_SIZES);                                                   \
                                                                        \
        throw std::out_of_range("grid dimension Y too large");          \
    }                                                                   \
                                                                        \
    template<typename CELL, long DIM_X, long DIM_Y, typename FUNCTOR>   \
    void bind_size_z(                                                   \
        const std::size_t dim_z,                                        \
        char *data,                                                     \
        FUNCTOR& functor) const                                         \
    {                                                                   \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_Z,                                    \
            unused,                                                     \
            Z_SIZES);                                                   \
                                                                        \
        throw std::out_of_range("grid dimension Z too large");          \
    }                                                                   \
                                                                        \
    template<typename CELL, long DIM_X, long DIM_Y, typename FUNCTOR>   \
    void bind_size_z(                                                   \
        const std::size_t dim_z,                                        \
        const char *data,                                               \
        FUNCTOR& functor) const                                         \
    {                                                                   \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_Z,                                    \
            unused,                                                     \
            Z_SIZES);                                                   \
                                                                        \
        throw std::out_of_range("grid dimension Z too large");          \
    }

#define LIBFLATARRAY_CUSTOM_SIZES_1D_UNIFORM(SIZES)                     \
    typedef void has_sizes;                                             \
                                                                        \
    template<typename CELL, typename FUNCTOR>                           \
    void select_size(                                                   \
        char *data,                                                     \
        FUNCTOR& functor,                                               \
        const std::size_t dim_x = 1,                                    \
        const std::size_t dim_y = 1,                                    \
        const std::size_t dim_z = 1)                                    \
    {                                                                   \
        if (dim_y != 1) {                                               \
            throw std::out_of_range("expected 1D grid, but y != 1");    \
        }                                                               \
        if (dim_z != 1) {                                               \
            throw std::out_of_range("expected 1D grid, but z != 1");    \
        }                                                               \
        std::size_t maxDim = dim_x;                                     \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_MAX_1D,                               \
            unused,                                                     \
            SIZES);                                                     \
                                                                        \
        throw std::out_of_range("max grid dimension too large");        \
    }                                                                   \
                                                                        \
    template<typename CELL, typename FUNCTOR>                           \
    void select_size(                                                   \
        const char *data,                                               \
        FUNCTOR& functor,                                               \
        const std::size_t dim_x = 1,                                    \
        const std::size_t dim_y = 1,                                    \
        const std::size_t dim_z = 1)                                    \
    {                                                                   \
        if (dim_y != 1) {                                               \
            throw std::out_of_range("expected 1D grid, but y != 1");    \
        }                                                               \
        if (dim_z != 1) {                                               \
            throw std::out_of_range("expected 1D grid, but z != 1");    \
        }                                                               \
        using std::max;                                                 \
        std::size_t maxDim = max(dim_x, dim_z);                         \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_MAX_1D,                               \
            unused,                                                     \
            SIZES);                                                     \
                                                                        \
        throw std::out_of_range("max grid dimension too large");        \
    }

#define LIBFLATARRAY_CUSTOM_SIZES_2D_UNIFORM(SIZES)                     \
    typedef void has_sizes;                                             \
                                                                        \
    template<typename CELL, typename FUNCTOR>                           \
    void select_size(                                                   \
        char *data,                                                     \
        FUNCTOR& functor,                                               \
        const std::size_t dim_x = 1,                                    \
        const std::size_t dim_y = 1,                                    \
        const std::size_t dim_z = 1)                                    \
    {                                                                   \
        if (dim_z != 1) {                                               \
            throw std::out_of_range("expected 2D grid, but z != 1");    \
        }                                                               \
        using std::max;                                                 \
        std::size_t maxDim = max(dim_x, dim_y);                         \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_MAX_2D,                               \
            unused,                                                     \
            SIZES);                                                     \
                                                                        \
        throw std::out_of_range("max grid dimension too large");        \
    }                                                                   \
                                                                        \
    template<typename CELL, typename FUNCTOR>                           \
    void select_size(                                                   \
        const char *data,                                               \
        FUNCTOR& functor,                                               \
        const std::size_t dim_x = 1,                                    \
        const std::size_t dim_y = 1,                                    \
        const std::size_t dim_z = 1)                                    \
    {                                                                   \
        if (dim_z != 1) {                                               \
            throw std::out_of_range("expected 2D grid, but z != 1");    \
        }                                                               \
        using std::max;                                                 \
        std::size_t maxDim = max(dim_x, dim_y);                         \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_MAX_2D,                               \
            unused,                                                     \
            SIZES);                                                     \
                                                                        \
        throw std::out_of_range("max grid dimension too large");        \
    }

#define LIBFLATARRAY_CUSTOM_SIZES_3D_UNIFORM(SIZES)                     \
    typedef void has_sizes;                                             \
                                                                        \
    template<typename CELL, typename FUNCTOR>                           \
    void select_size(                                                   \
        char *data,                                                     \
        FUNCTOR& functor,                                               \
        const std::size_t dim_x = 1,                                    \
        const std::size_t dim_y = 1,                                    \
        const std::size_t dim_z = 1)                                    \
    {                                                                   \
        using std::max;                                                 \
        std::size_t maxDim = max(dim_x, max(dim_y, dim_z));             \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_MAX_3D,                               \
            unused,                                                     \
            SIZES);                                                     \
                                                                        \
        throw std::out_of_range("max grid dimension too large");        \
    }                                                                   \
                                                                        \
    template<typename CELL, typename FUNCTOR>                           \
    void select_size(                                                   \
        const char *data,                                               \
        FUNCTOR& functor,                                               \
        const std::size_t dim_x = 1,                                    \
        const std::size_t dim_y = 1,                                    \
        const std::size_t dim_z = 1)                                    \
    {                                                                   \
        using std::max;                                                 \
        std::size_t maxDim = max(dim_x, max(dim_y, dim_z));             \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            LIBFLATARRAY_CASE_DIM_MAX_3D,                               \
            unused,                                                     \
            SIZES);                                                     \
                                                                        \
        throw std::out_of_range("max grid dimension too large");        \
    }

#endif
