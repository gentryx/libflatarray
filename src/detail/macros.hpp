/**
 * Copyright 2014-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_MACROS_HPP
#define FLAT_ARRAY_DETAIL_MACROS_HPP

#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/comparison/less.hpp>
#include <boost/preprocessor/if.hpp>
#include <boost/preprocessor/seq.hpp>
#include <libflatarray/detail/generic_destruct.hpp>
#include <libflatarray/detail/soa_array_member_copy_helper.hpp>

#define LIBFLATARRAY_INDEX(X, Y, Z, DIM_X, DIM_Y, DIM_Z, INDEX) \
    (INDEX + Z * (DIM_X * DIM_Y) + Y * DIM_X + X)

#define LIBFLATARRAY_PARAMS_FULL(X, Y, Z, DIM_X, DIM_Y, DIM_Z, INDEX)	\
    DIM_X, DIM_Y, DIM_Z, LIBFLATARRAY_INDEX(X, Y, Z, DIM_X, DIM_Y, DIM_Z, INDEX)

// expands to A if MEMBER is scalar (e.g. double foo),
// expands to B if MEMBER is an array member (e.g. double foo[4]).
#define LIBFLATARRAY_ARRAY_CONDITIONAL(MEMBER, A, B)    \
    BOOST_PP_IF(BOOST_PP_LESS(BOOST_PP_SEQ_SIZE(MEMBER), 3), A, B)

#define LIBFLATARRAY_ARRAY_ARITY(MEMBER)                                \
    BOOST_PP_SEQ_ELEM(BOOST_PP_SUB(BOOST_PP_SEQ_SIZE(MEMBER), 1), MEMBER)

#define LIBFLATARRAY_DEFINE_FIELD_OFFSET(r, CELL_TYPE, MEMBER)          \
    namespace detail {                                                  \
    namespace flat_array {                                              \
    template<>                                                          \
    class offset<CELL_TYPE, r - 1>                                      \
    {                                                                   \
    public:                                                             \
        static const std::size_t OFFSET =                               \
            offset<CELL_TYPE, r - 2>::OFFSET +                          \
            sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)) *                      \
            LIBFLATARRAY_ARRAY_CONDITIONAL(                             \
                MEMBER,                                                 \
                1,                                                      \
                LIBFLATARRAY_ARRAY_ARITY(MEMBER));                      \
                                                                        \
        template<typename MEMBER_TYPE>                                  \
        inline                                                          \
        int operator()(MEMBER_TYPE CELL_TYPE:: *member_ptr)             \
        {                                                               \
            return offset<CELL_TYPE, r - 2>()(member_ptr);              \
        }                                                               \
                                                                        \
        inline                                                          \
        int operator()(                                                 \
            BOOST_PP_SEQ_ELEM(0, MEMBER) (CELL_TYPE:: *member_ptr)      \
            LIBFLATARRAY_ARRAY_CONDITIONAL(                             \
                MEMBER,                                                 \
                ,                                                       \
                [LIBFLATARRAY_ARRAY_ARITY(MEMBER)]))                    \
        {                                                               \
            if (member_ptr ==                                           \
                &CELL_TYPE::BOOST_PP_SEQ_ELEM(1, MEMBER)) {             \
                return offset<CELL_TYPE, r - 2>::OFFSET;                \
            } else {                                                    \
                return offset<CELL_TYPE, r - 2>()(member_ptr);          \
            }                                                           \
        }                                                               \
                                                                        \
        template<int ARITY>                                             \
        inline                                                          \
        int operator()(BOOST_PP_SEQ_ELEM(0, MEMBER) (CELL_TYPE:: *member_ptr)[ARITY]) \
        {                                                               \
            return offset<CELL_TYPE, r - 2>()(member_ptr);              \
        }                                                               \
    };                                                                  \
    }                                                                   \
    }

#define LIBFLATARRAY_DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, CONST, INDEX_VAR) \
    LIBFLATARRAY_ARRAY_CONDITIONAL(MEMBER, , template<int ARRAY_INDEX >) \
    inline                                                              \
    __host__ __device__                                                 \
    CONST BOOST_PP_SEQ_ELEM(0, MEMBER)&                                 \
        BOOST_PP_SEQ_ELEM(1, MEMBER)() CONST                            \
    {                                                                   \
        return *(BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                        \
            data +                                                      \
            (DIM_PROD) * (                                              \
                (sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)) *                 \
                 LIBFLATARRAY_ARRAY_CONDITIONAL(MEMBER, 0, ARRAY_INDEX))  + \
                detail::flat_array::offset<CELL, MEMBER_INDEX - 2>:: OFFSET) + \
            INDEX_VAR * long(sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER))) +    \
            INDEX     * long(sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER))));    \
    }                                                                   \
                                                                        \
    LIBFLATARRAY_ARRAY_CONDITIONAL(                                     \
        MEMBER,                                                         \
        ,                                                               \
        inline                                                          \
        __host__ __device__                                             \
        typename LibFlatArray::detail::                                 \
        soa_array_member_copy_helper<DIM_PROD>::                        \
        template inner1<CELL>::                                         \
        template inner2<BOOST_PP_SEQ_ELEM(0, MEMBER)>::                 \
        reference BOOST_PP_SEQ_ELEM(1, MEMBER)() CONST                  \
        {                                                               \
            return typename LibFlatArray::detail::                      \
                soa_array_member_copy_helper<DIM_PROD>::                \
                template inner1<CELL>::                                 \
                template inner2<BOOST_PP_SEQ_ELEM(0, MEMBER)>::         \
                reference((char*)&BOOST_PP_SEQ_ELEM(1, MEMBER)<0>());   \
        } )


#define LIBFLATARRAY_DECLARE_SOA_MEMBER_CONST(MEMBER_INDEX, CELL, MEMBER) \
    LIBFLATARRAY_DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, const, index)

#define LIBFLATARRAY_DECLARE_SOA_MEMBER_NORMAL(MEMBER_INDEX, CELL, MEMBER) \
    LIBFLATARRAY_DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER,      , index)

#define LIBFLATARRAY_DECLARE_SOA_MEMBER_LIGHT_CONST(MEMBER_INDEX, CELL, MEMBER) \
    LIBFLATARRAY_DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, const, *index)

#define LIBFLATARRAY_DECLARE_SOA_MEMBER_LIGHT_NORMAL(MEMBER_INDEX, CELL, MEMBER) \
    LIBFLATARRAY_DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER,      , *index)

#define LIBFLATARRAY_COPY_SOA_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER)     \
    BOOST_PP_SEQ_ELEM(1, MEMBER)() = cell.BOOST_PP_SEQ_ELEM(1, MEMBER);

#define LIBFLATARRAY_COPY_SOA_ARRAY_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER) \
    {                                                                   \
        typename LibFlatArray::detail::soa_array_member_copy_helper<DIM_PROD>:: \
            template inner1<CELL>::                                     \
            template inner2<BOOST_PP_SEQ_ELEM(0, MEMBER)>::             \
            template inner3<LIBFLATARRAY_ARRAY_ARITY(MEMBER)>::         \
            template inner4<&CELL::BOOST_PP_SEQ_ELEM(1, MEMBER)>::      \
            template copy_in<LIBFLATARRAY_ARRAY_ARITY(MEMBER)>()(       \
                cell,                                                   \
                &BOOST_PP_SEQ_ELEM(1, MEMBER)<0>());                    \
    }

#define LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER) \
    LIBFLATARRAY_ARRAY_CONDITIONAL(                                     \
        MEMBER,                                                         \
        LIBFLATARRAY_COPY_SOA_MEMBER_IN(      MEMBER_INDEX, CELL, MEMBER), \
        LIBFLATARRAY_COPY_SOA_ARRAY_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER))

#define LIBFLATARRAY_COPY_SOA_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER)    \
    cell.BOOST_PP_SEQ_ELEM(1, MEMBER) = this->BOOST_PP_SEQ_ELEM(1, MEMBER)();

#define LIBFLATARRAY_COPY_SOA_ARRAY_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER) \
    {                                                                   \
        typename LibFlatArray::detail::                                 \
            soa_array_member_copy_helper<DIM_PROD>::                    \
            template inner1<CELL>::                                     \
            template inner2<BOOST_PP_SEQ_ELEM(0, MEMBER)>::             \
            template inner3<LIBFLATARRAY_ARRAY_ARITY(MEMBER)>::         \
            template inner4<&CELL::BOOST_PP_SEQ_ELEM(1, MEMBER)>::      \
            template copy_out<LIBFLATARRAY_ARRAY_ARITY(MEMBER)>()(      \
                cell,                                                   \
                &BOOST_PP_SEQ_ELEM(1, MEMBER)<0>());                    \
    }

#define LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER) \
    LIBFLATARRAY_ARRAY_CONDITIONAL(                                     \
        MEMBER,                                                         \
        LIBFLATARRAY_COPY_SOA_MEMBER_OUT(      MEMBER_INDEX, CELL, MEMBER), \
        LIBFLATARRAY_COPY_SOA_ARRAY_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER))

#define LIBFLATARRAY_COPY_SOA_MEMBER_ARRAY_IN(MEMBER_INDEX, CELL, MEMBER) \
    {                                                                   \
        for (int i = offset; i < (offset + count); ++i) {               \
            (&this->BOOST_PP_SEQ_ELEM(1, MEMBER)())[i] =                \
            ((const BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                     \
                source +                                                \
                detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET * \
                stride))[i];                                            \
        }                                                               \
    }

#define LIBFLATARRAY_COPY_SOA_ARRAY_MEMBER_ARRAY_IN(MEMBER_INDEX, CELL, MEMBER) \
    {                                                                   \
        typename LibFlatArray::detail::                                 \
            soa_array_member_copy_helper<DIM_PROD>::                    \
            template inner_a<BOOST_PP_SEQ_ELEM(0, MEMBER)>::            \
            template copy_array_in<LIBFLATARRAY_ARRAY_ARITY(MEMBER)>()( \
                (const BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                  \
                    source +                                            \
                    detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET * \
                    stride),                                            \
                &(this->BOOST_PP_SEQ_ELEM(1, MEMBER)()[0]),             \
                count,                                                  \
                offset,                                                 \
                stride);                                                \
    }

#define LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_ARRAY_IN(MEMBER_INDEX, CELL, MEMBER) \
    LIBFLATARRAY_ARRAY_CONDITIONAL(                                     \
        MEMBER,                                                         \
        LIBFLATARRAY_COPY_SOA_MEMBER_ARRAY_IN(      MEMBER_INDEX, CELL, MEMBER), \
        LIBFLATARRAY_COPY_SOA_ARRAY_MEMBER_ARRAY_IN(MEMBER_INDEX, CELL, MEMBER))


#define LIBFLATARRAY_COPY_SOA_MEMBER_ARRAY_OUT(MEMBER_INDEX, CELL, MEMBER) \
    {                                                                   \
        for (int i = offset; i < (offset + count); ++i) {               \
            ((BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                           \
                target +                                                \
                detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET * \
                stride))[i] =                                           \
            (&this->BOOST_PP_SEQ_ELEM(1, MEMBER)())[i];                 \
        }                                                               \
    }

#define LIBFLATARRAY_COPY_SOA_ARRAY_MEMBER_ARRAY_OUT(MEMBER_INDEX, CELL, MEMBER) \
    {                                                                   \
        typename LibFlatArray::detail::                                 \
            soa_array_member_copy_helper<DIM_PROD>::                    \
            template inner_a<BOOST_PP_SEQ_ELEM(0, MEMBER)>::            \
            template copy_array_out<LIBFLATARRAY_ARRAY_ARITY(MEMBER)>()( \
                (BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                        \
                    target +                                            \
                    detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET * \
                    stride),                                            \
                &(this->BOOST_PP_SEQ_ELEM(1, MEMBER)()[0]),             \
                count,                                                  \
                offset,                                                 \
                stride);                                                \
    }

#define LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER_ARRAY_OUT(MEMBER_INDEX, CELL, MEMBER) \
    LIBFLATARRAY_ARRAY_CONDITIONAL(                                     \
        MEMBER,                                                         \
        LIBFLATARRAY_COPY_SOA_MEMBER_ARRAY_OUT(      MEMBER_INDEX, CELL, MEMBER), \
        LIBFLATARRAY_COPY_SOA_ARRAY_MEMBER_ARRAY_OUT(MEMBER_INDEX, CELL, MEMBER))

#define LIBFLATARRAY_COPY_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER)        \
    {                                                                   \
        std::copy(                                                      \
            &(other.BOOST_PP_SEQ_ELEM(1, MEMBER)()),                    \
            &(other.BOOST_PP_SEQ_ELEM(1, MEMBER)()) + count,            \
            &(this->BOOST_PP_SEQ_ELEM(1, MEMBER)()));                   \
    }

#define LIBFLATARRAY_COPY_SOA_ARRAY_MEMBER(MEMBER_INDEX, CELL, MEMBER)  \
    {                                                                   \
        for (int i = 0; i < LIBFLATARRAY_ARRAY_ARITY(MEMBER); ++i) {    \
            std::copy(                                                  \
                &(other.BOOST_PP_SEQ_ELEM(1, MEMBER)()[i]),             \
                &(other.BOOST_PP_SEQ_ELEM(1, MEMBER)()[i]) + count,     \
                &(this->BOOST_PP_SEQ_ELEM(1, MEMBER)()[i]));            \
        }                                                               \
    }

#define LIBFLATARRAY_COPY_SOA_GENERIC_MEMBER(MEMBER_INDEX, CELL, MEMBER) \
    LIBFLATARRAY_ARRAY_CONDITIONAL(                                     \
        MEMBER,                                                         \
        LIBFLATARRAY_COPY_SOA_MEMBER(      MEMBER_INDEX, CELL, MEMBER), \
        LIBFLATARRAY_COPY_SOA_ARRAY_MEMBER(MEMBER_INDEX, CELL, MEMBER))


#define LIBFLATARRAY_INIT_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER)  \
    {                                                                   \
        BOOST_PP_SEQ_ELEM(0, MEMBER) *instance =                        \
            &(this->BOOST_PP_SEQ_ELEM(1, MEMBER)());                    \
        new (instance) BOOST_PP_SEQ_ELEM(0, MEMBER)();                  \
    }

#define LIBFLATARRAY_INIT_SOA_ARRAY_MEMBER(MEMBER_INDEX, CELL, MEMBER) \
    {                                                                   \
        for (int i = 0; i < LIBFLATARRAY_ARRAY_ARITY(MEMBER); ++i) {    \
            new (&(this->BOOST_PP_SEQ_ELEM(1, MEMBER)()[i])) BOOST_PP_SEQ_ELEM(0, MEMBER)(); \
        }                                                               \
    }

#define LIBFLATARRAY_INIT_SOA_GENERIC_MEMBER(MEMBER_INDEX, CELL, MEMBER) \
    LIBFLATARRAY_ARRAY_CONDITIONAL(                                     \
        MEMBER,                                                         \
        LIBFLATARRAY_INIT_SOA_MEMBER(      MEMBER_INDEX, CELL, MEMBER), \
        LIBFLATARRAY_INIT_SOA_ARRAY_MEMBER(MEMBER_INDEX, CELL, MEMBER))

#define LIBFLATARRAY_DESTROY_SOA_MEMBER_ARRAY(MEMBER_INDEX, CELL, MEMBER)  \
    {                                                                   \
        BOOST_PP_SEQ_ELEM(0, MEMBER) *instance =                        \
            &(this->BOOST_PP_SEQ_ELEM(1, MEMBER)());                    \
        detail::flat_array::generic_destruct(instance);                 \
    }

#define LIBFLATARRAY_DESTROY_SOA_ARRAY_MEMBER_ARRAY(MEMBER_INDEX, CELL, MEMBER) \
    {                                                                   \
        for (int i = 0; i < LIBFLATARRAY_ARRAY_ARITY(MEMBER); ++i) {    \
            detail::flat_array::generic_destruct(                       \
                &(this->BOOST_PP_SEQ_ELEM(1, MEMBER)()[i]));            \
        }                                                               \
    }

#define LIBFLATARRAY_DESTROY_SOA_GENERIC_MEMBER(MEMBER_INDEX, CELL, MEMBER) \
    LIBFLATARRAY_ARRAY_CONDITIONAL(                                     \
        MEMBER,                                                         \
        LIBFLATARRAY_DESTROY_SOA_MEMBER_ARRAY(      MEMBER_INDEX, CELL, MEMBER), \
        LIBFLATARRAY_DESTROY_SOA_ARRAY_MEMBER_ARRAY(MEMBER_INDEX, CELL, MEMBER))

#define LIBFLATARRAY_CASE_DIM_X(SIZE_INDEX, UNUSED, SIZE)            \
    if (dim_x <= SIZE) {                                             \
        bind_size_y<CELL, SIZE>(                                     \
            dim_y, dim_z, data, functor);                            \
        return;                                                      \
    }

#define LIBFLATARRAY_CASE_DIM_Y(SIZE_INDEX, UNUSED, SIZE)            \
    if (dim_y <= SIZE) {                                             \
        bind_size_z<CELL, DIM_X, SIZE>(                              \
            dim_z, data, functor);                                   \
        return;                                                      \
    }

#define LIBFLATARRAY_CASE_DIM_Z(SIZE_INDEX, UNUSED, SIZE)               \
    if (dim_z <= SIZE) {                                                \
        LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, SIZE, 0>  accessor( \
            data, 0);                                                   \
        functor(accessor);                                              \
        return;                                                         \
    }

#define LIBFLATARRAY_CASE_DIM_MAX_3D(SIZE_INDEX, UNUSED, SIZE)          \
    if (max <= SIZE) {                                                  \
        LibFlatArray::soa_accessor<CELL, SIZE, SIZE, SIZE, 0>  accessor( \
            data, 0);                                                   \
        functor(accessor);                                              \
        return;                                                         \
    }

#define LIBFLATARRAY_CASE_DIM_MAX_2D(SIZE_INDEX, UNUSED, SIZE)          \
    if (max <= SIZE) {                                                  \
        LibFlatArray::soa_accessor<CELL, SIZE, SIZE, 1, 0>  accessor(   \
            data, 0);                                                   \
        functor(accessor);                                              \
        return;                                                         \
    }

#endif
