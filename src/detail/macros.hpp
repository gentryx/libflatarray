/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_MACROS_HPP
#define FLAT_ARRAY_DETAIL_MACROS_HPP

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

#define DEFINE_FIELD_OFFSET(r, CELL_TYPE, MEMBER)                       \
    namespace detail {                                                  \
    namespace flat_array {                                              \
    template<>                                                          \
    class offset<CELL_TYPE, r - 1>                                      \
    {                                                                   \
    public:                                                             \
        static const std::size_t OFFSET = offset<CELL_TYPE, r - 2>::OFFSET +  \
            sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER));                       \
                                                                        \
        template<typename MEMBER_TYPE>                                  \
        inline                                                          \
        int operator()(MEMBER_TYPE CELL_TYPE:: *member_ptr)             \
        {                                                               \
            return offset<CELL_TYPE, r - 2>()(member_ptr);              \
        }                                                               \
                                                                        \
        inline                                                          \
        int operator()(BOOST_PP_SEQ_ELEM(0, MEMBER) CELL_TYPE:: *member_ptr) \
        {                                                               \
            if (member_ptr ==                                           \
                &CELL_TYPE::BOOST_PP_SEQ_ELEM(1, MEMBER)) {             \
                return offset<CELL_TYPE, r - 2>::OFFSET;                \
            } else {                                                    \
                return offset<CELL_TYPE, r - 2>()(member_ptr);          \
            }                                                           \
        }                                                               \
                                                                        \
    };                                                                  \
    }                                                                   \
    }

#define DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, CONST, INDEX_VAR) \
    inline                                                              \
    __host__ __device__                                                 \
    CONST BOOST_PP_SEQ_ELEM(0, MEMBER)&                                 \
        BOOST_PP_SEQ_ELEM(1, MEMBER)() CONST                            \
    {                                                                   \
        return *(BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                        \
            data + (DIM_X * DIM_Y * DIM_Z) *                            \
            detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET + \
            INDEX_VAR * sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)) +          \
            INDEX  * sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)));             \
    }

#define DECLARE_SOA_MEMBER_CONST(MEMBER_INDEX, CELL, MEMBER)            \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, const, index)

#define DECLARE_SOA_MEMBER_NORMAL(MEMBER_INDEX, CELL, MEMBER)           \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER,      , index)

#define DECLARE_SOA_MEMBER_LIGHT_CONST(MEMBER_INDEX, CELL, MEMBER)      \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, const, *index)

#define DECLARE_SOA_MEMBER_LIGHT_NORMAL(MEMBER_INDEX, CELL, MEMBER)     \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER,      , *index)

#define COPY_SOA_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER)                  \
    BOOST_PP_SEQ_ELEM(1, MEMBER)() = cell.BOOST_PP_SEQ_ELEM(1, MEMBER);

#define COPY_SOA_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER)                 \
    cell.BOOST_PP_SEQ_ELEM(1, MEMBER) = this->BOOST_PP_SEQ_ELEM(1, MEMBER)();

#define COPY_SOA_MEMBER_ARRAY_IN(MEMBER_INDEX, CELL, MEMBER)            \
    std::copy(                                                          \
        (const BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                          \
            source +                                                    \
            detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET *\
            count),                                                     \
        (const BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                          \
            source +                                                    \
            detail::flat_array::offset<CELL, MEMBER_INDEX - 1>::OFFSET *\
            count),                                                     \
        &this->BOOST_PP_SEQ_ELEM(1, MEMBER)());

#define COPY_SOA_MEMBER_ARRAY_OUT(MEMBER_INDEX, CELL, MEMBER)           \
    std::copy(                                                          \
        &this->BOOST_PP_SEQ_ELEM(1, MEMBER)(),                          \
        &this->BOOST_PP_SEQ_ELEM(1, MEMBER)() + count,                  \
        (BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                                \
            target +                                                    \
            detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET *\
            count));

#define CASE_DIM_X(SIZE_INDEX, UNUSED, SIZE)                         \
    if (dim_x <= SIZE) {                                             \
        bind_size_y<CELL, SIZE>(                                     \
            dim_y, dim_z, data, functor);                            \
        return;                                                      \
    }

#define CASE_DIM_Y(SIZE_INDEX, UNUSED, SIZE)                         \
    if (dim_y <= SIZE) {                                             \
        bind_size_z<CELL, DIM_X, SIZE>(                              \
            dim_z, data, functor);                                   \
        return;                                                      \
    }

#define CASE_DIM_Z(SIZE_INDEX, UNUSED, SIZE)                            \
    if (dim_z <= SIZE) {                                                \
        soa_accessor<CELL, DIM_X, DIM_Y, SIZE, 0>  accessor(            \
            data, 0);                                                   \
        functor(accessor,                                               \
                &accessor.index);                                       \
        return;                                                         \
    }

#define CASE_DIM_MAX(SIZE_INDEX, UNUSED, SIZE)                          \
    if (max <= SIZE) {                                                  \
        soa_accessor<CELL, SIZE, SIZE, SIZE, 0>  accessor(              \
            data, 0);                                                   \
        functor(accessor,                                               \
                &accessor.index);                                       \
        return;                                                         \
    }

#endif
