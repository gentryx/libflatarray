/**
 * Copyright 2014 Andreas Schäfer
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

// expands to A if MEMBER is scalar (e.g. double foo),
// expands to B if MEMBER is an array member (e.g. double foo[4]).
#define LIBFLATARRAY_ARRAY_CONDITIONAL(MEMBER, A, B)    \
    BOOST_PP_IF(BOOST_PP_LESS(BOOST_PP_SEQ_SIZE(MEMBER), 3), A, B)

#define LIBFLATARRAY_ARRAY_ARITY(MEMBER)                                \
    BOOST_PP_SEQ_ELEM(BOOST_PP_SUB(BOOST_PP_SEQ_SIZE(MEMBER), 1), MEMBER)

// fixme: libflatarray prefix missing
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

// fixme: libflatarray prefix missing
#define DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, CONST, INDEX_VAR) \
    LIBFLATARRAY_ARRAY_CONDITIONAL(MEMBER, , template<int ARRAY_INDEX >) \
    inline                                                              \
    __host__ __device__                                                 \
    CONST BOOST_PP_SEQ_ELEM(0, MEMBER)&                                 \
        BOOST_PP_SEQ_ELEM(1, MEMBER)() CONST                            \
    {                                                                   \
        return *(BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                        \
            data +                                                      \
            (DIM_X * DIM_Y * DIM_Z) * (                                 \
                sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)) * BOOST_PP_IF(BOOST_PP_LESS(BOOST_PP_SEQ_SIZE(MEMBER), 3), 0, BOOST_PP_SEQ_ELEM(BOOST_PP_SUB(BOOST_PP_SEQ_SIZE(MEMBER), 1), MEMBER)) + \
                detail::flat_array::offset<CELL, MEMBER_INDEX - 2>:: OFFSET) + \
            INDEX_VAR * long(sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER))) +    \
            INDEX     * long(sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER))));    \
    }

// fixme: libflatarray prefix missing
#define DECLARE_SOA_MEMBER_CONST(MEMBER_INDEX, CELL, MEMBER)            \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, const, index)

// fixme: libflatarray prefix missing
#define DECLARE_SOA_MEMBER_NORMAL(MEMBER_INDEX, CELL, MEMBER)           \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER,      , index)

// fixme: libflatarray prefix missing
#define DECLARE_SOA_MEMBER_LIGHT_CONST(MEMBER_INDEX, CELL, MEMBER)      \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, const, *index)

// fixme: libflatarray prefix missing
#define DECLARE_SOA_MEMBER_LIGHT_NORMAL(MEMBER_INDEX, CELL, MEMBER)     \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER,      , *index)

// fixme: libflatarray prefix missing
#define COPY_SOA_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER)                  \
    BOOST_PP_SEQ_ELEM(1, MEMBER)() = cell.BOOST_PP_SEQ_ELEM(1, MEMBER);

template<int SIZE>
class CellCopyHelperFixme
{
public:
    template<typename CELL>
    class Inner1
    {
    public:
        template<typename MEMBER>
        class Inner2
        {
        public:
            template<int ARITY>
            class Inner3
            {
            public:
                template<MEMBER (CELL:: *MEMBER_POINTER)[ARITY]>
                class Inner4
                {
                public:
                    template<int INDEX, typename DUMMY = int>
                    class CopyHelperIn
                    {
                    public:
                        inline
                        void operator()(const CELL& cell, MEMBER *data)
                        {
                            CopyHelperIn<INDEX - 1>()(cell, data);
                            data[SIZE * INDEX - 1] = (cell.*MEMBER_POINTER)[INDEX - 1];
                        }
                    };

                    template<typename DUMMY>
                    class CopyHelperIn<0, DUMMY>
                    {
                    public:
                        inline
                        void operator()(const CELL& cell, MEMBER *data)
                        {}
                    };

                    template<int INDEX, typename DUMMY = int>
                    class CopyHelperOut
                    {
                    public:
                        inline
                        void operator()(CELL& cell, const MEMBER *data)
                        {
                            CopyHelperOut<INDEX - 1>()(cell, data);
                            (cell.*MEMBER_POINTER)[INDEX - 1] = data[SIZE * INDEX - 1];
                        }
                    };

                    template<typename DUMMY>
                    class CopyHelperOut<0, DUMMY>
                    {
                    public:
                        inline
                        void operator()(CELL& cell, const MEMBER *data)
                        {}
                    };
};
            };
        };
    };
};

#define COPY_SOA_ARRAY_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER)            \
    {                                                                   \
        typedef CellCopyHelperFixme<DIM_X * DIM_Y * DIM_Z> Foo1;        \
        typedef typename Foo1::template Inner1<CELL> Foo2;              \
        typedef typename Foo2::template Inner2<BOOST_PP_SEQ_ELEM(0, MEMBER)> Foo3; \
        typedef typename Foo3::template Inner3<LIBFLATARRAY_ARRAY_ARITY(MEMBER)> Foo4; \
        typedef typename Foo4::template Inner4<&CELL::BOOST_PP_SEQ_ELEM(1, MEMBER)> Foo5; \
        typedef typename Foo5::template CopyHelperIn<LIBFLATARRAY_ARRAY_ARITY(MEMBER)> Foo6; \
        Foo6()(                                                         \
            cell,                                                       \
            &BOOST_PP_SEQ_ELEM(1, MEMBER)<0>());                        \
    }

#define COPY_SOA_GENERIC_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER)          \
    LIBFLATARRAY_ARRAY_CONDITIONAL(                                     \
        MEMBER,                                                         \
        COPY_SOA_MEMBER_IN(      MEMBER_INDEX, CELL, MEMBER),           \
        COPY_SOA_ARRAY_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER))

// fixme: libflatarray prefix missing
#define COPY_SOA_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER)                 \
    cell.BOOST_PP_SEQ_ELEM(1, MEMBER) = this->BOOST_PP_SEQ_ELEM(1, MEMBER)();

#define COPY_SOA_ARRAY_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER)           \
    {                                                                   \
        typedef CellCopyHelperFixme<DIM_X * DIM_Y * DIM_Z> Foo1;        \
        typedef typename Foo1::template Inner1<CELL> Foo2;              \
        typedef typename Foo2::template Inner2<BOOST_PP_SEQ_ELEM(0, MEMBER)> Foo3; \
        typedef typename Foo3::template Inner3<LIBFLATARRAY_ARRAY_ARITY(MEMBER)> Foo4; \
        typedef typename Foo4::template Inner4<&CELL::BOOST_PP_SEQ_ELEM(1, MEMBER)> Foo5; \
        typedef typename Foo5::template CopyHelperOut<LIBFLATARRAY_ARRAY_ARITY(MEMBER)> Foo6; \
        Foo6()(                                                         \
            cell,                                                       \
            &BOOST_PP_SEQ_ELEM(1, MEMBER)<0>());                        \
    }

#define COPY_SOA_GENERIC_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER)         \
    LIBFLATARRAY_ARRAY_CONDITIONAL(                                     \
        MEMBER,                                                         \
        COPY_SOA_MEMBER_OUT(      MEMBER_INDEX, CELL, MEMBER),          \
        COPY_SOA_ARRAY_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER))

// fixme: libflatarray prefix missing
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

// fixme: libflatarray prefix missing
#define COPY_SOA_MEMBER_ARRAY_OUT(MEMBER_INDEX, CELL, MEMBER)           \
    std::copy(                                                          \
        &this->BOOST_PP_SEQ_ELEM(1, MEMBER)(),                          \
        &this->BOOST_PP_SEQ_ELEM(1, MEMBER)() + count,                  \
        (BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                                \
            target +                                                    \
            detail::flat_array::offset<CELL, MEMBER_INDEX - 2>::OFFSET *\
            count));

// fixme: libflatarray prefix missing
#define CASE_DIM_X(SIZE_INDEX, UNUSED, SIZE)                         \
    if (dim_x <= SIZE) {                                             \
        bind_size_y<CELL, SIZE>(                                     \
            dim_y, dim_z, data, functor);                            \
        return;                                                      \
    }

// fixme: libflatarray prefix missing
#define CASE_DIM_Y(SIZE_INDEX, UNUSED, SIZE)                         \
    if (dim_y <= SIZE) {                                             \
        bind_size_z<CELL, DIM_X, SIZE>(                              \
            dim_z, data, functor);                                   \
        return;                                                      \
    }

// fixme: libflatarray prefix missing
#define CASE_DIM_Z(SIZE_INDEX, UNUSED, SIZE)                            \
    if (dim_z <= SIZE) {                                                \
        soa_accessor<CELL, DIM_X, DIM_Y, SIZE, 0>  accessor(            \
            data, 0);                                                   \
        functor(accessor);                                              \
        return;                                                         \
    }

// fixme: libflatarray prefix missing
#define CASE_DIM_MAX_3D(SIZE_INDEX, UNUSED, SIZE)                       \
    if (max <= SIZE) {                                                  \
        soa_accessor<CELL, SIZE, SIZE, SIZE, 0>  accessor(              \
            data, 0);                                                   \
        functor(accessor);                                              \
        return;                                                         \
    }

// fixme: libflatarray prefix missing
#define CASE_DIM_MAX_2D(SIZE_INDEX, UNUSED, SIZE)                       \
    if (max <= SIZE) {                                                  \
        soa_accessor<CELL, SIZE, SIZE, 1, 0>  accessor(                 \
            data, 0);                                                   \
        functor(accessor);                                              \
        return;                                                         \
    }

#endif
