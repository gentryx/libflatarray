/**
 * Copyright 2016 Andreas Schäfer
 * Copyright 2018 Google
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_LOOP_PEELER_HPP
#define FLAT_ARRAY_LOOP_PEELER_HPP

#include <libflatarray/config.h>
#include <libflatarray/detail/sibling_short_vec_switch.hpp>

#ifdef _MSC_BUILD
/**
 * This is a shim to ease handling of unaligned or not vectorizable
 * iterations at the begin/end of loops. It will invoke FUNCTION with
 * a suitable variant of SHORT_VEC (with its arity adjusted) to that
 * the main chunk of the iterations will be running with full
 * vectorization (as given by SHORT_VEC) and only the initial
 * (possibly unaligned) and trailing (less than SHORT_VEC's arity)
 * iterations will be done with an arity of 1 (i.e. scalar).
 *
 * X is expected to be increased by FUNCTION (e.g. by passing it via
 * reference).
 */
#define LIBFLATARRAY_LOOP_PEELER(SHORT_VEC_TYPE, COUNTER_TYPE,          \
                                 X, END_X, FUNCTION, ...)               \
    __pragma( warning( push ) )                                         \
    __pragma( warning( disable : 4710 4711 ) )                          \
    LIBFLATARRAY_LOOP_PEELER_IMPLEMENTATION(                            \
        , SHORT_VEC_TYPE, COUNTER_TYPE, X, END_X, FUNCTION, __VA_ARGS__) \
    __pragma( warning( pop ) )
#else
/**
 * This is a shim to ease handling of unaligned or not vectorizable
 * iterations at the begin/end of loops. It will invoke FUNCTION with
 * a suitable variant of SHORT_VEC (with its arity adjusted) to that
 * the main chunk of the iterations will be running with full
 * vectorization (as given by SHORT_VEC) and only the initial
 * (possibly unaligned) and trailing (less than SHORT_VEC's arity)
 * iterations will be done with an arity of 1 (i.e. scalar).
 *
 * X is expected to be increased by FUNCTION (e.g. by passing it via
 * reference).
 */
#define LIBFLATARRAY_LOOP_PEELER(SHORT_VEC_TYPE, COUNTER_TYPE,          \
                                 X, END_X, FUNCTION, ...)               \
    LIBFLATARRAY_LOOP_PEELER_IMPLEMENTATION(                            \
        , SHORT_VEC_TYPE, COUNTER_TYPE, X, END_X, FUNCTION, __VA_ARGS__)
#endif

#ifdef _MSC_BUILD
/**
 * Same as LIBFLATARRAY_LOOP_PEELER(), but for use in templates
 */
#define LIBFLATARRAY_LOOP_PEELER_TEMPLATE(SHORT_VEC_TYPE, COUNTER_TYPE, \
                                          X, END_X, FUNCTION, ...)      \
    __pragma( warning( push ) )                                         \
    __pragma( warning( disable : 4710 4711 ) )                          \
    LIBFLATARRAY_LOOP_PEELER_IMPLEMENTATION(                            \
        typename, SHORT_VEC_TYPE, COUNTER_TYPE, X, END_X, FUNCTION, __VA_ARGS__) \
    __pragma( warning( pop ) )
#else
/**
 * Same as LIBFLATARRAY_LOOP_PEELER(), but for use in templates
 */
#define LIBFLATARRAY_LOOP_PEELER_TEMPLATE(SHORT_VEC_TYPE, COUNTER_TYPE, \
                                          X, END_X, FUNCTION, ...)      \
    LIBFLATARRAY_LOOP_PEELER_IMPLEMENTATION(                            \
        typename, SHORT_VEC_TYPE, COUNTER_TYPE, X, END_X, FUNCTION, __VA_ARGS__)
#endif

#define LIBFLATARRAY_LOOP_PEELER_IMPLEMENTATION(                        \
    TYPENAME, SHORT_VEC_TYPE, COUNTER_TYPE, X, END_X, FUNCTION, ...)    \
    {                                                                   \
        typedef SHORT_VEC_TYPE lfa_local_short_vec;                     \
        typedef TYPENAME LibFlatArray::detail::flat_array::             \
            sibling_short_vec_switch<SHORT_VEC_TYPE, 1>::VALUE          \
            lfa_local_scalar;                                           \
                                                                        \
        COUNTER_TYPE remainder = (X) %                                  \
            COUNTER_TYPE(lfa_local_short_vec::ARITY);                   \
        COUNTER_TYPE next_stop = remainder ?                            \
            (X) + COUNTER_TYPE(lfa_local_short_vec::ARITY) - remainder : \
            (X);                                                        \
        COUNTER_TYPE last_stop = (END_X) -                              \
            (END_X) % COUNTER_TYPE(lfa_local_short_vec::ARITY);         \
                                                                        \
        FUNCTION<lfa_local_scalar   >(X, next_stop, __VA_ARGS__);       \
        FUNCTION<lfa_local_short_vec>(X, last_stop, __VA_ARGS__);       \
        FUNCTION<lfa_local_scalar   >(X, (END_X),   __VA_ARGS__);       \
    }

#ifdef LIBFLATARRAY_WITH_CPP14

namespace LibFlatArray {

template<typename SHORT_VEC_TYPE, typename COUNTER_TYPE1, typename COUNTER_TYPE2, typename LAMBDA>
void loop_peeler(COUNTER_TYPE1 *counter, const COUNTER_TYPE2& end, const LAMBDA& lambda)
{
    typedef SHORT_VEC_TYPE lfa_local_short_vec;
    typedef typename detail::flat_array::
        sibling_short_vec_switch<SHORT_VEC_TYPE, 1>::VALUE
        lfa_local_scalar;

    COUNTER_TYPE2 remainder = (*counter) % COUNTER_TYPE2(lfa_local_short_vec::ARITY);
    COUNTER_TYPE2 next_stop = remainder ?
        (*counter) + COUNTER_TYPE2(lfa_local_short_vec::ARITY) - remainder :
        (*counter);
    COUNTER_TYPE2 last_stop = end - end % COUNTER_TYPE2(lfa_local_short_vec::ARITY);

    lambda(lfa_local_scalar(),    counter, next_stop);
    lambda(lfa_local_short_vec(), counter, last_stop);
    lambda(lfa_local_scalar(),    counter, end      );
}

}

#endif

#endif
