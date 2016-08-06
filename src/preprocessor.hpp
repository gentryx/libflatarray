/**
 * Copyright 2016 Andreas Schäfer,
 * heavily based on the Boost Preprocessor library by Paul Mensonides (copyright 2002)
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_PREPROCESSOR_HPP
#define FLAT_ARRAY_PREPROCESSOR_HPP

#include <libflatarray/detail/preprocessor.hpp>

/**
 * Returns the element of LIST at position INDEX. Assumes 0-based
 * addressing.
 */
#define LIBFLATARRAY_ELEM(INDEX, LIST) LIBFLATARRAY_ELEM_I(INDEX, LIST)

/**
 * Return lenght of LIST. LIST is assumed to be of the form
 *
 * (foo)(bar)(goo)
 *
 * i.e. all elements are enclosed in parentheses.
 */
#define LIBFLATARRAY_SIZE(LIST) LIBFLATARRAY_SIZE_I(LIBFLATARRAY_SIZE_0 LIST)

// Expands to an empty string, useful for deleting arguments from a
// list.
#define LIBFLATARRAY_NULL(_)

// Returns a list which is identical to LIST, but with the first
// element removed. Will fail for empty lists.
#define LIBFLATARRAY_DEQUEUE(LIST) LIBFLATARRAY_NULL LIST

/**
 * Will instantiate MACRO for each element of LIST with three parameters:
 * 1. an integer index, starting at 0,
 * 2. PARAM
 * 3. the element of LIST at the given index.
 */
#define LIBFLATARRAY_FOR_EACH(MACRO, DEFAULT_ARG, LIST) LIBFLATARRAY_FOR_EACH_I(MACRO, DEFAULT_ARG, LIBFLATARRAY_DEQUEUE(LIST), LIST)

/**
 * Will expand to A if the size of LIST is less than LENGTH. Will
 * expand to B if the number of elements in LIST is equal to or larger
 * than LENGTH.
 */
#define LIBFLATARRAY_IF_SHORTER(LIST, LENGTH, A, B) LIBFLATARRAY_IF_SHORTER_I(LIBFLATARRAY_IF_SHORTER_ ## LENGTH, LIST, A, B)

#endif
