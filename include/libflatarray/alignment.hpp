/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_ALIGNMENT_HPP
#define FLAT_ARRAY_ALIGNMENT_HPP

#include <libflatarray/short_vec.hpp>
#include <libflatarray/streaming_short_vec.hpp>

namespace LibFlatArray {

template<typename vec>
class alignment;

template<typename T, std::size_t ARITY>
class alignment<short_vec<T, ARITY> >
{
public:
    typedef typename short_vec<T, ARITY>::strategy strategy;
    typedef typename strategy::template alignment<T> align;
    const static std::size_t VALUE = align::ALIGNMENT;
};

template<typename T, std::size_t ARITY>
class alignment<streaming_short_vec<T, ARITY> >
{
public:
    typedef typename short_vec<T, ARITY>::strategy strategy;
    typedef typename strategy::template alignment<T> align;
    const static std::size_t VALUE = align::ALIGNMENT;
};

}

#endif
