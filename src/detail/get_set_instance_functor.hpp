/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_GET_SET_INSTANCE_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_GET_SET_INSTANCE_FUNCTOR_HPP

#include <libflatarray/soa_accessor.hpp>

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * This helper class is used to retrieve objects from the SoA storage
 * with the help of an accessor.
 */
template<typename CELL>
class get_instance_functor
{
public:
    get_instance_functor(
        CELL *target,
        long x,
        long y,
        long z,
        long count) :
        target(target),
        x(x),
        y(y),
        z(z),
        count(count)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        accessor.index =
            z * DIM_X * DIM_Y +
            y * DIM_X +
            x;
        CELL *cursor = target;

        for (long i = 0; i < count; ++i) {
            accessor >> *cursor;
            ++cursor;
            ++accessor.index;
        }
    }

private:
    CELL *target;
    long x;
    long y;
    long z;
    long count;
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
        long x,
        long y,
        long z,
        long count) :
        source(source),
        x(x),
        y(y),
        z(z),
        count(count)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        accessor.index =
            z * DIM_X * DIM_Y +
            y * DIM_X +
            x;
        const CELL *cursor = source;
        for (long i = 0; i < count; ++i) {
            accessor << *cursor;
            ++cursor;
            ++accessor.index;
        }
    }

private:
    const CELL *source;
    long x;
    long y;
    long z;
    long count;
};

}

}

}

#endif
