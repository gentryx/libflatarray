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
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor, int *index) const
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

}

}

}

#endif
