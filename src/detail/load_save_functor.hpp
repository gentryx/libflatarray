/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_LOAD_SAVE_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_LOAD_SAVE_FUNCTOR_HPP

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * The purpose of this functor is to load a row of cells which are
 * already prepackaged (in SoA form) in a raw data segment (i.e. all
 * members are stored in a consecutive array of the given length and
 * all arrays are concatenated).
 */
template<typename CELL>
class load_functor
{
public:
    load_functor(
        size_t x,
        size_t y,
        size_t z,
        const char *source,
        long count) :
        source(source),
        count(count),
        x(x),
        y(y),
        z(z)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor, long *index) const
    {
        *index = x + y * DIM_X + z * DIM_X * DIM_Y;
        accessor.load(source, count);
    }

private:
    const char *source;
    long count;
    long x;
    long y;
    long z;
};

/**
 * Same as save_functor, but the other way around.
 */
template<typename CELL>
class save_functor
{
public:
    save_functor(
        size_t x,
        size_t y,
        size_t z,
        char *target,
        long count) :
        target(target),
        count(count),
        x(x),
        y(y),
        z(z)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor, long *index) const
    {
        *index = x + y * DIM_X + z * DIM_X * DIM_Y;
        accessor.save(target, count);
    }

private:
    char *target;
    long count;
    long x;
    long y;
    long z;
};

}

}

}

#endif
