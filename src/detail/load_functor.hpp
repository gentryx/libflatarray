/**
 * Copyright 2014, 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_LOAD_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_LOAD_FUNCTOR_HPP

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
        std::size_t x,
        std::size_t y,
        std::size_t z,
        const char *source,
        long count) :
        source(source),
        count(count),
        x(x),
        y(y),
        z(z)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        accessor.index = x + y * DIM_X + z * DIM_X * DIM_Y;
        accessor.load(source, count);
    }

private:
    const char *source;
    long count;
    long x;
    long y;
    long z;
};

}

}

}

#endif
