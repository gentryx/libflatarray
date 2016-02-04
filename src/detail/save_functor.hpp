/**
 * Copyright 2014, 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SAVE_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_SAVE_FUNCTOR_HPP

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * Same as load_functor, but the other way around.
 */
template<typename CELL>
class save_functor
{
public:
    save_functor(
        std::size_t x,
        std::size_t y,
        std::size_t z,
        char *target,
        long count) :
        target(target),
        count(count),
        x(x),
        y(y),
        z(z)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        accessor.index = x + y * DIM_X + z * DIM_X * DIM_Y;
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
