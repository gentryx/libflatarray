/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SET_BYTE_SIZE_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_SET_BYTE_SIZE_FUNCTOR_HPP

#include <libflatarray/aggregated_member_size.hpp>

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * This helper class uses the dimension specified in the accessor to
 * compute how many bytes a grid needs to allocate im memory.
 */
template<typename CELL>
class set_byte_size_functor
{
public:
    explicit set_byte_size_functor(std::size_t *byte_size) :
        byte_size(byte_size)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(const soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        *byte_size = aggregated_member_size<CELL>::VALUE * DIM_X * DIM_Y * DIM_Z;
    }

private:
    std::size_t *byte_size;
};

}

}

}

#endif
