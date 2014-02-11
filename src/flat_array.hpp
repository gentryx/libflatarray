/**
 * Copyright 2012-2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_FLAT_ARRAY_HPP
#define FLAT_ARRAY_FLAT_ARRAY_HPP

#include <libflatarray/detail/macros.hpp>
#include <libflatarray/detail/offset.hpp>
#include <libflatarray/aligned_allocator.hpp>
#include <libflatarray/soa_accessor.hpp>
#include <libflatarray/soa_array.hpp>
#include <libflatarray/soa_grid.hpp>
#include <libflatarray/macros.hpp>

#include <boost/preprocessor/seq.hpp>

namespace LibFlatArray {

/**
 * Allow the user to access the number of data members of the SoA type.
 *
 * Will be instantiated by LIBFLATARRAY_REGISTER_SOA().
 */
template<typename CELL_TYPE>
class number_of_members;

template<int X, int Y, int Z>
class coord
{};

/**
 * Lets user code discover a member's offset in the SoA layout.
 *
 * Will be instantiated by LIBFLATARRAY_REGISTER_SOA().
 */
class member_ptr_to_offset
{
public:
    template<typename MEMBER_TYPE, typename CELL_TYPE>
    int operator()(MEMBER_TYPE CELL_TYPE:: *member_ptr)
    {
        return detail::flat_array::offset<
            CELL_TYPE,
            number_of_members<CELL_TYPE>::VALUE>()(member_ptr);
    }
};

}

#endif
