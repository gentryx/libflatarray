/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_MEMBER_PTR_TO_OFFSET_HPP
#define FLAT_ARRAY_MEMBER_PTR_TO_OFFSET_HPP

#include <libflatarray/number_of_members.hpp>

namespace LibFlatArray {

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
