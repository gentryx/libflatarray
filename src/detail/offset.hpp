/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_OFFSET_HPP
#define FLAT_ARRAY_DETAIL_OFFSET_HPP

#include <stdexcept>

namespace LibFlatArray {

namespace detail {

namespace flat_array {

template<typename CELL, int I>
class offset;

template<typename CELL>
class offset<CELL, 0>
{
public:
    static const std::size_t OFFSET = 0;

    template<typename MEMBER_TYPE>
    int operator()(MEMBER_TYPE CELL:: *member_ptr)
    {
        throw std::invalid_argument("member was not registered with LibFlatArray");
    }
};

}

}

}

#endif
