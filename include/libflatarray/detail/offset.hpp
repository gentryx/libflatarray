/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2018 Google
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_OFFSET_HPP
#define FLAT_ARRAY_DETAIL_OFFSET_HPP

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 )
#endif

#include <stdexcept>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibFlatArray {

namespace detail {

namespace flat_array {

template<typename CELL, long I>
class offset;

template<typename CELL>
class offset<CELL, 0>
{
public:
    static const long OFFSET = 0;

    template<typename MEMBER_TYPE>
    int operator()(MEMBER_TYPE CELL::* /* member_ptr */)
    {
        throw std::invalid_argument("member was not registered with LibFlatArray");
    }
};

}

}

}

#endif
