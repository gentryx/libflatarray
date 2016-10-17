/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_STREAMING_SHORT_VEC_SWITCH_HPP
#define FLAT_ARRAY_DETAIL_STREAMING_SHORT_VEC_SWITCH_HPP

#include <libflatarray/short_vec.hpp>
#include <libflatarray/streaming_short_vec.hpp>

namespace LibFlatArray {
namespace detail {
namespace flat_array {

template<typename CARGO, int ARITY, int STREAMING_FLAG>
class streaming_short_vec_switch
{
public:
    typedef streaming_short_vec<CARGO, ARITY> VALUE;
};

template<typename CARGO, int ARITY>
class streaming_short_vec_switch<CARGO, ARITY, 0>
{
public:
    typedef short_vec<CARGO, ARITY> VALUE;
};

}
}
}

#endif
