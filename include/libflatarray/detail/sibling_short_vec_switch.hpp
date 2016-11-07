/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SIBLING_SHORT_VEC_SWITCH_HPP
#define FLAT_ARRAY_DETAIL_SIBLING_SHORT_VEC_SWITCH_HPP

namespace LibFlatArray {
namespace detail {
namespace flat_array {

template<typename SHORT_VEC, std::size_t TARGET_ARITY>
class sibling_short_vec_switch;

template<
    template<typename CARGO_PARAM, std::size_t ARITY_PARAM> class SHORT_VEC_TEMPLATE,
    typename CARGO,
    std::size_t ARITY,
    std::size_t TARGET_ARITY>
class sibling_short_vec_switch<SHORT_VEC_TEMPLATE<CARGO, ARITY>, TARGET_ARITY>
{
public:
    typedef SHORT_VEC_TEMPLATE<CARGO, TARGET_ARITY> VALUE;
};

}
}
}

#endif
