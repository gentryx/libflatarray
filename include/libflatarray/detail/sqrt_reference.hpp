/**
 * Copyright 2014-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SQRT_REFERENCE_HPP
#define FLAT_ARRAY_DETAIL_SQRT_REFERENCE_HPP

namespace LibFlatArray {

template<typename CARGO, std::size_t ARITY>
class sqrt_reference;

template<typename CARGO, std::size_t ARITY>
short_vec<CARGO, ARITY> operator/(const sqrt_reference<CARGO, ARITY>& a, const short_vec<CARGO, ARITY>& b)
{
    return short_vec<CARGO, ARITY>(a) / b;
}

template<typename CARGO, std::size_t ARITY>
inline short_vec<CARGO, ARITY> operator/(const sqrt_reference<CARGO, ARITY>& a, const CARGO b)
{
    return short_vec<CARGO, ARITY>(a) / b;
}

}

#endif
