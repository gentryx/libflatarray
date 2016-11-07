/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SHORT_VEC_BASE_HPP
#define FLAT_ARRAY_SHORT_VEC_BASE_HPP

namespace LibFlatArray {

template<typename CARGO, std::size_t ARITY>
class short_vec_base
{
public:
    static inline
    std::size_t size()
    {
        return ARITY;
    }

};

}

#endif
