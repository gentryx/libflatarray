/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SIMPLE_STREAK_HPP
#define FLAT_ARRAY_DETAIL_SIMPLE_STREAK_HPP

namespace LibFlatArray {

namespace detail {

namespace flat_array {

    class simple_streak {
    public:
        explicit simple_streak(std::size_t x = 0, std::size_t y = 0, std::size_t z = 0, std::size_t count = 0) :
            count(count)
        {
            origin[0] = x;
            origin[1] = y;
            origin[2] = z;
        }

        std::size_t length() const
        {
            return count;
        }

        std::size_t origin[3];
        std::size_t count;
    };

}

}

}

#endif
