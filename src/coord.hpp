/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_COORD_HPP
#define FLAT_ARRAY_COORD_HPP

namespace LibFlatArray {

/**
 * A utility class to specify (relative) coordinates. The class is to
 * be used with soa_accessor.
 *
 * Since the coordinates are fixed at compile time, all dependent
 * address calculations can be done at compile time.
 */
template<long X, long Y, long Z>
class coord
{};

}

#endif
