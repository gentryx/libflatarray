/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_AGGREGATED_MEMBER_SIZE_HPP
#define FLAT_ARRAY_AGGREGATED_MEMBER_SIZE_HPP

namespace LibFlatArray {

/**
 * Accumulate the sizes of the individual data members. This may be
 * lower than sizeof(CELL_TYPE) as structs/objects in C++ may need
 * padding. We can avoid the padding of individual members in a SoA
 * memory layout.
 */
template<typename CELL_TYPE>
class aggregated_member_size;

}

#endif
