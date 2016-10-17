/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_NUMBER_OF_MEMBERS_HPP
#define FLAT_ARRAY_NUMBER_OF_MEMBERS_HPP

namespace LibFlatArray {

/**
 * Allow the user to access the number of data members of the SoA type.
 *
 * Will be instantiated by LIBFLATARRAY_REGISTER_SOA().
 */
template<typename CELL_TYPE>
class number_of_members;

}

#endif
