/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SOA_ACCESSOR_HPP
#define FLAT_ARRAY_SOA_ACCESSOR_HPP

namespace LibFlatArray {

/**
 * This class provides an object-oriented view to a "Struct of
 * Arrays"-style grid. It requires the user to register the type CELL
 * using the macro LIBFLATARRAY_REGISTER_SOA. It provides an
 * operator[] which can be used to access neighboring cells.
 */
template<typename CELL, int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
class soa_accessor;

template<typename CELL, int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
class const_soa_accessor;

template<typename CELL, int GRID_DIM, int INDEX>
class const_soa_accessor_final;

}

#endif

