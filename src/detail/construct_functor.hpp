/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_CONSTRUCT_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_CONSTRUCT_FUNCTOR_HPP

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * Will initialize all grid cells by, relies on the SoA (Struct of
 * Arrays) accessor to initialize a cell's members individually.
 */
template<typename CELL>
class construct_functor
{
public:
    construct_functor(
        size_t dim_x,
        size_t dim_y,
        size_t dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        for (size_t z = 0; z < dim_z; ++z) {
            for (size_t y = 0; y < dim_y; ++y) {
                accessor.index = soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>::gen_index(0, y, z);

                for (size_t x = 0; x < dim_x; ++x) {
                    accessor.construct_members();
                    ++accessor;
                }
            }
        }
    }

private:
    size_t dim_x;
    size_t dim_y;
    size_t dim_z;
};

}

}

}

#endif
