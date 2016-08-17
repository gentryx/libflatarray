/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_COPY_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_COPY_FUNCTOR_HPP

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * Will copy all members of all grid cells by invoking std::copy on
 * all member instances. We can't just memcpy() as members may still
 * be C++ objects that need to run their copy c-tors for
 * allocation/deallocation
 */
template<typename CELL>
class copy_functor
{
public:
    copy_functor(
        std::size_t dim_x,
        std::size_t dim_y,
        std::size_t dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(ACCESSOR1& source_accessor, ACCESSOR2 target_accessor) const
    {
        for (std::size_t z = 0; z < dim_z; ++z) {
            for (std::size_t y = 0; y < dim_y; ++y) {
                target_accessor.index() = ACCESSOR1::gen_index(0, y, z);
                source_accessor.index() = target_accessor.index();
                target_accessor.copy_members(source_accessor, dim_x);
            }
        }
    }

private:
    std::size_t dim_x;
    std::size_t dim_y;
    std::size_t dim_z;
};

}

}

}

#endif
