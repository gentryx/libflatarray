/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SOA_VECTOR_HPP
#define FLAT_ARRAY_SOA_VECTOR_HPP

#include <libflatarray/soa_accessor.hpp>
#include <libflatarray/soa_grid.hpp>
#include <stdexcept>

namespace LibFlatArray {

/**
 * This is the runtime resizable counterpart to soa_array. The goal is
 * to provide an interface similar to std::vector and simultaneously
 * have a callback to expose the struct-of-arrays layout.
 */
template<typename T>
class soa_vector
{
public:
    typedef T value_type;

    inline
    __host__ __device__
    explicit soa_vector(std::size_t count = 0) :
        grid(elements, 1, 1),
        count(count)
    {}

    inline
    __host__ __device__
    explicit soa_vector(std::size_t count, const value_type& value) :
        grid(count, 1, 1)
    {
        // fixme: copy in default value
    }

    inline
    __host__ __device__
    std::size_t size()
    {
        return count;
    }
private:
    soa_grid<T> grid;
    std::size_t count;

    // fixme: at()
    // fixme: operator[]
    // fixme: resize
    // fixme: reserve
    // fixme: size
    // fixme: begin
    // fixme: end
    // fixme: push_back
};

}

#endif

