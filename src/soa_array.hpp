/**
 * Copyright 2014, 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SOA_ARRAY_HPP
#define FLAT_ARRAY_SOA_ARRAY_HPP

#include <libflatarray/aggregated_member_size.hpp>
#include <libflatarray/detail/macros.hpp>
#include <libflatarray/soa_accessor.hpp>
#include <stdexcept>

namespace LibFlatArray {

/**
 * soa_array is a container with "Struct of Arrays"-style memory
 * layout, but "Array of Structs" (AoS) user interface. This allows
 * the user to write short, concise code, but facilitates efficient
 * vectorization, which wouldn't have been possible with an AoS
 * layout.
 *
 * Its capacity is fixed at compile time and it uses PoD-style copy
 * semantics (i.e. it doesn't use pointers). The last two properties
 * simplify (and accelerate) handling with MPI and CUDA.
 */
template<typename CELL, int MY_SIZE>
class soa_array
{
public:
    typedef CELL Cell;
    static const std::size_t SIZE = MY_SIZE;
    static const std::size_t BYTE_SIZE = aggregated_member_size<CELL>::VALUE * SIZE;

    inline
    __host__ __device__
    explicit soa_array(std::size_t elements = 0, const CELL& value = CELL()) :
        elements(elements)
    {
        construct_all_instances();
        for (soa_accessor<CELL, SIZE, 1, 1, 0> accessor(data, 0); accessor.index < int(elements); accessor += 1) {
            accessor << value;
        }
    }

    template<int OTHER_SIZE>
    inline
    soa_array(soa_array<CELL, OTHER_SIZE>& other)
    {
        construct_all_instances();
        copy_in(other);
    }


    template<int OTHER_SIZE>
    inline
    soa_array(const soa_array<CELL, OTHER_SIZE>& other)
    {
        construct_all_instances();
        copy_in(other);
    }

    inline
    ~soa_array()
    {
        for (soa_accessor<CELL, SIZE, 1, 1, 0> accessor(data, 0); accessor.index < MY_SIZE; accessor += 1) {
            accessor.destroy_members();
        }

    }

    template<int OTHER_SIZE>
    inline
    soa_array& operator=(soa_array<CELL, OTHER_SIZE>& other)
    {
        copy_in(other);
        return *this;
    }

    template<int OTHER_SIZE>
    inline
    soa_array& operator=(const soa_array<CELL, OTHER_SIZE>& other)
    {
        copy_in(other);
        return *this;
    }

    inline
    __host__ __device__
    soa_accessor<CELL, SIZE, 1, 1, 0> operator[](const int index)
    {
        return soa_accessor<CELL, SIZE, 1, 1, 0>(data, index);
    }

    inline
    __host__ __device__
    const const_soa_accessor<CELL, SIZE, 1, 1, 0> operator[](const int index) const
    {
        return const_soa_accessor<CELL, SIZE, 1, 1, 0>(data, index);
    }

    inline
    __host__ __device__
    soa_accessor<CELL, SIZE, 1, 1, 0> at(const int index)
    {
        return (*this)[index];
    }

    inline
    __host__ __device__
    const_soa_accessor<CELL, SIZE, 1, 1, 0> at(const int index) const
    {
        return (*this)[index];
    }

   inline
    __host__ __device__
    void operator<<(const CELL& cell)
    {
        if (elements >= SIZE) {
            throw std::out_of_range("capacity exceeded");
        }

        (*this)[elements] = cell;
        ++elements;
    }

    inline
    __host__ __device__
    void push_back(const CELL& cell)
    {
        *this << cell;
    }

    inline
    __host__ __device__
    std::size_t size() const
    {
        return elements;
    }

    char *get_data()
    {
        return data;
    }

    std::size_t byte_size() const
    {
        return elements * aggregated_member_size<CELL>::VALUE;
    }

private:
    std::size_t elements;
    char data[BYTE_SIZE];

    inline
    void construct_all_instances()
    {
        for (soa_accessor<CELL, SIZE, 1, 1, 0> accessor(data, 0); accessor.index < MY_SIZE; accessor += 1) {
            accessor.construct_members();
        }
    }

    template<int OTHER_SIZE>
    inline
    void copy_in(const soa_array<CELL, OTHER_SIZE>& other)
    {
        if (other.size() > SIZE) {
            throw std::out_of_range("insufficient capacity for assignment (other soa_array too large)");
        }

        at(0).copy_members(other[0], other.size());
        elements = other.size();
    }
};

}

#endif
