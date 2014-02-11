/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SOA_ARRAY_HPP
#define FLAT_ARRAY_SOA_ARRAY_HPP

#include <libflatarray/aggregated_member_size.hpp>

#include <stdexcept>

namespace LibFlatArray {

template<typename CELL, int MY_SIZE>
class soa_array
{
public:
    typedef CELL Cell;
    static const int SIZE = MY_SIZE;
    static const int BYTE_SIZE = aggregated_member_size<CELL>::VALUE * SIZE;

    inline
    __host__ __device__
    soa_array(int elements = 0, const CELL& value = CELL()) :
        elements(elements),
        index(0)
    {
        int i = 0;
        soa_accessor<CELL, SIZE, 0, 0, 0> accessor(data, &i);
        for (; i < elements; ++i) {
            accessor << value;
        }
    }

    inline
    __host__ __device__
    soa_accessor<CELL, SIZE, 1, 1, 0> operator[](int& index)
    {
        return soa_accessor<CELL, SIZE, 1, 1, 0>(data, &index);
    }

    inline
    __host__ __device__
    const const_soa_accessor<CELL, SIZE, 1, 1, 0> operator[](int& index) const
    {
        return const_soa_accessor<CELL, SIZE, 1, 1, 0>(data, &index);
    }

    inline
    __host__ __device__
    soa_accessor<CELL, SIZE, 1, 1, 0> at(int& index)
    {
        return (*this)[index];
    }

    inline
    __host__ __device__
    const_soa_accessor<CELL, SIZE, 1, 1, 0> at(int& index) const
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
    size_t size() const
    {
        return elements;
    }

private:
    char data[BYTE_SIZE];
    int elements;
    int index;
};

}

#endif
