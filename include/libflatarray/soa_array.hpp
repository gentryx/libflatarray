/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2018 Google
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SOA_ARRAY_HPP
#define FLAT_ARRAY_SOA_ARRAY_HPP

#include <libflatarray/aggregated_member_size.hpp>
#include <libflatarray/detail/macros.hpp>
#include <libflatarray/soa_accessor.hpp>

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <stdexcept>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibFlatArray {

// padding is fine, as is not inlining functions:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 4820 )
#endif

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
template<typename T, int MY_SIZE>
class soa_array
{
public:
    template<typename T2, int MY_SIZE2>
    friend void swap(soa_array<T2, MY_SIZE2>& a, soa_array<T2, MY_SIZE2>& b);

    typedef T value_type;
    typedef soa_accessor<value_type, MY_SIZE, 1, 1, 0> iterator;
    static const std::size_t SIZE = MY_SIZE;
    static const std::size_t BYTE_SIZE = aggregated_member_size<value_type>::VALUE * SIZE;

    inline
    __host__ __device__
    explicit soa_array(std::size_t elements = 0, const value_type& value = value_type()) :
        elements(elements)
    {
        construct_all_instances();
        for (soa_accessor<value_type, SIZE, 1, 1, 0> accessor(my_data, 0); accessor.index() < int(elements); accessor += 1) {
            accessor << value;
        }
    }

    template<int OTHER_SIZE>
    inline
    __host__ __device__
    explicit soa_array(soa_array<value_type, OTHER_SIZE>& other)
    {
        construct_all_instances();
        copy_in(other);
    }


    template<int OTHER_SIZE>
    inline
    __host__ __device__
    explicit soa_array(const soa_array<value_type, OTHER_SIZE>& other)
    {
        construct_all_instances();
        copy_in(other);
    }

    inline
    __host__ __device__
    ~soa_array()
    {
        for (soa_accessor<value_type, SIZE, 1, 1, 0> accessor(my_data, 0); accessor.index() < MY_SIZE; accessor += 1) {
            accessor.destroy_members();
        }

    }

    template<int OTHER_SIZE>
    inline
    __host__ __device__
    soa_array& operator=(soa_array<value_type, OTHER_SIZE>& other)
    {
        copy_in(other);
        return *this;
    }

    template<int OTHER_SIZE>
    inline
    __host__ __device__
    soa_array& operator=(const soa_array<value_type, OTHER_SIZE>& other)
    {
        copy_in(other);
        return *this;
    }

    inline
    __host__ __device__
    soa_accessor<value_type, SIZE, 1, 1, 0> operator[](const int index)
    {
        return soa_accessor<value_type, SIZE, 1, 1, 0>(my_data, index);
    }

    inline
    __host__ __device__
    const const_soa_accessor<value_type, SIZE, 1, 1, 0> operator[](const int index) const
    {
        return const_soa_accessor<value_type, SIZE, 1, 1, 0>(my_data, index);
    }

    inline
    __host__ __device__
    soa_accessor<value_type, SIZE, 1, 1, 0> at(const int index)
    {
        return (*this)[index];
    }

    inline
    __host__ __device__
    soa_accessor<value_type, SIZE, 1, 1, 0> at(const std::size_t index)
    {
        return (*this)[static_cast<std::ptrdiff_t>(index)];
    }

    inline
    __host__ __device__
    const_soa_accessor<value_type, SIZE, 1, 1, 0> at(const int index) const
    {
        return (*this)[index];
    }

    inline
    __host__ __device__
    const_soa_accessor<value_type, SIZE, 1, 1, 0> at(const std::size_t index) const
    {
        return (*this)[static_cast<std::ptrdiff_t>(index)];
    }

    inline
    __host__ __device__
    soa_array<value_type, SIZE>& operator<<(const value_type& cell)
    {
#ifndef __CUDA_ARCH__
        if (elements >= SIZE) {
            throw std::out_of_range("capacity exceeded");
        }
#endif
        (*this)[static_cast<std::ptrdiff_t>(elements)] = cell;
        ++elements;

        return *this;
    }

    template<long OTHER_SIZE>
    inline
    __host__ __device__
    void load(const soa_accessor<value_type, OTHER_SIZE, 1, 1, 0>& accessor, std::size_t num)
    {
        load(accessor, num, elements);
    }

    template<long OTHER_SIZE>
    inline
    __host__ __device__
    void load(const soa_accessor<value_type, OTHER_SIZE, 1, 1, 0>& accessor, std::size_t num, std::size_t offset)
    {
        using std::max;
        std::size_t new_elements = max(elements, num + offset);
        if (new_elements > SIZE) {
            throw std::out_of_range("insufficient capacity for assignment (other soa_array too large)");
        }

        at(offset).load(accessor.data(), num, std::size_t(accessor.index()), std::size_t(OTHER_SIZE));
        elements = new_elements;
    }

    inline
    __host__ __device__
    void clear()
    {
        elements = 0;
    }

    inline
    __host__ __device__
    std::size_t capacity() const
    {
        return SIZE;
    }

    inline
    __host__ __device__
    soa_accessor<value_type, SIZE, 1, 1, 0> back()
    {
        return at(elements - 1);
    }

    inline
    __host__ __device__
    soa_accessor<value_type, SIZE, 1, 1, 0> begin()
    {
        return at(0);
    }

    inline
    __host__ __device__
    soa_accessor<value_type, SIZE, 1, 1, 0> end()
    {
        return at(elements);
    }

    inline
    __host__ __device__
    void pop_back()
    {
#ifndef __CUDA_ARCH__
        if (elements == 0) {
            throw std::out_of_range("soa_array is already empty");
        }
#endif

        --elements;
    }

    inline
    __host__ __device__
    void push_back(const value_type& cell)
    {
        *this << cell;
    }

    inline
    __host__ __device__
    std::size_t size() const
    {
        return elements;
    }

    char *data()
    {
        return my_data;
    }

    std::size_t byte_size() const
    {
        return elements * aggregated_member_size<value_type>::VALUE;
    }

private:
    std::size_t elements;
    char my_data[BYTE_SIZE];

    inline
    __host__ __device__
    void construct_all_instances()
    {
        for (soa_accessor<value_type, SIZE, 1, 1, 0> accessor(my_data, 0); accessor.index() < MY_SIZE; accessor += 1) {
            accessor.construct_members();
        }
    }

    template<int OTHER_SIZE>
    inline
    __host__ __device__
    void copy_in(const soa_array<value_type, OTHER_SIZE>& other)
    {
        if (other.size() > SIZE) {
            throw std::out_of_range("insufficient capacity for assignment (other soa_array too large)");
        }

        at(0).copy_members(other[0], other.size());
        elements = other.size();
    }
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif


template<typename value_type, int size>
void swap(soa_array<value_type, size>& a, soa_array<value_type, size>& b)
{
    using std::swap;
    swap(a.elements, b.elements);
    swap(a.my_data,  b.my_data);
}

}

#endif
