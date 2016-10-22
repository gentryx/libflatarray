/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SOA_VECTOR_HPP
#define FLAT_ARRAY_SOA_VECTOR_HPP

#include <libflatarray/aligned_allocator.hpp>
#include <libflatarray/soa_accessor.hpp>
#include <libflatarray/soa_grid.hpp>
#include <stdexcept>

namespace LibFlatArray {

/**
 * This is the runtime resizable counterpart to soa_array. The goal is
 * to provide an interface similar to std::vector and simultaneously
 * have a callback to expose the struct-of-arrays layout.
 */
template<
    typename T,
    typename ALLOCATOR = aligned_allocator<char, 4096>,
    bool USE_CUDA_FUNCTORS = false>
class soa_vector
{
public:
    friend class TestResizeAndReserve;

    typedef T value_type;

    inline
    __host__ __device__
    explicit soa_vector(std::size_t count = 0) :
        grid(count, 1, 1),
        count(count)
    {}

    inline
    __host__ __device__
    explicit soa_vector(std::size_t count, const value_type& value) :
        grid(count, 1, 1),
        count(count)
    {
        grid.broadcast(0, 0, 0, value, count);
    }

    /**
     * Copies an element to the given index. We're intentionally not
     * using at() or operator[] to avoid mismatched expectations here:
     * we can't yield references to a T here.
     */
    inline
    __host__ __device__
    void set(std::size_t index, const T& element)
    {
        grid.set(index, 0, 0, element);
    }

    /**
     * Copy out an element. Again we're not using at() or operator[]
     * here to avoid confusion with the API: we can't return
     * references from an SoA container.
     */
    inline
    __host__ __device__
    T get(std::size_t index) const
    {
        return grid.get(index, 0, 0);
    }

    inline
    __host__ __device__
    std::size_t size() const
    {
        return count;
    }

    inline
    __host__ __device__
    void resize(std::size_t new_count)
    {
        if (new_count > capacity()) {
            reserve(new_count);
        }

        count = new_count;
    }

    inline
    __host__ __device__
    void reserve(std::size_t new_count)
    {
        soa_grid<T, ALLOCATOR, USE_CUDA_FUNCTORS> new_grid(new_count, 1, 1);
        new_grid.resize(new_grid.extent_x(), 1, 1);

        detail::flat_array::simple_streak iter[2] = {
            detail::flat_array::simple_streak(0, 0, 0, count),
            detail::flat_array::simple_streak()
        };

        new_grid.load(iter + 0, iter + 1, grid.data(), grid.extent_x());
        swap(new_grid, grid);
    }

    inline
    __host__ __device__
    std::size_t capacity() const
    {
        return grid.dim_x();
    }

    inline
    __host__ __device__
    void clear()
    {
        count = 0;
    }

    inline
    __host__ __device__
    void push_back(const T& element)
    {
        if (count == grid.extent_x()) {
            // fixme: make this configurable
            reserve(count * 1.2);
        }
        set(count, element);
        ++count;
    }

    inline
    __host__ __device__
    void pop_back()
    {
        --count;
        // destroy last element by overwriting with default element:
        set(count, T());
    }

    template<typename FUNCTOR>
    inline
    __host__ __device__
    void callback(FUNCTOR functor)
    {
        grid.callback(functor);
    }

    template<typename FUNCTOR>
    inline
    __host__ __device__
    void callback(FUNCTOR functor) const
    {
        grid.callback(functor);
    }

private:
    soa_grid<T, ALLOCATOR, USE_CUDA_FUNCTORS> grid;
    std::size_t count;

    // fixme: retrieval of multiple elements
    // fixme: emplace
    // fixme: add cuda test
};

}

#endif

