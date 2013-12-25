/**
 * Copyright 2012-2013 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef CUDA_ALLOCATOR_HPP
#define CUDA_ALLOCATOR_HPP

#include <cuda.h>

namespace LibFlatArray {

template<class T>
class cuda_allocator
{
public:
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    pointer allocate(std::size_t n, const void* = 0)
    {
        pointer ret;
        cudaMalloc(&ret, n * sizeof(T));
        return ret;
    }

    pointer deallocate(pointer p, std::size_t)
    {
        cudaFree(p);
    }
};

}

#endif
