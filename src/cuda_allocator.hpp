/**
 * Copyright 2012-2015 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_CUDA_ALLOCATOR_HPP
#define FLAT_ARRAY_CUDA_ALLOCATOR_HPP

#ifdef __ICC
// disabling this warning as implicit type conversion here as it's an intented feature for dim3
#pragma warning push
#pragma warning (disable: 2304)
#endif

#include <cuda.h>

#ifdef __ICC
#pragma warning pop
#endif

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

    void deallocate(pointer p, std::size_t)
    {
        cudaFree(p);
    }
};

}

#endif
