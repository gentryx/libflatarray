/**
 * Copyright 2012-2013 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef ALIGNED_ALLOCATOR_HPP
#define ALIGNED_ALLOCATOR_HPP

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include <memory>

namespace LibFlatArray {

template<class T, std::size_t ALIGNMENT>
class AlignedAllocator
{
public:
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    inline pointer address(reference x) const
    {
        return &x;
    }

    inline const_pointer address(const_reference x) const
    {
        return &x;
    }

    pointer allocate(std::size_t n, const void* = 0)
    {
        // This code whould have been a piece of cake if it would have
        // worket with posix_memalign, which it doesn't. Alternatively
        // we allocate a larger chunk of memory in which we can
        // accomodate an array of the selected size, shifted to the
        // desired offset. Since we need the original address for the
        // deallocation, we store it directly in front of the aligned
        // array's start. Ugly, but it works.
        char *chunk = std::allocator<char>().allocate(upsize(n));
        if (chunk == 0) {
            return (pointer)chunk;
        }

        std::size_t offset = (std::size_t)chunk % ALIGNMENT;
        std::size_t correction = ALIGNMENT - offset;
        if (correction < sizeof(char*))
            correction += ALIGNMENT;
        char *ret = chunk + correction;
        *((char**)ret - 1) = chunk;
        return (pointer)ret;
    }

    void deallocate(pointer p, std::size_t n)
    {
        if (p == 0) {
            return;
        }

        char *actual;
        // retrieve the original pointer which sits in front of its
        // aligned brother
        actual = *((char**)p - 1);
        std::allocator<char>().deallocate(actual, upsize(n));
    }

    std::size_t max_size() const throw()
    {
        return std::allocator<T>().max_size();
    }

    void construct(pointer p, const_reference val)
    {
        std::allocator<T>().construct(p, val);
    }

    void destroy(pointer p)
    {
        std::allocator<T>().destroy(p);
    }

private:
    std::size_t graceOffset()
    {
        return ALIGNMENT + sizeof(char*);
    }

    std::size_t upsize(std::size_t n)
    {
        return n * sizeof(T) + graceOffset();
    }
};

}

#endif
