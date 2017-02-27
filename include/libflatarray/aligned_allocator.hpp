/**
 * Copyright 2012-2017 Andreas Schäfer
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_ALIGNED_ALLOCATOR_HPP
#define FLAT_ARRAY_ALIGNED_ALLOCATOR_HPP

#include <memory>

namespace LibFlatArray {

template<class T, std::size_t ALIGNMENT>
class aligned_allocator
{
public:
    typedef std::ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef std::size_t size_type;

    template<typename OTHER>
    struct rebind
    {
        typedef aligned_allocator<OTHER, ALIGNMENT> other;
    };

    template<typename OTHER, int OTHER_ALIGNMENT>
    inline explicit aligned_allocator(const aligned_allocator<OTHER, OTHER_ALIGNMENT>& /* other */)
    {}

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
        // This code would have been a piece of cake if it would have
        // worked with posix_memalign -- which it didn't. Instead
        // we allocate a larger chunk of memory in which we can
        // accomodate an array of the required size, shifted to the
        // desired offset. Since we need the original address for the
        // deallocation, we store it directly in front of the aligned
        // array's start. Ugly, but it works.
        char *chunk = std::allocator<char>().allocate(upsize(n));
        if (chunk == 0) {
            return reinterpret_cast<pointer>(chunk);
        }

        std::size_t offset = reinterpret_cast<std::size_t>(chunk) % ALIGNMENT;
        std::size_t correction = ALIGNMENT - offset;
        if (correction < sizeof(char*)) {
            correction += ALIGNMENT;
        }
        char *ret = chunk + correction;
        *(reinterpret_cast<char**>(ret) - 1) = chunk;
        return reinterpret_cast<pointer>(ret);
    }

    void deallocate(pointer p, std::size_t n)
    {
        if (p == 0) {
            return;
        }

        char *actual;
        // retrieve the original pointer which sits in front of its
        // aligned brother
        actual = *(reinterpret_cast<char**>(p) - 1);
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

    /**
     * Added due to compiling for Intel MIC with CPP14=TRUE
     * GCC Bug Report: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51626
     */
    void construct(pointer p)
    {
        std::allocator<T>().construct(p, value_type());
    }

    void destroy(pointer p)
    {
        std::allocator<T>().destroy(p);
    }

    bool operator!=(const aligned_allocator& other) const
    {
        return !(*this == other);
    }

    bool operator==(const aligned_allocator& other) const
    {
        return true;
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
