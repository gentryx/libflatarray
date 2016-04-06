/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_STAGING_BUFFER_HPP
#define FLAT_ARRAY_DETAIL_STAGING_BUFFER_HPP

#include <libflatarray/cuda_array.hpp>

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * Dummy class which presents the same interface as cuda_array,
 * but won't actually buffer the data. Instead the pointers are
 * forwarded directly so no additional copies of the data need to
 * be made.
 */
template<typename CELL, bool ENABLE_CUDA = false>
class staging_buffer
{
public:
    void resize(std::size_t /* unused */)
    {
        // intentionally left blank
    }

    void load(const CELL *new_data)
    {
        data_pointer = const_cast<CELL*>(new_data);
    }

    void save(CELL *new_data) const
    {
        // intentionally left blank
    }

    const CELL *data() const
    {
        return data_pointer;
    }

    CELL *data()
    {
        return data_pointer;
    }

    void prep(CELL *new_data)
    {
        data_pointer = new_data;
    }
private:
    CELL *data_pointer;
};

#ifdef __CUDACC__

template<typename CELL>
class staging_buffer<CELL, true>
{
public:
   void resize(std::size_t n)
    {
        delegate.resize(n);
    }

    void load(const CELL *new_data)
    {
        delegate.load(new_data);
    }

    void save(CELL *new_data) const
    {
        delegate.save(new_data);
    }

    const CELL *data() const
    {
        return delegate.data();
    }

    CELL *data()
    {
        return delegate.data();
    }

    void prep(CELL *new_data)
    {
        // intentionally left blank
    }

private:
    cuda_array<CELL> delegate;
};

#endif

}

}

}

#endif

