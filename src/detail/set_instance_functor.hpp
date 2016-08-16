/**
 * Copyright 2014-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SET_INSTANCE_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_SET_INSTANCE_FUNCTOR_HPP

#include <libflatarray/soa_accessor.hpp>

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * This helper class uses an accessor to push an object's members into
 * the SoA storage.
 */
template<typename CELL, bool USE_CUDA_FUNCTORS = false>
class set_instance_functor
{
public:
    set_instance_functor(
        const CELL *source,
        long x,
        long y,
        long z,
        long count) :
        source(source),
        x(x),
        y(y),
        z(z),
        count(count)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        accessor.index = soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>::gen_index(x, y, z);
        const CELL *cursor = source;

        for (std::size_t i = 0; i < count; ++i) {
            accessor << *cursor;
            ++cursor;
            ++accessor.index;
        }
    }

private:
    const CELL *source;
    std::size_t x;
    std::size_t y;
    std::size_t z;
    std::size_t count;
};

#ifdef LIBFLATARRAY_WITH_CUDA
#ifdef __CUDACC__

template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
__global__
void set_kernel(const CELL *source, char *target, long count, long x, long y, long z)
{
    long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= count) {
        return;
    }

    typedef soa_accessor_light<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor_type;

    long index = accessor_type::gen_index(x + offset, y, z);
    accessor_type accessor(target, index);

    accessor << source[offset];
}

/**
 * Specialization for CUDA
 */
template<typename CELL>
class set_instance_functor<CELL, true>
{
public:
    set_instance_functor(
        const CELL *source,
        long x,
        long y,
        long z,
        long count) :
        source(source),
        x(x),
        y(y),
        z(z),
        count(count)
    {
    }

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        dim3 grid_dim;
        dim3 block_dim;
        generate_launch_config()(&grid_dim, &block_dim, count, 1, 1);

        set_kernel<CELL, DIM_X, DIM_Y, DIM_Z, INDEX><<<grid_dim, block_dim>>>(
            source,
            accessor.data(),
            count,
            x,
            y,
            z);
    }

private:
    const CELL *source;
    std::size_t x;
    std::size_t y;
    std::size_t z;
    std::size_t count;

};

#endif
#endif

}

}

}

#endif
