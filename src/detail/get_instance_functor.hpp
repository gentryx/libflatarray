/**
 * Copyright 2014-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_GET_INSTANCE_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_GET_INSTANCE_FUNCTOR_HPP

#include <libflatarray/soa_accessor.hpp>

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * This helper class is used to retrieve objects from the SoA storage
 * with the help of an accessor.
 */
template<typename CELL, bool USE_CUDA_FUNCTORS = false>
class get_instance_functor
{
public:
    get_instance_functor(
        CELL *target,
        long x,
        long y,
        long z,
        long count) :
        target(target),
        x(x),
        y(y),
        z(z),
        count(count)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        typedef soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor_type;
        accessor.index = accessor_type::gen_index(x, y, z);
        CELL *cursor = target;

        for (long i = 0; i < count; ++i) {
            accessor >> *cursor;
            ++cursor;
            ++accessor.index;
        }
    }

private:
    CELL *target;
    long x;
    long y;
    long z;
    long count;
};

#ifdef LIBFLATARRAY_WITH_CUDA
#ifdef __CUDACC__

template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
__global__
void get_kernel(CELL *target, const char *source, long count, long x, long y, long z)
{
    long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= count) {
        return;
    }

    typedef const_soa_accessor_light<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor_type;

    long index = accessor_type::gen_index(x + offset, y, z);
    accessor_type accessor(source, index);

    accessor >> target[offset];
}

/**
 * Specialization for CUDA
 */
template<typename CELL>
class get_instance_functor<CELL, true>
{
public:
    get_instance_functor(
        CELL *target,
        long x,
        long y,
        long z,
        long count) :
        target(target),
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

        get_kernel<CELL, DIM_X, DIM_Y, DIM_Z, INDEX><<<grid_dim, block_dim>>>(
            target,
            accessor.data(),
            count,
            x,
            y,
            z);
    }

private:
    CELL *target;
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
