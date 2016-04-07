/**
 * Copyright 2014, 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SAVE_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_SAVE_FUNCTOR_HPP

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * Same as load_functor, but the other way around.
 */
template<typename CELL, bool USE_CUDA_FUNCTORS = false>
class save_functor
{
public:
    save_functor(
        std::size_t x,
        std::size_t y,
        std::size_t z,
        char *target,
        long count) :
        target(target),
        count(count),
        x(x),
        y(y),
        z(z)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        typedef soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor_type;
        accessor.index = accessor_type::gen_index(x, y, z);
        accessor.save(target, count);
    }

private:
    char *target;
    long count;
    long x;
    long y;
    long z;
};

#ifdef LIBFLATARRAY_WITH_CUDA
#ifdef __CUDACC__

template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
__global__
void save_kernel(const char *source, char *target, long count, long x, long y, long z)
{
    long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= count) {
        return;
    }

    typedef const_soa_accessor_light<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor_type;

    long index = accessor_type::gen_index(x, y, z);
    accessor_type accessor(source, index);

    // data is assumed to be stored with stride "count":
    accessor.save(target, 1, offset, count);
}

/**
 * Specialization for CUDA
 */
template<typename CELL>
class save_functor<CELL, true>
{
public:
    save_functor(
        std::size_t x,
        std::size_t y,
        std::size_t z,
        char *target,
        std::size_t count) :
        target(target),
        count(count),
        x(x),
        y(y),
        z(z)
    {
    }

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        dim3 grid_dim;
        dim3 block_dim;
        generate_launch_config()(&grid_dim, &block_dim, count, 1, 1);

        save_kernel<CELL, DIM_X, DIM_Y, DIM_Z, INDEX><<<grid_dim, block_dim>>>(
            accessor.get_data(),
            target,
            count,
            x,
            y,
            z);
    }

private:
    char *target;
    std::size_t count;
    std::size_t x;
    std::size_t y;
    std::size_t z;

};

#endif
#endif

}

}

}

#endif
