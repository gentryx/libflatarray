/**
 * Copyright 2014-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_LOAD_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_LOAD_FUNCTOR_HPP

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * The purpose of this functor is to load a row of cells which are
 * already prepackaged (in SoA form) in a raw data segment (i.e. all
 * members are stored in a consecutive array of the given length and
 * all arrays are concatenated).
 */
template<typename CELL, bool USE_CUDA_FUNCTORS = false>
class load_functor
{
public:
    load_functor(
        std::size_t x,
        std::size_t y,
        std::size_t z,
        const char *source,
        std::size_t count) :
        source(source),
        count(count),
        x(x),
        y(y),
        z(z)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        accessor.index = soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>::gen_index(x, y, z);
        accessor.load(source, count);
    }

private:
    const char *source;
    std::size_t count;
    std::size_t x;
    std::size_t y;
    std::size_t z;
};

#ifdef LIBFLATARRAY_WITH_CUDA
#ifdef __CUDACC__

template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
__global__
void load_kernel(const char *source, char *target, long count, long x, long y, long z)
{
    long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= count) {
        return;
    }

    typedef soa_accessor_light<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor_type;

    long index = accessor_type::gen_index(x, y, z);
    accessor_type accessor(target, index);

    // data is assumed to be stored with stride "count":
    accessor.load(source, 1, offset, count);
}

/**
 * Specialization for CUDA
 */
template<typename CELL>
class load_functor<CELL, true>
{
public:
    load_functor(
        std::size_t x,
        std::size_t y,
        std::size_t z,
        const char *source,
        std::size_t count) :
        source(source),
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

        load_kernel<CELL, DIM_X, DIM_Y, DIM_Z, INDEX><<<grid_dim, block_dim>>>(
            source,
            accessor.get_data(),
            count,
            x,
            y,
            z);
    }

private:
    const char *source;
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
