/**
 * Copyright 2016-2017 Andreas Schäfer
 * Copyright 2018 Google
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_DESTROY_FUNCTOR_HPP
#define FLAT_ARRAY_DETAIL_DESTROY_FUNCTOR_HPP

#include <libflatarray/config.h>
#include <libflatarray/detail/generate_cuda_launch_config.hpp>

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * Will call the destructor on all grid cells, relies on the SoA
 * (Struct of Arrays) accessor to destroy a cell's members
 * individually.
 */
template<typename CELL, bool USE_CUDA_FUNCTORS = false>
class destroy_functor
{
public:
    destroy_functor(
        std::size_t dim_x,
        std::size_t dim_y,
        std::size_t dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        for (std::size_t z = 0; z < dim_z; ++z) {
            for (std::size_t y = 0; y < dim_y; ++y) {
                accessor.index() = long(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>::gen_index(0, y, z));

                for (std::size_t x = 0; x < dim_x; ++x) {
                    accessor.destroy_members();
                    ++accessor;
                }
            }
        }
    }

private:
    std::size_t dim_x;
    std::size_t dim_y;
    std::size_t dim_z;
};

#ifdef LIBFLATARRAY_WITH_CUDA
#ifdef __CUDACC__

template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
__global__
void destroy_kernel(char *data, long dim_x, long dim_y, long dim_z)
{
    long x = blockDim.x * blockIdx.x + threadIdx.x;
    long y = blockDim.y * blockIdx.y + threadIdx.y;
    long z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x) {
        return;
    }

    if (y >= dim_y) {
        return;
    }

    if (z >= dim_z) {
        return;
    }

    typedef soa_accessor_light<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor_type;

    long index = accessor_type::gen_index(x, y, z);
    accessor_type accessor(data, index);
    accessor.destroy_members();
}

/**
 * Specialization for CUDA
 */
template<typename CELL>
class destroy_functor<CELL, true>
{
public:
    destroy_functor(
        std::size_t dim_x,
        std::size_t dim_y,
        std::size_t dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        dim3 grid_dim;
        dim3 block_dim;
        generate_cuda_launch_config()(&grid_dim, &block_dim, dim_x, dim_y, dim_z);

        destroy_kernel<CELL, DIM_X, DIM_Y, DIM_Z, INDEX><<<grid_dim, block_dim>>>(accessor.data(), dim_x, dim_y, dim_z);
    }

private:
    std::size_t dim_x;
    std::size_t dim_y;
    std::size_t dim_z;
};

#endif
#endif

}

}

}

#endif

