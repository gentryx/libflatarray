/**
 * Copyright 2014-2016 Andreas Schäfer
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
template<typename CELL, typename ITERATOR, bool USE_CUDA_FUNCTORS = false>
class save_functor
{
public:
    save_functor(
        const ITERATOR& start,
        const ITERATOR& end,
        char *target,
        std::size_t count) :
        start(start),
        end(end),
        target(target),
        count(count)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        std::size_t offset = 0;

        for (ITERATOR i = start; i != end; ++i) {
            accessor.index() = soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>::gen_index(
                static_cast<long>(i->origin[0]),
                static_cast<long>(i->origin[1]),
                static_cast<long>(i->origin[2]));
            accessor.save(
                target,
                static_cast<std::size_t>(i->length()),
                offset,
                count);

            offset += i->length();
        }
    }

private:
    ITERATOR start;
    ITERATOR end;
    char *target;
    std::size_t count;
};

#ifdef LIBFLATARRAY_WITH_CUDA
#ifdef __CUDACC__

template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
__global__
void save_kernel(const char *source, char *target, long count, long stride, long x, long y, long z, long offset)
{
    long thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_index >= count) {
        return;
    }

    typedef const_soa_accessor_light<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor_type;

    long index = accessor_type::gen_index(x, y, z) + thread_index;
    accessor_type accessor(source, index);

    accessor.save(target, 1, offset + thread_index, stride);
}

/**
 * Specialization for CUDA
 */
template<typename CELL, typename ITERATOR>
class save_functor<CELL, ITERATOR, true>
{
public:
    save_functor(
        const ITERATOR& start,
        const ITERATOR& end,
        char *target,
        std::size_t count) :
        start(start),
        end(end),
        target(target),
        count(count)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor) const
    {
        std::size_t offset = 0;

        for (ITERATOR i = start; i != end; ++i) {
            dim3 grid_dim;
            dim3 block_dim;
            generate_launch_config()(&grid_dim, &block_dim, i->length(), 1, 1);

            save_kernel<CELL, DIM_X, DIM_Y, DIM_Z, INDEX><<<grid_dim, block_dim>>>(
                accessor.data(),
                target,
                i->length(),
                count,
                i->origin[0],
                i->origin[1],
                i->origin[2],
                offset);

            offset += i->length();
        }
    }

private:
    ITERATOR start;
    ITERATOR end;
    char *target;
    std::size_t count;

};

#endif
#endif

}

}

}

#endif
