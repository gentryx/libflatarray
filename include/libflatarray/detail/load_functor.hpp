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
template<typename CELL, typename ITERATOR, bool USE_CUDA_FUNCTORS = false>
class load_functor
{
public:
    load_functor(
        const ITERATOR& start,
        const ITERATOR& end,
        const char *source,
        std::size_t count) :
        start(start),
        end(end),
        source(source),
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
            accessor.load(
                source,
                static_cast<std::size_t>(i->length()),
                offset,
                count);

            offset += i->length();
        }
    }

private:
    ITERATOR start;
    ITERATOR end;
    const char *source;
    std::size_t count;
};

#ifdef LIBFLATARRAY_WITH_CUDA
#ifdef __CUDACC__

template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
__global__
void load_kernel(const char *source, char *target, long count, long stride, long x, long y, long z, long offset)
{
    long thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_index >= count) {
        return;
    }

    typedef soa_accessor_light<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor_type;

    long index = accessor_type::gen_index(x, y, z) + thread_index;
    accessor_type accessor(target, index);

    accessor.load(source, 1, offset + thread_index, stride);
}

/**
 * Specialization for CUDA
 */
template<typename CELL, typename ITERATOR>
class load_functor<CELL, ITERATOR, true>
{
public:
    load_functor(
        const ITERATOR& start,
        const ITERATOR& end,
        const char *source,
        std::size_t count) :
        start(start),
        end(end),
        source(source),
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

            load_kernel<CELL, DIM_X, DIM_Y, DIM_Z, INDEX><<<grid_dim, block_dim>>>(
                source,
                accessor.data(),
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
    const char *source;
    std::size_t count;

};

#endif
#endif

}

}

}

#endif
