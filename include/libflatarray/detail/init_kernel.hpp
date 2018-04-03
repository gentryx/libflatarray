/**
 * Copyright 2018 Google
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_INIT_KERNEL_HPP
#define FLAT_ARRAY_DETAIL_INIT_KERNEL_HPP

#include <libflatarray/config.h>

namespace LibFlatArray {

namespace detail {

namespace flat_array {

#ifdef LIBFLATARRAY_WITH_CUDA
#ifdef __CUDACC__

template<typename CELL>
__global__
void init_kernel(CELL source, CELL *target, long count)
{
    long thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_index >= count) {
        return;
    }

    target[thread_index] = source;
}

#endif
#endif

}

}

}

#endif

