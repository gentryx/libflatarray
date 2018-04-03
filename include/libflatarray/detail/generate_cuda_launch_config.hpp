/**
 * Copyright 2016 Andreas Schäfer
 * Copyright 2018 Google
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_GENERATE_CUDA_LAUNCH_CONFIG_HPP
#define FLAT_ARRAY_DETAIL_GENERATE_CUDA_LAUNCH_CONFIG_HPP

#include <libflatarray/config.h>

#ifdef LIBFLATARRAY_WITH_CUDA
#ifdef __CUDACC__

namespace LibFlatArray {

namespace detail {

namespace flat_array {

/**
 * Returns a somewhat sensible decomposition of the grid into thread
 * blocks for launching CUDA kernels.
 */
class generate_cuda_launch_config
{
public:
    void operator()(dim3 *grid_dim, dim3 *block_dim, int x, int y, int z)
    {
        if (y >= 4) {
            *block_dim = dim3(128, 4, 1);
        } else {
            *block_dim = dim3(512, 1, 1);
        }

        grid_dim->x = divide_and_round_up(x, block_dim->x);
        grid_dim->y = divide_and_round_up(y, block_dim->y);
        grid_dim->z = divide_and_round_up(z, block_dim->z);
    }

private:
    int divide_and_round_up(int i, int dividend)
    {
        int ret = i / dividend;
        if (i % dividend) {
            ret += 1;
        }

        return ret;
    }
};

}

}

}

#endif
#endif

#endif

