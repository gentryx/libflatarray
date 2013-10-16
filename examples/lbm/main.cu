/**
 * Copyright 2013 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <cuda.h>

#include "cell.h"
#include "util.h"
#include "update_lbm_classic.h"
#include "update_lbm_object_oriented.h"
#include "update_lbm_cuda_flat_array.h"

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " CUDA_DEVICE\n";
        return 1;
    }

    std::stringstream s;
    s << argv[1];
    int cudaDevice;
    s >> cudaDevice;
    cudaSetDevice(cudaDevice);

    std::cout << "# test name              ; dim ; performance\n";
    benchmark_lbm_cuda_object_oriented().evaluate();
    benchmark_lbm_cuda_classic().evaluate();
    benchmark_lbm_cuda_flat_array().evaluate();

    return 0;
}
