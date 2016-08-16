/**
 * Copyright 2013-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef LIBFLATARRAY_EXAMPLES_LBM_UPDATE_LBM_CUDA_FLAT_ARRAY_H
#define LIBFLATARRAY_EXAMPLES_LBM_UPDATE_LBM_CUDA_FLAT_ARRAY_H

#include <libflatarray/soa_grid.hpp>

#include "util.h"
#include "cudalineupdatefunctorprototype.h"

class benchmark_lbm_cuda_flat_array : public benchmark_lbm_cuda
{
    virtual long long cudaExec(int dim, dim3 dimBlock, dim3 dimGrid, int repeats)
    {
        LibFlatArray::soa_grid<CellLBM> gridA(dim, dim, 256);
        LibFlatArray::soa_grid<CellLBM> gridB(dim, dim, 256);
        // fixme: init grid?

        char *dataA = gridA.data();
        char *dataB = gridB.data();

        char *buf;
        cudaMalloc(reinterpret_cast<void**>(&buf), gridA.byte_size());
        gridA.set_data(buf);
        cudaMalloc(reinterpret_cast<void**>(&buf), gridB.byte_size());
        gridB.set_data(buf);

        LibFlatArray::soa_grid<CellLBM> *gridOld = &gridA;
        LibFlatArray::soa_grid<CellLBM> *gridNew = &gridB;

        cudaDeviceSynchronize();
        long long t_start = time_usec();

        CudaLineUpdateFunctorPrototype<CellLBM> updater(dimBlock, dimGrid);

        for (int t = 0; t < repeats; ++t) {
            gridOld->callback(gridNew, updater);
            std::swap(gridOld, gridNew);
        }

        cudaDeviceSynchronize();
        long long t_end = time_usec();
        check_cuda_error();

        cudaFree(gridA.data());
        cudaFree(gridB.data());

        gridA.set_data(dataA);
        gridB.set_data(dataB);

        return t_end - t_start;
    }

    virtual std::string name()
    {
        return "lbm_cuda_flat_array";
    }
};

#endif
