#ifndef LIBFLATARRAY_EXAMPLES_LBM_UTIL_H
#define LIBFLATARRAY_EXAMPLES_LBM_UTIL_H

/**
 * Copyright 2013-2015 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <stdexcept>

long long time_usec()
{
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    return now.time_of_day().total_microseconds();
}

void check_cuda_error()
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(error) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

class benchmark
{
public:
    virtual ~benchmark()
    {}

    void evaluate()
    {
        for (int dim = 32; dim <= 160; dim += 4) {
            run(dim);
        }
    }

    void run(int dim)
    {
        int repeats = 10;
	if (dim <= 96) {
            repeats *= 10;
        }

        long long useconds = exec(dim, repeats);

        double updates = 1.0 * gridSize(dim) * repeats;
        double seconds = useconds * 10e-6;
        double glups = 10e-9 * updates / seconds;

        std::cout << std::setiosflags(std::ios::left);
        std::cout << std::setw(24) << name() << " ; "
                  << std::setw( 3) << dim << " ; "
                  << std::setw( 9) << glups << " GLUPS\n";
    }

protected:
    virtual long long exec(int dim, int repeats) = 0;
    virtual std::string name() = 0;
    virtual size_t gridSize(int dim) = 0;
};

class benchmark_lbm_cuda : public benchmark
{
protected:
    long long exec(int dim, int repeats)
    {
        dim3 dimBlock;
        dim3 dimGrid;
        gen_dims(&dimBlock, &dimGrid, dim);

        return cudaExec(dim, dimBlock, dimGrid, repeats);
    }

    virtual size_t gridSize(int dim)
    {
        dim3 dimBlock;
        dim3 dimGrid;
        gen_dims(&dimBlock, &dimGrid, dim);

        return dimGrid.x * dimBlock.x * dimGrid.y * dimBlock.y * (256 - 4);
    }

    virtual long long cudaExec(int dim, dim3 dimBlock, dim3 dimGrid, int repeats) = 0;

    void gen_dims(dim3 *dimBlock, dim3 *dimGrid, int dim)
    {
        int blockWidth = 1;
        for (; blockWidth <= dim; blockWidth *= 2) {
        }
        blockWidth /= 2;
        blockWidth = std::min(256, blockWidth);
        *dimBlock = dim3(blockWidth, 2, 1);
        *dimGrid = dim3(dim / dimBlock->x, dim / dimBlock->y, 1);
    }
};

class benchmark_lbm_cuda_basic : public benchmark_lbm_cuda
{
protected:
    virtual ~benchmark_lbm_cuda_basic()
    {}

    virtual long long cudaExec(int dim, dim3 dimBlock, dim3 dimGrid, int repeats)
    {
        int size = dim * dim * (256 + 64) * 20;
        int bytesize = size * sizeof(double);
        std::vector<double> grid(size, 4711);

        double *devGridOld;
        double *devGridNew;
        cudaMalloc(&devGridOld, bytesize);
        cudaMalloc(&devGridNew, bytesize);
        check_cuda_error();

        cudaMemcpy(devGridOld, &grid[0], bytesize, cudaMemcpyHostToDevice);
        cudaMemcpy(devGridNew, &grid[0], bytesize, cudaMemcpyHostToDevice);
        check_cuda_error();

        cudaDeviceSynchronize();
        long long t_start = time_usec();

        for (int t = 0; t < repeats; ++t) {
            update(dimGrid, dimBlock, dim, dim, 256, devGridOld, devGridNew);
            std::swap(devGridOld, devGridNew);
        }

        cudaDeviceSynchronize();
        long long t_end = time_usec();
        check_cuda_error();

        cudaMemcpy(&grid[0], devGridNew, bytesize, cudaMemcpyDeviceToHost);
        cudaFree(devGridOld);
        cudaFree(devGridNew);
        check_cuda_error();

        long long time = t_end - t_start;
        return time;
    }

    virtual void update(dim3 dimGrid, dim3 dimBlock, int dimX, int dimY, int dimZ, double *devGridOld, double *devGridNew) = 0;

};

#endif
