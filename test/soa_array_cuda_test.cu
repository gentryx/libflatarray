/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <libflatarray/soa_array.hpp>
#include <libflatarray/macros.hpp>
#include <map>

#include "test.hpp"

class CellWithArrayMember
{
public:
    __host__
    __device__
    inline
    explicit CellWithArrayMember(int j = 0) :
        j(j)
    {
        i[0] = j + 1;
        i[1] = j + 2;
        i[2] = j + 3;

        x[0] = j + 0.4;
        x[1] = j + 0.5;
    }

    __host__
    __device__
    inline
    CellWithArrayMember(int newI[3], double newX[2], int j) :
        j(j)
    {
        i[0] = newI[0];
        i[1] = newI[1];
        i[1] = newI[2];

        x[0] = newX[0];
        x[1] = newX[1];
    }

    int i[3];
    int j;
    double x[2];
};

LIBFLATARRAY_REGISTER_SOA(CellWithArrayMember,
                          ((int)(i)(3))
                          ((int)(j))
                          ((double)(x)(2)) )


namespace LibFlatArray {

typedef soa_array<CellWithArrayMember, 1000> soa_array_type;

__global__
void test_insert(soa_array_type *array)
{
    int size = array->size();
    for (int i = 0; i < size; ++i) {
        CellWithArrayMember cell = (*array)[i];
        cell.i[0] += 10000;
        cell.i[1] += 20000;
        cell.i[2] += 30000;

        (*array) << cell;
    }
}

__global__
void test_modify(soa_array_type *array)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= array->size()) {
        return;
    }

    (*array)[index].i()[0] += index;
    (*array)[index].i()[1] -= index;
    (*array)[index].i()[2]  = 2011 + 2014;
}

ADD_TEST(TestCUDABasic)
{
    soa_array_type host_array;

    for (int i = 0; i < 100; ++i) {
        CellWithArrayMember cell;
        cell.i[0] = i;
        cell.i[1] = i + 1000;
        cell.i[2] = i + 2000;
        host_array << cell;
    }

    soa_array_type *device_array = 0;
    cudaMalloc(&device_array, sizeof(soa_array_type));
    cudaMemcpy(device_array, &host_array, sizeof(soa_array_type), cudaMemcpyHostToDevice);

    test_insert<<<1, 1>>>(device_array);
    cudaMemcpy(&host_array, device_array, sizeof(soa_array_type), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 100; ++i) {
        BOOST_TEST((i +     0) == host_array[i +   0].i()[0]);
        BOOST_TEST((i +  1000) == host_array[i +   0].i()[1]);
        BOOST_TEST((i +  2000) == host_array[i +   0].i()[2]);

        BOOST_TEST((i + 10000) == host_array[i + 100].i()[0]);
        BOOST_TEST((i + 21000) == host_array[i + 100].i()[1]);
        BOOST_TEST((i + 32000) == host_array[i + 100].i()[2]);
    }

    test_modify<<<7, 32>>>(device_array);
    cudaMemcpy(&host_array, device_array, sizeof(soa_array_type), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 100; ++i) {
        BOOST_TEST((i + i +     0) == host_array[i +   0].i()[0]);
        BOOST_TEST((0 +      1000) == host_array[i +   0].i()[1]);
        BOOST_TEST((         4025) == host_array[i +   0].i()[2]);

        BOOST_TEST((i + i + 10100) == host_array[i + 100].i()[0]);
        BOOST_TEST((0 +     20900) == host_array[i + 100].i()[1]);
        BOOST_TEST((         4025) == host_array[i + 100].i()[2]);
    }

    cudaFree(device_array);
}

}

int main(int argc, char **argv)
{
    return 0;
}
