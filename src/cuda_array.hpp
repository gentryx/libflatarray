/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_CUDA_ARRAY_HPP
#define FLAT_ARRAY_CUDA_ARRAY_HPP

#include <libflatarray/cuda_allocator.hpp>
#include <cuda.h>

namespace LibFlatArray {

/**
 * A CUDA-enabled counterpart to std::vector. Handles memory
 * allocation and data transfer (intra and inter GPU) on NVIDIA GPUs.
 * No default initialization is done though as this would involve
 * (possibly slow) mem copies to the device.
 */
template<typename ELEMENT_TYPE>
class CUDAArray
{
public:
    explicit inline CUDAArray(std::size_t size = 0) :
        mySize(size),
        myCapacity(size),
        dataPointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(size))
    {}

    inline CUDAArray(std::size_t size, const ELEMENT_TYPE& defaultValue) :
        mySize(size),
        myCapacity(size),
        dataPointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(size))
    {
        for (std::size_t i = 0; i < size; ++i) {
            cudaMemcpy(dataPointer + i, &defaultValue, sizeof(ELEMENT_TYPE), cudaMemcpyHostToDevice);
        }
    }

    inline CUDAArray(const CUDAArray& array) :
        mySize(array.size()),
        myCapacity(array.size()),
        dataPointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(array.size()))
    {
        cudaMemcpy(dataPointer, array.dataPointer, byteSize(), cudaMemcpyDeviceToDevice);
    }

    inline CUDAArray(const ELEMENT_TYPE *hostData, std::size_t size) :
        mySize(size),
        myCapacity(size),
        dataPointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(size))
    {
        cudaMemcpy(dataPointer, hostData, byteSize(), cudaMemcpyHostToDevice);
    }

    explicit inline CUDAArray(const std::vector<ELEMENT_TYPE>& hostVector) :
        mySize(hostVector.size()),
        myCapacity(hostVector.size()),
        dataPointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(hostVector.size()))
    {
        cudaMemcpy(dataPointer, &hostVector.front(), byteSize(), cudaMemcpyHostToDevice);
    }

    inline ~CUDAArray()
    {
        LibFlatArray::cuda_allocator<ELEMENT_TYPE>().deallocate(dataPointer, myCapacity);
    }

    inline void operator=(const CUDAArray& array)
    {
        if (array.size() > myCapacity) {
            LibFlatArray::cuda_allocator<ELEMENT_TYPE>().deallocate(dataPointer, myCapacity);
            dataPointer = LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(array.size());

            myCapacity = array.size();
        }

        mySize = array.size();
        cudaMemcpy(dataPointer, array.dataPointer, byteSize(), cudaMemcpyDeviceToDevice);
    }

    inline std::size_t size() const
    {
        return mySize;
    }

    inline std::size_t capacity() const
    {
        return myCapacity;
    }

    inline void resize(std::size_t newSize)
    {
        resizeImplementation(newSize, 0);
    }

    inline void resize(std::size_t newSize, const ELEMENT_TYPE& defaultElement)
    {
        resizeImplementation(newSize, &defaultElement);
    }

    inline void reserve(std::size_t newCapacity)
    {
        if (newCapacity <= myCapacity) {
            return;
        }

        ELEMENT_TYPE *newData = LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(newCapacity);
        cudaMemcpy(newData, dataPointer, byteSize(), cudaMemcpyDeviceToDevice);
        LibFlatArray::cuda_allocator<ELEMENT_TYPE>().deallocate(dataPointer, myCapacity);

        dataPointer = newData;
        myCapacity = newCapacity;
    }

    inline void load(const ELEMENT_TYPE *hostData)
    {
        cudaMemcpy(dataPointer, hostData, byteSize(), cudaMemcpyHostToDevice);
    }

    inline void save(ELEMENT_TYPE *hostData) const
    {
        cudaMemcpy(hostData, dataPointer, byteSize(), cudaMemcpyDeviceToHost);
    }

    inline std::size_t byteSize() const
    {
        return mySize * sizeof(ELEMENT_TYPE);
    }

    __host__ __device__
    inline ELEMENT_TYPE *data()
    {
        return dataPointer;
    }

    __host__ __device__
    inline const ELEMENT_TYPE *data() const
    {
        return dataPointer;
    }

private:
    std::size_t mySize;
    std::size_t myCapacity;
    ELEMENT_TYPE *dataPointer;

    inline void resizeImplementation(std::size_t newSize, const ELEMENT_TYPE *defaultElement = 0)
    {
        if (newSize <= myCapacity) {
            if (defaultElement != 0) {
                for (std::size_t i = mySize; i < newSize; ++i) {
                    cudaMemcpy(
                        dataPointer + i,
                        defaultElement,
                        sizeof(ELEMENT_TYPE),
                        cudaMemcpyHostToDevice);
                }
            }

            mySize = newSize;
            return;
        }

        ELEMENT_TYPE *newData = LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(newSize);
        cudaMemcpy(newData, dataPointer, byteSize(), cudaMemcpyDeviceToDevice);
        LibFlatArray::cuda_allocator<ELEMENT_TYPE>().deallocate(dataPointer, myCapacity);

        if (defaultElement != 0) {
            for (std::size_t i = mySize; i < newSize; ++i) {
                cudaMemcpy(
                    newData + i,
                    defaultElement,
                    sizeof(ELEMENT_TYPE),
                    cudaMemcpyHostToDevice);
            }
        }

        dataPointer = newData;
        mySize = newSize;
        myCapacity = newSize;
    }

};

}

#endif
