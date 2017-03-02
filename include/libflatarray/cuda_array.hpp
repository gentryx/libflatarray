/**
 * Copyright 2016-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_CUDA_ARRAY_HPP
#define FLAT_ARRAY_CUDA_ARRAY_HPP

#include <libflatarray/cuda_allocator.hpp>

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <vector>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#ifdef __CUDACC__

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <cuda.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibFlatArray {

/**
 * A CUDA-enabled counterpart to std::vector. Handles memory
 * allocation and data transfer (intra and inter GPU) on NVIDIA GPUs.
 * No default initialization is done though as this would involve
 * (possibly slow) mem copies to the device.
 */
template<typename ELEMENT_TYPE>
class cuda_array
{
public:
    explicit inline cuda_array(std::size_t size = 0) :
        my_size(size),
        my_capacity(size),
        data_pointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(size))
    {}

    inline cuda_array(std::size_t size, const ELEMENT_TYPE& defaultValue) :
        my_size(size),
        my_capacity(size),
        data_pointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(size))
    {
        for (std::size_t i = 0; i < size; ++i) {
            cudaMemcpy(data_pointer + i, &defaultValue, sizeof(ELEMENT_TYPE), cudaMemcpyHostToDevice);
        }
    }

    inline cuda_array(const cuda_array& array) :
        my_size(array.size()),
        my_capacity(array.size()),
        data_pointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(array.size()))
    {
        cudaMemcpy(data_pointer, array.data_pointer, byte_size(), cudaMemcpyDeviceToDevice);
    }

    inline cuda_array(const ELEMENT_TYPE *hostData, std::size_t size) :
        my_size(size),
        my_capacity(size),
        data_pointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(size))
    {
        cudaMemcpy(data_pointer, hostData, byte_size(), cudaMemcpyHostToDevice);
    }

    explicit inline cuda_array(const std::vector<ELEMENT_TYPE>& hostVector) :
        my_size(hostVector.size()),
        my_capacity(hostVector.size()),
        data_pointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(hostVector.size()))
    {
        cudaMemcpy(data_pointer, &hostVector.front(), byte_size(), cudaMemcpyHostToDevice);
    }

    inline ~cuda_array()
    {
        LibFlatArray::cuda_allocator<ELEMENT_TYPE>().deallocate(data_pointer, my_capacity);
    }

    inline void operator=(const cuda_array& array)
    {
        if (array.size() > my_capacity) {
            LibFlatArray::cuda_allocator<ELEMENT_TYPE>().deallocate(data_pointer, my_capacity);
            data_pointer = LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(array.size());

            my_capacity = array.size();
        }

        my_size = array.size();
        cudaMemcpy(data_pointer, array.data_pointer, byte_size(), cudaMemcpyDeviceToDevice);
    }

    inline std::size_t size() const
    {
        return my_size;
    }

    inline std::size_t capacity() const
    {
        return my_capacity;
    }

    inline void resize(std::size_t newSize)
    {
        resize_implementation(newSize, 0);
    }

    inline void resize(std::size_t newSize, const ELEMENT_TYPE& defaultElement)
    {
        resize_implementation(newSize, &defaultElement);
    }

    inline void reserve(std::size_t newCapacity)
    {
        if (newCapacity <= my_capacity) {
            return;
        }

        ELEMENT_TYPE *newData = LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(newCapacity);
        cudaMemcpy(newData, data_pointer, byte_size(), cudaMemcpyDeviceToDevice);
        LibFlatArray::cuda_allocator<ELEMENT_TYPE>().deallocate(data_pointer, my_capacity);

        data_pointer = newData;
        my_capacity = newCapacity;
    }

    inline void load(const ELEMENT_TYPE *hostData)
    {
        cudaMemcpy(data_pointer, hostData, byte_size(), cudaMemcpyHostToDevice);
    }

    inline void save(ELEMENT_TYPE *hostData) const
    {
        cudaMemcpy(hostData, data_pointer, byte_size(), cudaMemcpyDeviceToHost);
    }

    inline std::size_t byte_size() const
    {
        return my_size * sizeof(ELEMENT_TYPE);
    }

    __host__ __device__
    inline ELEMENT_TYPE *data()
    {
        return data_pointer;
    }

    __host__ __device__
    inline const ELEMENT_TYPE *data() const
    {
        return data_pointer;
    }

private:
    std::size_t my_size;
    std::size_t my_capacity;
    ELEMENT_TYPE *data_pointer;

    inline void resize_implementation(std::size_t newSize, const ELEMENT_TYPE *defaultElement = 0)
    {
        if (newSize <= my_capacity) {
            if (defaultElement != 0) {
                for (std::size_t i = my_size; i < newSize; ++i) {
                    cudaMemcpy(
                        data_pointer + i,
                        defaultElement,
                        sizeof(ELEMENT_TYPE),
                        cudaMemcpyHostToDevice);
                }
            }

            my_size = newSize;
            return;
        }

        ELEMENT_TYPE *newData = LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(newSize);
        cudaMemcpy(newData, data_pointer, byte_size(), cudaMemcpyDeviceToDevice);
        LibFlatArray::cuda_allocator<ELEMENT_TYPE>().deallocate(data_pointer, my_capacity);

        if (defaultElement != 0) {
            for (std::size_t i = my_size; i < newSize; ++i) {
                cudaMemcpy(
                    newData + i,
                    defaultElement,
                    sizeof(ELEMENT_TYPE),
                    cudaMemcpyHostToDevice);
            }
        }

        data_pointer = newData;
        my_size = newSize;
        my_capacity = newSize;
    }

};

}

#endif

#endif
