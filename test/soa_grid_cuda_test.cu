/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/detail/lightweight_test.hpp>
#include <libflatarray/flat_array.hpp>
#include <map>

#include "test.hpp"

class ActiveElement
{
public:
    __host__
    __device__
    ActiveElement()
    {
        val += 100000;
    }

    __host__
    __device__
    ~ActiveElement()
    {
        val += 1000000;
    }

    inline bool operator==(ActiveElement other) const
    {
        return val == other.val;
    }

    int val;
};

class PassiveElement
{
public:
    inline bool operator==(PassiveElement other) const
    {
        return val == other.val;
    }

    int val;
};

class ConstructorDestructorTestCellActive
{
public:
    inline
    explicit ConstructorDestructorTestCellActive(double temperature=0.0, bool alive=false) :
        temperature(temperature),
        alive(alive)
    {}

    inline bool operator==(const ConstructorDestructorTestCellActive& other) const
    {
        return
            (temperature == other.temperature) &&
            (alive == other.alive) &&
            (element == other.element);
    }

    inline bool operator!=(const ConstructorDestructorTestCellActive& other) const
    {
        return !(*this == other);
    }

    double temperature;
    bool alive;
    ActiveElement element;
};

class ConstructorDestructorTestCellPassive
{
public:
    inline
    explicit ConstructorDestructorTestCellPassive(double temperature=0.0, bool alive=false) :
        temperature(temperature),
        alive(alive)
    {}

    inline bool operator==(const ConstructorDestructorTestCellPassive& other) const
    {
        return
            (temperature == other.temperature) &&
            (alive == other.alive) &&
            (element == other.element);
    }

    inline bool operator!=(const ConstructorDestructorTestCellPassive& other) const
    {
        return !(*this == other);
    }

    double temperature;
    bool alive;
    PassiveElement element;
};

LIBFLATARRAY_REGISTER_SOA(ConstructorDestructorTestCellActive,
                          ((double)(temperature))
                          ((ActiveElement)(element))
                          ((bool)(alive)) )

LIBFLATARRAY_REGISTER_SOA(ConstructorDestructorTestCellPassive,
                          ((double)(temperature))
                          ((PassiveElement)(element))
                          ((bool)(alive)) )

namespace LibFlatArray {

std::map<std::size_t, char*> allocation_cache;

/**
 * We fake allocation here to make sure our grids in the tests below
 * get the same pointers. We need this to be sure that we're working
 * on the same memory region with each.
 */
template<class T>
class fake_cuda_allocator
{
public:
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    pointer allocate(std::size_t n, const void* = 0)
    {
        if (allocation_cache[n] != 0) {
            return allocation_cache[n];
        }


        pointer ret = 0;
        cudaMalloc(&ret, n * sizeof(T));
        allocation_cache[n] = ret;
        return ret;
    }

    void deallocate(pointer p, std::size_t)
    {
        // intentionally left blank
    }

    void deallocate_all()
    {
        for (typename std::map<std::size_t, pointer>::iterator i = allocation_cache.begin(); i != allocation_cache.end(); ++i) {
            cudaFree(i->second);
            i->second = 0;
        }
    }
};

ADD_TEST(TestCUDAConstructionDestruction)
{
    char *data = 0;
    {
        // prep device memory with consecutive numbers:
        soa_grid<ConstructorDestructorTestCellPassive, fake_cuda_allocator<char>, true> grid(20, 10, 5);
        data = grid.get_data();

        soa_grid<ConstructorDestructorTestCellPassive> buffer(20, 10, 5);
        for (int z = 0; z < 5; ++z) {
            for (int y = 0; y < 10; ++y) {
                for (int x = 0; x < 20; ++x) {
                    ConstructorDestructorTestCellPassive cell((x + 1) * (y + 1), true);
                    cell.element.val = x + y * 20 + z * 20 * 10;
                    buffer.set(x, y, z, cell);

                    cell = buffer.get(x, y, z);
                }
            }
        }
        cudaMemcpy(grid.get_data(), buffer.get_data(), grid.byte_size(), cudaMemcpyHostToDevice);

    }
    {
        // ensure c-tor was run by checking increment on all elements:
        soa_grid<ConstructorDestructorTestCellActive,  fake_cuda_allocator<char>, true> grid(20, 10, 5);
        BOOST_TEST(data == grid.get_data());

        soa_grid<ConstructorDestructorTestCellPassive> buffer(20, 10, 5);
        cudaMemcpy(buffer.get_data(), grid.get_data(), grid.byte_size(), cudaMemcpyDeviceToHost);
        for (int z = 0; z < 5; ++z) {
            for (int y = 0; y < 10; ++y) {
                for (int x = 0; x < 20; ++x) {
                    ConstructorDestructorTestCellPassive cell = buffer.get(x, y, z);
                    int expected = x + y * 20 + z * 20 * 10 + 100000;

                    BOOST_TEST(cell.element.val == expected);
                    BOOST_TEST(cell.temperature == 0);
                    BOOST_TEST(cell.alive == false);
                }
            }
        }
    }
    {
        // ensure d-tor was run by checking increment on all elements:
        soa_grid<ConstructorDestructorTestCellPassive> buffer(20, 10, 5);
        cudaMemcpy(buffer.get_data(), data, buffer.byte_size(), cudaMemcpyDeviceToHost);
        for (int z = 0; z < 5; ++z) {
            for (int y = 0; y < 10; ++y) {
                for (int x = 0; x < 20; ++x) {
                    ConstructorDestructorTestCellPassive cell = buffer.get(x, y, z);
                    int expected = x + y * 20 + z * 20 * 10 + 1100000;

                    BOOST_TEST(cell.element.val == expected);
                    BOOST_TEST(cell.temperature == 0);
                    BOOST_TEST(cell.alive == false);
                }
            }
        }
    }

    fake_cuda_allocator<char>().deallocate_all();
}

ADD_TEST(TestCUDAGetSetSingleElements)
{
    soa_grid<ConstructorDestructorTestCellPassive, cuda_allocator<char>, true> grid(40, 13, 8);

    for (int z = 0; z < 8; ++z) {
        for (int y = 0; y < 13; ++y) {
            for (int x = 0; x < 40; ++x) {
                ConstructorDestructorTestCellPassive cell((x + 2) * (y + 2), true);
                cell.element.val = 10000 + x + y * 40 + z * 40 * 13;
                grid.set(x, y, z, cell);
            }
        }
    }

    for (int z = 0; z < 8; ++z) {
        for (int y = 0; y < 13; ++y) {
            for (int x = 0; x < 40; ++x) {
                ConstructorDestructorTestCellPassive cell = grid.get(x, y, z);

                int expected = 10000 + x + y * 40 + z * 40 * 13;
                BOOST_TEST(cell.element.val == expected);
                BOOST_TEST(cell.temperature == ((x + 2) * (y + 2)));
                BOOST_TEST(cell.alive       == true);
            }
        }
    }
}

ADD_TEST(TestCUDAGetSetMultipleElements)
{
    soa_grid<ConstructorDestructorTestCellPassive, cuda_allocator<char>, true> grid(35, 25, 15);

    for (int z = 0; z < 15; ++z) {
        for (int y = 0; y < 25; ++y) {
            std::vector<ConstructorDestructorTestCellPassive> cells(35);
            for (int x = 0; x < 35; ++x) {
                cells[x].alive = x % 2;
                cells[x].temperature = x * y * z;
                cells[x].element.val = 20000 + x + y * 35 + z * 35 * 25;
            }

            grid.set(0, y, z, cells.data(), 35);
        }
    }

    for (int z = 0; z < 15; ++z) {
        for (int y = 0; y < 25; ++y) {
            std::vector<ConstructorDestructorTestCellPassive> cells(35);
            grid.get(0, y, z, cells.data(), 35);

            for (int x = 0; x < 35; ++x) {
                int expected = 20000 + x + y * 35 + z * 35 * 25;

                BOOST_TEST(cells[x].element.val == expected);
                BOOST_TEST(cells[x].alive == (x % 2));
                BOOST_TEST(cells[x].temperature == (x * y * z));
            }
        }
    }
}

ADD_TEST(TestCUDALoadSaveElements)
{
    // fixme: rename all grids to host/device_grid
    soa_grid<ConstructorDestructorTestCellPassive> host_grid(21, 10, 9);
    for (int z = 0; z < 9; ++z) {
        for (int y = 0; y < 10; ++y) {
            for (int x = 0; x < 21; ++x) {
                ConstructorDestructorTestCellPassive cell;
                cell.alive = ((x % 3) == 0);
                cell.temperature = x * y * z * -1;
                cell.element.val = 30000 + x + y * 21 + z * 21 * 10;
                host_grid.set(x, y, z, cell);
            }
        }
    }

    std::vector<char> buffer(10 * (sizeof(int) + sizeof(double) + sizeof(bool)));
    host_grid.save(11, 9, 8, buffer.data(), 10);

    // fixme: get rid of cuda_array here, buffering should be done on the device
    cuda_array<char> array(buffer.data(), buffer.size());
    soa_grid<ConstructorDestructorTestCellPassive, cuda_allocator<char>, true> device_grid(31, 20, 19);
    device_grid.load(21, 19, 18, array.data(), 10);

    soa_grid<ConstructorDestructorTestCellPassive> host_grid2(31, 20, 19);
    cudaMemcpy(host_grid2.get_data(), device_grid.get_data(), device_grid.byte_size(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        ConstructorDestructorTestCellPassive cell = host_grid2.get(21 + i, 19, 18);

        bool expectedAlive = (((i + 11) % 3) == 0);
        double expectedTemperature = (11 + i) * 9 * 8 * -1;
        int expectedVal = 30000 + (11 + i) + 9 * 21 + 8 * 21 * 10;

        BOOST_TEST(cell.alive == expectedAlive);
        BOOST_TEST(cell.temperature == expectedTemperature);
        BOOST_TEST(cell.element.val == expectedVal);
    }

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(error) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

// fixme: need test with array member, too!
}

int main(int argc, char **argv)
{
    return 0;
}
