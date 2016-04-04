/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/detail/lightweight_test.hpp>
#include <libflatarray/flat_array.hpp>

#include "test.hpp"

class ActiveElement
{
public:
    __host__
    __device__
    ActiveElement() :
        val(13)
    {}

    __host__
    __device__
    ~ActiveElement()
    {
        val = 31;
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

class ContructorDestructorTestCellActive
{
public:
    inline
    explicit ContructorDestructorTestCellActive(double temperature=0.0, bool alive=false) :
        temperature(temperature),
        alive(alive)
    {}

    inline bool operator==(const ContructorDestructorTestCellActive& other) const
    {
        return
            (temperature == other.temperature) &&
            (alive == other.alive) &&
            (element == other.element);
    }

    inline bool operator!=(const ContructorDestructorTestCellActive& other) const
    {
        return !(*this == other);
    }

    double temperature;
    bool alive;
    ActiveElement element;
};

class ContructorDestructorTestCellPassive
{
public:
    inline
    explicit ContructorDestructorTestCellPassive(double temperature=0.0, bool alive=false) :
        temperature(temperature),
        alive(alive)
    {}

    inline bool operator==(const ContructorDestructorTestCellPassive& other) const
    {
        return
            (temperature == other.temperature) &&
            (alive == other.alive) &&
            (element == other.element);
    }

    inline bool operator!=(const ContructorDestructorTestCellPassive& other) const
    {
        return !(*this == other);
    }

    double temperature;
    bool alive;
    PassiveElement element;
};

LIBFLATARRAY_REGISTER_SOA(ContructorDestructorTestCellActive,
                          ((double)(temperature))
                          ((bool)(alive))
                          ((ActiveElement)(element)) )

LIBFLATARRAY_REGISTER_SOA(ContructorDestructorTestCellPassive,
                          ((double)(temperature))
                          ((bool)(alive))
                          ((PassiveElement)(element)) )

namespace LibFlatArray {

ADD_TEST(TestCUDASetGet)
{
    soa_grid<ContructorDestructorTestCellActive, cuda_allocator<char>, true> grid(20, 10, 5);
}

}

int main(int argc, char **argv)
{
    return 0;
}
