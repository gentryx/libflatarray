/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/detail/lightweight_test.hpp>
#include <libflatarray/flat_array.hpp>

#include "test.hpp"

class HeatedGameOfLifeCell
{
public:
    inline
    explicit HeatedGameOfLifeCell(double temperature=0.0, bool alive=false) :
        temperature(temperature),
        alive(alive)
    {}

    inline bool operator==(const HeatedGameOfLifeCell& other) const
    {
        return
            (temperature == other.temperature) &&
            (alive == other.alive);
    }

    inline bool operator!=(const HeatedGameOfLifeCell& other) const
    {
        return !(*this == other);
    }

    double temperature;
    bool alive;
};

LIBFLATARRAY_REGISTER_SOA(HeatedGameOfLifeCell, ((double)(temperature))((bool)(alive)))

namespace LibFlatArray {

ADD_TEST(TestCUDASetGet)
{
    soa_grid<HeatedGameOfLifeCell, cuda_allocator<char>, true> grid(20, 10, 5);
}

}

int main(int argc, char **argv)
{
    return 0;
}
