/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <libflatarray/flat_array.hpp>

#include "test.hpp"

class Particle
{
public:
    float pos[3];
    float velocity[3];
    int time_to_live;
};

LIBFLATARRAY_REGISTER_SOA(
    Particle,
    ((float)(pos)(3))
    ((float)(velocity)(3))
    ((int)(time_to_live)))

namespace LibFlatArray {

ADD_TEST(TestConstructor)
{
}

}

int main(int argc, char **argv)
{
    return 0;
}
