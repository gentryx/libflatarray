/**
 * Copyright 2013-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <libflatarray/aligned_allocator.hpp>
#include <vector>

#include "test.hpp"

using namespace LibFlatArray;

ADD_TEST(test_alignment_64)
{
    int *p = aligned_allocator<int,   64>().allocate(3);
    BOOST_TEST(0 == (long(p) %  64));
    aligned_allocator<int, 64>().deallocate(p, 3);
}

ADD_TEST(test_alignment_128)
{
    char *p = aligned_allocator<char, 128>().allocate(199);
    BOOST_TEST(0 == (long(p) % 128));
    aligned_allocator<char, 128>().deallocate(p, 199);
}

ADD_TEST(test_alignment_512)
{
    long *p = aligned_allocator<long, 512>().allocate(256);
    BOOST_TEST(0 == (long(p) % 512));
    aligned_allocator<long, 512>().deallocate(p, 256);
}

ADD_TEST(test_usage_with_std_vector)
{
    typedef std::vector<int, aligned_allocator<int, 64> > vec_type;
    vec_type vec(40, -1);

    BOOST_TEST(0 == (std::size_t(&vec[0])) % 64);

    for (vec_type::iterator i = vec.begin(); i != vec.end(); ++i) {
        BOOST_TEST(-1 == *i);
    }

    vec.resize(80);
    for (int i = 0; i < 80; ++i) {
        vec[i] = 4711 + i;
    }
    for (int i = 0; i < 80; ++i) {
        BOOST_TEST((4711 + i) == vec[i]);
    }

    vec.resize(0);
    for (int i = 0; i < 90; ++i) {
        vec.push_back(23 + i);
    }
    for (int i = 0; i < 90; ++i) {
        BOOST_TEST((23 + i) == vec[i]);
    }

    vec.resize(0);
    vec.reserve(95);
    for (int i = 0; i < 95; ++i) {
        vec.push_back(69 + i);
    }
    for (int i = 0; i < 95; ++i) {
        BOOST_TEST((69 + i) == vec[i]);
    }

}

int main(int argc, char **argv)
{
    return 0;
}
