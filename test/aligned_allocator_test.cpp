/**
 * Copyright 2013 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/detail/lightweight_test.hpp>
#include <libflatarray/aligned_allocator.hpp>

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

int main(int argc, char **argv)
{
    return 0;
}
