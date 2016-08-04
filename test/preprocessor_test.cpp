/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <iostream>
#include <libflatarray/preprocessor.hpp>
#include <boost/detail/lightweight_test.hpp>

#include "test.hpp"

#define LIST_A
#define LIST_B (10)(20)(30)(40)(50)
#define LIST_C (60)

ADD_TEST(TestSeqElem)
{
    BOOST_TEST(LIBFLATARRAY_PP_SEQ_ELEM(0, LIST_B) == 10);
    BOOST_TEST(LIBFLATARRAY_PP_SEQ_ELEM(1, LIST_B) == 20);
    BOOST_TEST(LIBFLATARRAY_PP_SEQ_ELEM(2, LIST_B) == 30);
    BOOST_TEST(LIBFLATARRAY_PP_SEQ_ELEM(3, LIST_B) == 40);
    BOOST_TEST(LIBFLATARRAY_PP_SEQ_ELEM(4, LIST_B) == 50);

    BOOST_TEST(LIBFLATARRAY_PP_SEQ_ELEM(0, LIST_C) == 60);
}

ADD_TEST(TestSeqSize)
{
    BOOST_TEST(LIBFLATARRAY_PP_SEQ_SIZE(LIST_A) == 0);
    BOOST_TEST(LIBFLATARRAY_PP_SEQ_SIZE(LIST_B) == 5);
    BOOST_TEST(LIBFLATARRAY_PP_SEQ_SIZE(LIST_C) == 1);
}

ADD_TEST(TestSeqForEach)
{
    // fixme
}

ADD_TEST(TestSeqDequeue)
{
    // fixme
}

ADD_TEST(TestIfShorter)
{
    // fixme
}

int main(int argc, char **argv)
{
    return 0;
}
