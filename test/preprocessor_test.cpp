/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <iostream>
#include <libflatarray/preprocessor.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <vector>

#include "test.hpp"

#define LIST_A
#define LIST_B (10)(20)(30)(40)(50)
#define LIST_C (60)

#define LAMBDA(INDEX, STANDARD_ARG, ITERATOR) vec[ITERATOR] = (INDEX + STANDARD_ARG + ITERATOR);

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
    std::vector<int> vec(60, 0);
    LIBFLATARRAY_PP_SEQ_FOR_EACH(LAMBDA, 100, LIST_B);

    BOOST_TEST(vec[10] == (0 + 10 + 100));
    BOOST_TEST(vec[20] == (1 + 20 + 100));
    BOOST_TEST(vec[30] == (2 + 30 + 100));
    BOOST_TEST(vec[40] == (3 + 40 + 100));
    BOOST_TEST(vec[50] == (4 + 50 + 100));
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
