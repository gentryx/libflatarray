/**
 * Copyright 2016-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <libflatarray/preprocessor.hpp>

// globally disable some warnings with MSVC, that are issued not for a
// specific header, but rather for the interaction of system headers
// and LibFlatArray source:
#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif

// Don't warn about these functions being stripped from an executable
// as they're not being used, that's actually expected behavior.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <iostream>
#include <vector>

#include "test.hpp"

#define LIST_A
#define LIST_B (10)(20)(30)(40)(50)
#define LIST_C (60)

#define LIST_D LIBFLATARRAY_DEQUEUE(LIST_B)
#define LIST_E LIBFLATARRAY_DEQUEUE(LIST_C)

#define LAMBDA(INDEX, STANDARD_ARG, ITERATOR) vec[ITERATOR] = (INDEX + STANDARD_ARG + ITERATOR);

// Don't warn about the conditional expressions being constant, that's
// intentional here:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4127 )
#endif

ADD_TEST(TestElem)
{
    BOOST_TEST(LIBFLATARRAY_ELEM(0, LIST_B) == 10);
    BOOST_TEST(LIBFLATARRAY_ELEM(1, LIST_B) == 20);
    BOOST_TEST(LIBFLATARRAY_ELEM(2, LIST_B) == 30);
    BOOST_TEST(LIBFLATARRAY_ELEM(3, LIST_B) == 40);
    BOOST_TEST(LIBFLATARRAY_ELEM(4, LIST_B) == 50);

    BOOST_TEST(LIBFLATARRAY_ELEM(0, LIST_C) == 60);
}

ADD_TEST(TestSize)
{
    BOOST_TEST(LIBFLATARRAY_SIZE(LIST_A) == 0);
    BOOST_TEST(LIBFLATARRAY_SIZE(LIST_B) == 5);
    BOOST_TEST(LIBFLATARRAY_SIZE(LIST_C) == 1);
}

ADD_TEST(TestForEach)
{
    std::vector<int> vec(60, 0);
    LIBFLATARRAY_FOR_EACH(LAMBDA, 100, LIST_B);

    BOOST_TEST(vec[10] == (0 + 10 + 100));
    BOOST_TEST(vec[20] == (1 + 20 + 100));
    BOOST_TEST(vec[30] == (2 + 30 + 100));
    BOOST_TEST(vec[40] == (3 + 40 + 100));
    BOOST_TEST(vec[50] == (4 + 50 + 100));
}

ADD_TEST(TestDequeue)
{
    BOOST_TEST_EQ(LIBFLATARRAY_SIZE(LIST_D),     4);
    BOOST_TEST_EQ(LIBFLATARRAY_ELEM(0, LIST_D), 20);
    BOOST_TEST_EQ(LIBFLATARRAY_ELEM(1, LIST_D), 30);
    BOOST_TEST_EQ(LIBFLATARRAY_ELEM(2, LIST_D), 40);
    BOOST_TEST_EQ(LIBFLATARRAY_ELEM(3, LIST_D), 50);

    BOOST_TEST_EQ(LIBFLATARRAY_SIZE(LIST_E),     0);
}

ADD_TEST(TestIfShorter)
{
    bool a0 = LIBFLATARRAY_IF_SHORTER(LIST_A, 0, false, true);
    bool a1 = LIBFLATARRAY_IF_SHORTER(LIST_A, 1, true, false);
    bool a2 = LIBFLATARRAY_IF_SHORTER(LIST_A, 2, true, false);
    bool a3 = LIBFLATARRAY_IF_SHORTER(LIST_A, 3, true, false);
    bool a4 = LIBFLATARRAY_IF_SHORTER(LIST_A, 4, true, false);

    bool b0 = LIBFLATARRAY_IF_SHORTER(LIST_B, 0, false, true);
    bool b1 = LIBFLATARRAY_IF_SHORTER(LIST_B, 1, false, true);
    bool b2 = LIBFLATARRAY_IF_SHORTER(LIST_B, 2, false, true);
    bool b3 = LIBFLATARRAY_IF_SHORTER(LIST_B, 3, false, true);
    bool b4 = LIBFLATARRAY_IF_SHORTER(LIST_B, 4, false, true);
    bool b5 = LIBFLATARRAY_IF_SHORTER(LIST_B, 5, false, true);
    bool b6 = LIBFLATARRAY_IF_SHORTER(LIST_B, 6, true, false);
    bool b7 = LIBFLATARRAY_IF_SHORTER(LIST_B, 7, true, false);
    bool b8 = LIBFLATARRAY_IF_SHORTER(LIST_B, 8, true, false);
    bool b9 = LIBFLATARRAY_IF_SHORTER(LIST_B, 9, true, false);

    bool c0 = LIBFLATARRAY_IF_SHORTER(LIST_C, 0, false, true);
    bool c1 = LIBFLATARRAY_IF_SHORTER(LIST_C, 1, false, true);
    bool c2 = LIBFLATARRAY_IF_SHORTER(LIST_C, 2, true, false);
    bool c3 = LIBFLATARRAY_IF_SHORTER(LIST_C, 3, true, false);
    bool c4 = LIBFLATARRAY_IF_SHORTER(LIST_C, 4, true, false);

    BOOST_TEST(a0);
    BOOST_TEST(a1);
    BOOST_TEST(a2);
    BOOST_TEST(a3);
    BOOST_TEST(a4);

    BOOST_TEST(b0);
    BOOST_TEST(b1);
    BOOST_TEST(b2);
    BOOST_TEST(b3);
    BOOST_TEST(b4);
    BOOST_TEST(b5);
    BOOST_TEST(b6);
    BOOST_TEST(b7);
    BOOST_TEST(b8);
    BOOST_TEST(b9);

    BOOST_TEST(c0);
    BOOST_TEST(c1);
    BOOST_TEST(c2);
    BOOST_TEST(c3);
    BOOST_TEST(c4);
}

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

int main(int /* argc */, char** /* argv */)
{
    return 0;
}
