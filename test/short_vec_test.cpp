/**
 * Copyright 2013-2016 Andreas Schäfer
 * Copyright 2015 Di Xiao
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <libflatarray/config.h>
#include <libflatarray/aligned_allocator.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <libflatarray/macros.hpp>
#include <libflatarray/short_vec.hpp>
#include <stdexcept>
#include <vector>
#include <cstring>

#include "test.hpp"

namespace LibFlatArray {

template<typename CARGO, int ARITY>
void testImplementationReal()
{
    typedef short_vec<CARGO, ARITY> ShortVec;
    int numElements = ShortVec::ARITY * 10;

    std::vector<CARGO, aligned_allocator<CARGO, 64> > vec1(numElements);
    std::vector<CARGO, aligned_allocator<CARGO, 64> > vec2(numElements, 4711);

    // init vec1:
    for (int i = 0; i < numElements; ++i) {
        vec1[i] = i + 0.1;
    }

    // test default c-tor:
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST(4711 == vec2[i]);
    }
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v;
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST(0 == vec2[i]);
    }

    // tests vector load/store:
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        TEST_REAL((i + 0.1), vec2[i]);
    }

    // tests scalar load, vector add:
    ShortVec w = vec1[0];

    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        &vec2[i] << (v + w);
    }
    for (int i = 0; i < numElements; ++i) {
        TEST_REAL((i + 0.2), vec2[i]);
    }

    // tests +=
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        v += w;
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        TEST_REAL((2 * i + 0.3), vec2[i]);
    }

    // test -
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        &vec2[i] << (v - w);
    }
    for (int i = 0; i < numElements; ++i) {
        TEST_REAL((-i - 0.2), vec2[i]);
    }

    // test -=
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        v -= w;
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        TEST_REAL((2 * i + 0.3), vec2[i]);
    }

    // test *
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        &vec2[i] << (v * w);
    }
    for (int i = 0; i < numElements; ++i) {
        double reference = ((i + 0.1) * (2 * i + 0.3));
        TEST_REAL(reference, vec2[i]);
    }

    // test *=
    for (int i = 0; i < numElements; ++i) {
        vec2[i] = i + 0.2;
    }
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        v *= w;
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        TEST_REAL((i + 0.1) * (i + 0.2), vec2[i]);
    }

    // test /
    for (int i = 0; i < numElements; ++i) {
        vec2[i] = i + 0.2;
    }
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        &vec2[i] << (v / w);
    }
    for (int i = 0; i < numElements; ++i) {
        // accept lower accuracy for estimated division, really low
        // accuracy accepted because of results from ARM NEON:
        TEST_REAL_ACCURACY((i + 0.1) / (i + 0.2), vec2[i], 0.0025);
    }

    // test /=
    for (int i = 0; i < numElements; ++i) {
        vec2[i] = i + 0.2;
    }
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        v /= w;
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        // here, too, lower accuracy is acceptable. As with divisions,
        // ARM NEON costs us an order of magnitude here compared to X86.
        TEST_REAL_ACCURACY((i + 0.1) / (i + 0.2), vec2[i], 0.0025);
    }

    // test sqrt()
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        &vec2[i] << sqrt(v);
    }
    for (int i = 0; i < numElements; ++i) {
        // lower accuracy, mainly for ARM NEON
        TEST_REAL_ACCURACY(std::sqrt(double(i + 0.1)), vec2[i], 0.0025);
    }

    // test "/ sqrt()"
    for (int i = 0; i < numElements; ++i) {
        vec2[i] = i + 0.2;
    }
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        &vec2[i] << w / sqrt(v);
    }
    for (int i = 0; i < numElements; ++i) {
        // the expression "foo / sqrt(bar)" will again result in an
        // estimated result for single precision floats, so lower accuracy is acceptable:
        TEST_REAL_ACCURACY((i + 0.2) / std::sqrt(double(i + 0.1)), vec2[i], 0.0035);
    }

    // test string conversion
    for (int i = 0; i < ShortVec::ARITY; ++i) {
        vec1[i] = i + 0.1;
    }
    ShortVec v(&vec1[0]);
    std::ostringstream buf1;
    buf1 << v;

    std::ostringstream buf2;
    buf2 << "[";
    for (int i = 0; i < (ShortVec::ARITY - 1); ++i) {
        buf2 << (i + 0.1) << ", ";
    }
    buf2 << (ShortVec::ARITY - 1 + 0.1) << "]";

    BOOST_TEST(buf1.str() == buf2.str());

    // test gather
    {
        CARGO array[ARITY * 10];
        std::vector<int, aligned_allocator<int, 64> > indices(ARITY);
        CARGO actual[ARITY];
        CARGO expected[ARITY];
        std::memset(array, '\0', sizeof(CARGO) * ARITY * 10);

        for (int i = 0; i < ARITY * 10; ++i) {
            if (i % 10 == 0) {
                array[i] = i * 0.75;
            }
        }

        for (int i = 0; i < ARITY; ++i) {
            indices[i] = i * 10;
            expected[i] = (i * 10) * 0.75;
        }

        ShortVec vec;
        vec.gather(array, &indices[0]);
        actual << vec;

        for (int i = 0; i < ARITY; ++i) {
            TEST_REAL_ACCURACY(actual[i], expected[i], 0.001);
        }
    }

#ifdef LIBFLATARRAY_WITH_CPP14
    // test gather via initializer_list
    {
        CARGO actual1[ARITY];
        CARGO actual2[ARITY];
        CARGO expected[ARITY];
        for (int i = 0; i < ARITY; ++i) {
            expected[i] = (i * 10) * 0.75;
        }

        // max: 32
        ShortVec vec1 = { 0.0, 7.5, 15.0, 22.50, 30.0, 37.5, 45.0, 52.5,
                          60.0, 67.5, 75.0, 82.5, 90.0, 97.5, 105.0, 112.5,
                          120.0, 127.5, 135.0, 142.5, 150.0, 157.5, 165.0, 172.5,
                          180.0, 187.5, 195.0, 202.5, 210.0, 217.5, 225.0, 232.5 };
        ShortVec vec2;
        vec2 = { 0.0, 7.5, 15.0, 22.50, 30.0, 37.5, 45.0, 52.5,
                 60.0, 67.5, 75.0, 82.5, 90.0, 97.5, 105.0, 112.5,
                 120.0, 127.5, 135.0, 142.5, 150.0, 157.5, 165.0, 172.5,
                 180.0, 187.5, 195.0, 202.5, 210.0, 217.5, 225.0, 232.5 };
        actual1 << vec1;
        actual2 << vec2;
        for (int i = 0; i < ARITY; ++i) {
            TEST_REAL_ACCURACY(actual1[i], expected[i], 0.001);
            TEST_REAL_ACCURACY(actual2[i], expected[i], 0.001);
        }
    }
#endif

    // test scatter
    {
        ShortVec vec;
        CARGO array[ARITY * 10];
        CARGO expected[ARITY * 10];
        std::vector<int, aligned_allocator<int, 64> > indices(ARITY);
        std::memset(array,    '\0', sizeof(CARGO) * ARITY * 10);
        std::memset(expected, '\0', sizeof(CARGO) * ARITY * 10);
        for (int i = 0; i < ARITY * 10; ++i) {
            if (i % 10 == 0) {
                expected[i] = i * 0.75;
            }
        }
        for (int i = 0; i < ARITY; ++i) {
            indices[i] = i * 10;
        }

        vec.gather(expected, &indices[0]);
        vec.scatter(array, &indices[0]);
        for (int i = 0; i < ARITY * 10; ++i) {
            TEST_REAL_ACCURACY(array[i], expected[i], 0.001);
        }
    }

    // test non temporal stores
    {
        std::vector<CARGO, aligned_allocator<CARGO, 64> > array(ARITY);
        std::vector<CARGO, aligned_allocator<CARGO, 64> > expected(ARITY);

        for (int i = 0; i < ARITY; ++i) {
            expected[i] = 5.0;
        }
        ShortVec v1 = 5.0;
        v1.store_nt(&array[0]);
        for (int i = 0; i < ARITY; ++i) {
            TEST_REAL_ACCURACY(array[i], expected[i], 0.001);
        }

        for (int i = 0; i < ARITY; ++i) {
            expected[i] = i + 0.1;
        }
        ShortVec v2 = &expected[0];
        v2.store_nt(&array[0]);
        for (int i = 0; i < ARITY; ++i) {
            TEST_REAL_ACCURACY(array[i], expected[i], 0.001);
        }
    }

    // test aligned stores
    {
        std::vector<CARGO, aligned_allocator<CARGO, 64> > array(ARITY);
        std::vector<CARGO, aligned_allocator<CARGO, 64> > expected(ARITY);

        for (int i = 0; i < ARITY; ++i) {
            expected[i] = 5.0;
        }
        ShortVec v1 = 5.0;
        v1.store_aligned(&array[0]);
        for (int i = 0; i < ARITY; ++i) {
            TEST_REAL_ACCURACY(array[i], expected[i], 0.001);
        }

        for (int i = 0; i < ARITY; ++i) {
            expected[i] = i + 0.1;
        }
        ShortVec v2 = &expected[0];
        v2.store_aligned(&array[0]);
        for (int i = 0; i < ARITY; ++i) {
            TEST_REAL_ACCURACY(array[i], expected[i], 0.001);
        }
    }

    // test aligned loads
    {
        std::vector<CARGO, aligned_allocator<CARGO, 64> > array(ARITY);
        std::vector<CARGO, aligned_allocator<CARGO, 64> > expected(ARITY);

        for (int i = 0; i < ARITY; ++i) {
            array[i]    = i + 0.1;
            expected[i] = 0;
        }
        ShortVec v1;
        v1.load_aligned(&array[0]);
        v1.store(&expected[0]);
        for (int i = 0; i < ARITY; ++i) {
            TEST_REAL_ACCURACY(array[i], expected[i], 0.001);
        }
    }

    // test comparison
    {
        for (int test_value = 0; test_value <= ARITY; ++test_value) {
            std::vector<CARGO, aligned_allocator<CARGO, 64> > array1(ARITY);
            std::vector<CARGO, aligned_allocator<CARGO, 64> > array2(ARITY);
            std::vector<CARGO, aligned_allocator<CARGO, 64> > array3(ARITY);

            for (int i = 0; i < ARITY; ++i) {
                array1[i] = i;
                array2[i] = test_value;
            }

            ShortVec v1(&array1[0]);
            ShortVec v2(&array2[0]);
            ShortVec v3;

            // test operator<()
            v3 = (v1 < v2);
            &array3[0] << v3;

            for (int i = 0; i < ARITY; ++i) {
                if (i < test_value) {
                    BOOST_TEST(array3[i] != 0);
                } else {
                    BOOST_TEST(array3[i] == 0);
                }
            }

            // test reduction to bool:
            bool actual = v3.any();
            bool expected = (test_value > 0);
            BOOST_TEST_EQ(actual, expected);

            // test operator<=()
            v3 = (v1 <= v2);
            &array3[0] << v3;

            for (int i = 0; i < ARITY; ++i) {
                if (i <= test_value) {
                    BOOST_TEST(array3[i] != 0);
                } else {
                    BOOST_TEST(array3[i] == 0);
                }
            }

            // test operator==()
            v3 = (v1 == v2);
            &array3[0] << v3;

            for (int i = 0; i < ARITY; ++i) {
                if (i == test_value) {
                    BOOST_TEST(array3[i] != 0);
                } else {
                    BOOST_TEST(array3[i] == 0);
                }
            }

            // test reduction to bool:
            actual = v3.any();
            expected = (test_value < ARITY);
            BOOST_TEST_EQ(actual, expected);

            // test operator>()
            v3 = (v1 > v2);
            &array3[0] << v3;

            for (int i = 0; i < ARITY; ++i) {
                if (i > test_value) {
                    BOOST_TEST(array3[i] != 0);
                } else {
                    BOOST_TEST(array3[i] == 0);
                }
            }

            // test operator>=()
            v3 = (v1 >= v2);
            &array3[0] << v3;

            for (int i = 0; i < ARITY; ++i) {
                if (i >= test_value) {
                    BOOST_TEST(array3[i] != 0);
                } else {
                    BOOST_TEST(array3[i] == 0);
                }
            }

            // test reduction to bool, again:
            actual = v3.any();
            expected = (test_value < ARITY);
            BOOST_TEST_EQ(actual, expected);
        }
    }
}

template<typename CARGO, int ARITY>
void testImplementationInt()
{
    typedef short_vec<CARGO, ARITY> ShortVec;
    const int numElements = ShortVec::ARITY * 10;

    std::vector<CARGO> vec1(numElements);
    std::vector<CARGO> vec2(numElements, 4711);

    // init vec1:
    for (int i = 0; i < numElements; ++i) {
        vec1[i] = i;
    }

    // test default c-tor:
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST(4711 == vec2[i]);
    }
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v;
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST(0 == vec2[i]);
    }

    // tests vector load/store:
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST_EQ(i, vec2[i]);
    }

    // tests scalar load, vector add:
    ShortVec w = vec1[1];

    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        &vec2[i] << (v + w);
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST_EQ((i + 1), vec2[i]);
    }

    // tests +=
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        v += w;
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST_EQ((2 * i + 1), vec2[i]);
    }

    // test -
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        &vec2[i] << (v - w);
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST_EQ((-i - 1), vec2[i]);
    }

    // test -=
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        v -= w;
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST_EQ((2 * i + 1), vec2[i]);
    }

    // test *
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        &vec2[i] << (v * w);
    }
    for (int i = 0; i < numElements; ++i) {
        int reference = (i * (2 * i + 1));
        BOOST_TEST_EQ(reference, vec2[i]);
    }

    // test *=
    for (int i = 0; i < numElements; ++i) {
        vec2[i] = i + 2;
    }
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        v *= w;
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST_EQ(i * (i + 2), vec2[i]);
    }

    // test /
    for (int i = 0; i < numElements; ++i) {
        vec1[i] = 4 * (i + 1);
        vec2[i] = (i + 1);
    }
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        &vec2[i] << (v / w);
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST_EQ(4, vec2[i]);
    }

    // test /=
    for (int i = 0; i < numElements; ++i) {
        vec1[i] = 4 * (i + 1);
        vec2[i] = (i + 1);
    }
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        v /= w;
        &vec2[i] << v;
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST_EQ(4, vec2[i]);
    }

    // test sqrt()
    for (int i = 0; i < numElements; ++i) {
        vec1[i] = i * i;
    }
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        &vec2[i] << sqrt(v);
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST_EQ(i, vec2[i]);
    }

    // test "/ sqrt()"
    for (int i = 0; i < numElements; ++i) {
        vec1[i] = (i + 1) * (i + 1);
        vec2[i] = (i + 1) * 2;
    }
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        ShortVec w = &vec2[i];
        &vec2[i] << w / sqrt(v);
    }
    for (int i = 0; i < numElements; ++i) {
        BOOST_TEST_EQ(2, vec2[i]);
    }

    // test string conversion
    for (int i = 0; i < ShortVec::ARITY; ++i) {
        vec1[i] = i + 5;
    }
    ShortVec v(&vec1[0]);
    std::ostringstream buf1;
    buf1 << v;

    std::ostringstream buf2;
    buf2 << "[";
    for (int i = 0; i < (ShortVec::ARITY - 1); ++i) {
        buf2 << (i + 5) << ", ";
    }
    buf2 << (ShortVec::ARITY - 1 + 5) << "]";

    BOOST_TEST(buf1.str() == buf2.str());

    // test gather
    {
        CARGO array[ARITY * 10];
        std::vector<int,   aligned_allocator<int,   64> > indices(ARITY);
        std::vector<CARGO, aligned_allocator<CARGO, 64> >  actual(ARITY);
        CARGO expected[ARITY];
        std::memset(array, '\0', sizeof(CARGO) * ARITY * 10);

        for (int i = 0; i < ARITY * 10; ++i) {
            if (i % 10 == 0) {
                array[i] = i + 5;
            }
        }

        for (int i = 0; i < ARITY; ++i) {
            indices[i] = i * 10;
            expected[i] = (i * 10) + 5;
        }

        ShortVec vec;
        vec.gather(array, &indices[0]);
        actual.data() << vec;

        for (int i = 0; i < ARITY; ++i) {
            BOOST_TEST_EQ(actual[i], expected[i]);
        }
    }

#ifdef LIBFLATARRAY_WITH_CPP14
    // test gather via initializer_list
    {
        std::vector<CARGO, aligned_allocator<CARGO, 64> > actual1(ARITY);
        std::vector<CARGO, aligned_allocator<CARGO, 64> > actual2(ARITY);
        CARGO expected[ARITY];
        for (int i = 0; i < ARITY; ++i) {
            expected[i] = (i * 10) + 5;
        }

        // max: 32
        ShortVec vec1 = { 5, 15, 25, 35, 45, 55, 65, 75,
                          85, 95, 105, 115, 125, 135, 145, 155,
                          165, 175, 185, 195, 205, 215, 225, 235,
                          245, 255, 265, 275, 285, 295, 305, 315 };
        ShortVec vec2;
        vec2 = { 5, 15, 25, 35, 45, 55, 65, 75,
                 85, 95, 105, 115, 125, 135, 145, 155,
                 165, 175, 185, 195, 205, 215, 225, 235,
                 245, 255, 265, 275, 285, 295, 305, 315 };
        actual1.data() << vec1;
        actual2.data() << vec2;

        for (int i = 0; i < ARITY; ++i) {
            BOOST_TEST_EQ(actual1[i], expected[i]);
            BOOST_TEST_EQ(actual2[i], expected[i]);
        }
    }
#endif

    // test scatter
    {
        ShortVec vec;
        CARGO array[ARITY * 10];
        CARGO expected[ARITY * 10];
        std::vector<int, aligned_allocator<int, 64> > indices(ARITY);
        std::memset(array,    '\0', sizeof(CARGO) * ARITY * 10);
        std::memset(expected, '\0', sizeof(CARGO) * ARITY * 10);
        for (int i = 0; i < ARITY * 10; ++i) {
            if (i % 10 == 0) {
                expected[i] = i + 5;
            }
        }
        for (int i = 0; i < ARITY; ++i) {
            indices[i] = i * 10;
        }

        vec.gather(expected, &indices[0]);
        vec.scatter(array, &indices[0]);
        for (int i = 0; i < ARITY * 10; ++i) {
            BOOST_TEST_EQ(array[i], expected[i]);
        }
    }

    // test non temporal stores
    {
        std::vector<CARGO, aligned_allocator<CARGO, 64> > array(ARITY);
        std::vector<CARGO, aligned_allocator<CARGO, 64> > expected(ARITY);

        for (int i = 0; i < ARITY; ++i) {
            expected[i] = 5;
        }
        ShortVec v1 = 5;
        v1.store_nt(&array[0]);
        for (int i = 0; i < ARITY; ++i) {
            BOOST_TEST_EQ(array[i], expected[i]);
        }

        for (int i = 0; i < ARITY; ++i) {
            expected[i] = i;
        }
        ShortVec v2 = &expected[0];
        v2.store_nt(&array[0]);
        for (int i = 0; i < ARITY; ++i) {
            BOOST_TEST_EQ(array[i], expected[i]);
        }
    }

    // test aligned stores
    {
        std::vector<CARGO, aligned_allocator<CARGO, 64> > array(ARITY);
        std::vector<CARGO, aligned_allocator<CARGO, 64> > expected(ARITY);

        for (int i = 0; i < ARITY; ++i) {
            expected[i] = 5;
        }
        ShortVec v1 = 5;
        v1.store_aligned(&array[0]);
        for (int i = 0; i < ARITY; ++i) {
            BOOST_TEST_EQ(array[i], expected[i]);
        }

        for (int i = 0; i < ARITY; ++i) {
            expected[i] = i;
        }
        ShortVec v2 = &expected[0];
        v2.store_aligned(&array[0]);
        for (int i = 0; i < ARITY; ++i) {
            BOOST_TEST_EQ(array[i], expected[i]);
        }
    }

    // test aligned loads
    {
        std::vector<CARGO, aligned_allocator<CARGO, 64> > array(ARITY);
        std::vector<CARGO, aligned_allocator<CARGO, 64> > expected(ARITY);

        for (int i = 0; i < ARITY; ++i) {
            array[i]    = i;
            expected[i] = 0;
        }
        ShortVec v1;
        v1.load_aligned(&array[0]);
        v1.store(&expected[0]);
        for (int i = 0; i < ARITY; ++i) {
            BOOST_TEST_EQ(array[i], expected[i]);
        }
    }
}

ADD_TEST(TestBasic)
{
    testImplementationReal<double, 1>();
    testImplementationReal<double, 2>();
    testImplementationReal<double, 4>();
    testImplementationReal<double, 8>();
    testImplementationReal<double, 16>();
    testImplementationReal<double, 32>();

    testImplementationReal<float, 1>();
    testImplementationReal<float, 2>();
    testImplementationReal<float, 4>();
    testImplementationReal<float, 8>();
    testImplementationReal<float, 16>();
    testImplementationReal<float, 32>();

    testImplementationInt<int, 1>();
    testImplementationInt<int, 2>();
    testImplementationInt<int, 4>();
    testImplementationInt<int, 8>();
    testImplementationInt<int, 16>();
    testImplementationInt<int, 32>();
}

template<typename STRATEGY>
void checkForStrategy(STRATEGY, STRATEGY)
{}

ADD_TEST(TestImplementationStrategyDouble)
{
    // fixme: doc!
#define EXPECTED_TYPE short_vec_strategy::scalar
    checkForStrategy(short_vec<double, 1>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __SSE__
#define EXPECTED_TYPE short_vec_strategy::sse
#else
#define EXPECTED_TYPE short_vec_strategy::scalar
#endif
    checkForStrategy(short_vec<double, 2>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __VECTOR4DOUBLE___
#define EXPECTED_TYPE short_vec_strategy::qpx
#endif

#ifdef __SSE__
#  ifdef __AVX__
#  define EXPECTED_TYPE short_vec_strategy::avx
#  else
#  define EXPECTED_TYPE short_vec_strategy::sse
#  endif
#else
#define EXPECTED_TYPE short_vec_strategy::scalar
#endif
    checkForStrategy(short_vec<double, 4>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __MIC__
#define EXPECTED_TYPE short_vec_strategy::mic
#endif

#ifdef __VECTOR4DOUBLE___
#define EXPECTED_TYPE short_vec_strategy::qpx
#endif

#ifdef __SSE__
#  ifdef __AVX__
#    ifdef __AVX512F__
#    define EXPECTED_TYPE short_vec_strategy::avx512
#    else
#    define EXPECTED_TYPE short_vec_strategy::avx
#    endif
#  else
#  define EXPECTED_TYPE short_vec_strategy::sse
#  endif
#else
#define EXPECTED_TYPE short_vec_strategy::scalar
#endif
    checkForStrategy(short_vec<double, 8>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __MIC__
#define EXPECTED_TYPE short_vec_strategy::mic
#endif

#ifdef __VECTOR4DOUBLE___
#define EXPECTED_TYPE short_vec_strategy::qpx
#endif

#ifdef __AVX__
#  ifdef __AVX512F__
#  define EXPECTED_TYPE short_vec_strategy::avx512
#  else
#  define EXPECTED_TYPE short_vec_strategy::avx
#  endif
#else
#define EXPECTED_TYPE short_vec_strategy::scalar
#endif
    checkForStrategy(short_vec<double, 16>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __MIC__
#define EXPECTED_TYPE short_vec_strategy::mic
#endif

#ifdef __VECTOR4DOUBLE___
#define EXPECTED_TYPE short_vec_strategy::qpx
#endif

#ifdef __AVX512F__
#define EXPECTED_TYPE short_vec_strategy::avx512
#else
#  ifdef __AVX__
#    define EXPECTED_TYPE short_vec_strategy::avx
#  else
#    ifdef __SSE__
#      define EXPECTED_TYPE short_vec_strategy::sse
#    else
#      define EXPECTED_TYPE short_vec_strategy::scalar
#    endif
#  endif
#endif
    checkForStrategy(short_vec<double, 32>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE
}

ADD_TEST(TestImplementationStrategyFloat)
{
    // fixme: doc!
#define EXPECTED_TYPE short_vec_strategy::scalar
    checkForStrategy(short_vec<float, 1>::strategy(), EXPECTED_TYPE());
    checkForStrategy(short_vec<float, 2>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __SSE__
#define EXPECTED_TYPE short_vec_strategy::sse

#elif __ARM_NEON__
#define EXPECTED_TYPE short_vec_strategy::neon

#else
#define EXPECTED_TYPE short_vec_strategy::scalar
#endif
checkForStrategy(short_vec<float, 4>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __SSE__
#  ifdef __AVX__
#  define EXPECTED_TYPE short_vec_strategy::avx
#  else
#  define EXPECTED_TYPE short_vec_strategy::sse
#  endif
#endif

#ifdef __ARM_NEON__
#define EXPECTED_TYPE short_vec_strategy::neon
#endif

#ifndef EXPECTED_TYPE
#define EXPECTED_TYPE short_vec_strategy::scalar
#endif
    checkForStrategy(short_vec<float, 8>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __MIC__
#define EXPECTED_TYPE short_vec_strategy::mic
#endif
#ifdef __ARM_NEON__
#define EXPECTED_TYPE short_vec_strategy::neon
#endif

#ifdef __SSE__
#  ifdef __AVX__
#    ifdef __AVX512F__
#    define EXPECTED_TYPE short_vec_strategy::avx512
#    else
#    define EXPECTED_TYPE short_vec_strategy::avx
#    endif
#  else
#  define EXPECTED_TYPE short_vec_strategy::sse
#  endif
#endif

#ifndef EXPECTED_TYPE
#define EXPECTED_TYPE short_vec_strategy::scalar
#endif
    checkForStrategy(short_vec<float, 16>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __MIC__
#define EXPECTED_TYPE short_vec_strategy::mic
#endif
#ifdef __ARM_NEON__
#define EXPECTED_TYPE short_vec_strategy::neon
#endif

#ifdef __AVX__
#  ifdef __AVX512F__
#  define EXPECTED_TYPE short_vec_strategy::avx512
#  else
#  define EXPECTED_TYPE short_vec_strategy::avx
#  endif
#endif

#ifndef EXPECTED_TYPE
#define EXPECTED_TYPE short_vec_strategy::scalar
#endif
    checkForStrategy(short_vec<float, 32>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE
}

ADD_TEST(TestImplementationStrategyInt)
{
    // fixme: doc!
#define EXPECTED_TYPE short_vec_strategy::scalar
    checkForStrategy(short_vec<int, 1>::strategy(), EXPECTED_TYPE());
    checkForStrategy(short_vec<int, 2>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __SSE2__
#define EXPECTED_TYPE short_vec_strategy::sse
#else
#define EXPECTED_TYPE short_vec_strategy::scalar
#endif
    checkForStrategy(short_vec<int, 4>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __AVX2__
#define EXPECTED_TYPE short_vec_strategy::avx2
#else
#  ifdef __SSE2__
#  define EXPECTED_TYPE short_vec_strategy::sse
#  else
#  define EXPECTED_TYPE short_vec_strategy::scalar
#  endif
#endif
    checkForStrategy(short_vec<int, 8>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __AVX512F__
#define EXPECTED_TYPE short_vec_strategy::avx512
#else
#  ifdef __AVX2__
#  define EXPECTED_TYPE short_vec_strategy::avx2
#  else
#    ifdef __SSE2__
#    define EXPECTED_TYPE short_vec_strategy::sse
#    else
#    define EXPECTED_TYPE short_vec_strategy::scalar
#    endif
#  endif
#endif
    checkForStrategy(short_vec<int, 16>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE

#ifdef __AVX512F__
#define EXPECTED_TYPE short_vec_strategy::avx512
#else
#  ifdef __AVX2__
#  define EXPECTED_TYPE short_vec_strategy::avx2
#  else
#  define EXPECTED_TYPE short_vec_strategy::scalar
#  endif
#endif
    checkForStrategy(short_vec<int, 32>::strategy(), EXPECTED_TYPE());
#undef EXPECTED_TYPE
}

}

int main(int argc, char **argv)
{
    return 0;
}
