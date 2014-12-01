/**
 * Copyright 2013 - 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <cmath>
#include <boost/detail/lightweight_test.hpp>
#include <iostream>
#include <sstream>
#include <libflatarray/short_vec.hpp>
#include <stdexcept>
#include <vector>

#include "test.hpp"

namespace LibFlatArray {

template<typename CARGO, int ARITY>
void testImplementation()
{
    typedef short_vec<CARGO, ARITY> ShortVec;
    int numElements = ShortVec::ARITY * 10;

    std::vector<CARGO> vec1(numElements);
    std::vector<CARGO> vec2(numElements, 4711);

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
        // accept lower accuracy for estimated division:
        TEST_REAL_ACCURACY((i + 0.1) / (i + 0.2), vec2[i], 0.0002);
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
        // here, too, lower accuracy is acceptable.
        TEST_REAL_ACCURACY((i + 0.1) / (i + 0.2), vec2[i], 0.0002);
    }

    // test sqrt()
    for (int i = 0; i < (numElements - ShortVec::ARITY + 1); i += ShortVec::ARITY) {
        ShortVec v = &vec1[i];
        &vec2[i] << sqrt(v);
    }
    for (int i = 0; i < numElements; ++i) {
        TEST_REAL(std::sqrt(double(i + 0.1)), vec2[i]);
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
        TEST_REAL_ACCURACY((i + 0.2) / std::sqrt(double(i + 0.1)), vec2[i], 0.0003);
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
}

ADD_TEST(TestBasic)
{
    testImplementation<double, 1>();
    testImplementation<double, 2>();
    testImplementation<double, 4>();
    testImplementation<double, 8>();
    testImplementation<double, 16>();
    // testImplementation<double, 32>();

    testImplementation<float, 1>();
    testImplementation<float, 2>();
    testImplementation<float, 4>();
    testImplementation<float, 8>();
    testImplementation<float, 16>();
    testImplementation<float, 32>();
}

}


int main(int argc, char **argv)
{
    return 0;
}
