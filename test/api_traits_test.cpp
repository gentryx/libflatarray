/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/detail/lightweight_test.hpp>
#include <iostream>
#include <vector>
#include <libflatarray/api_traits.hpp>
#include <libflatarray/soa_grid.hpp>
#include <libflatarray/macros.hpp>

#include "test.hpp"

using namespace LibFlatArray;

class CellDefaultSizes
{
public:
    double memberA;
    double memberB;
    double memberC;
};

class CellDefault2DSizes
{
public:
    class API : public api_traits::has_default_2d_sizes
    {};

    double memberA;
    double memberB;
    double memberC;
};

class CellDefault3DSizes
{
public:
    class API : public api_traits::has_default_3d_sizes
    {};

    double memberA;
    double memberB;
    double memberC;
};

class CellCustomSizes
{
public:
    class API
    {
    public:
        LIBFLATARRAY_CUSTOM_SIZES(
            (10)(20)(30),
            (11)(44)(88)(99),
            (47)(53))
    };

    double memberA;
    double memberB;
    double memberC;
};

class CellCustomSizesUniform
{
public:
    class API
    {
    public:
        LIBFLATARRAY_CUSTOM_SIZES_3D_UNIFORM(
            (10)(20)(30)(40)(50)(60)(70)(80)(90))
    };

    double memberA;
    double memberB;
    double memberC;
};

LIBFLATARRAY_REGISTER_SOA(CellDefaultSizes,       ((double)(memberA))((double)(memberB))((double)(memberC)))
LIBFLATARRAY_REGISTER_SOA(CellDefault2DSizes,     ((double)(memberA))((double)(memberB))((double)(memberC)))
LIBFLATARRAY_REGISTER_SOA(CellDefault3DSizes,     ((double)(memberA))((double)(memberB))((double)(memberC)))
LIBFLATARRAY_REGISTER_SOA(CellCustomSizes,        ((double)(memberA))((double)(memberB))((double)(memberC)))
LIBFLATARRAY_REGISTER_SOA(CellCustomSizesUniform, ((double)(memberA))((double)(memberB))((double)(memberC)))

class TestFunctor
{
public:
    explicit TestFunctor(std::vector<long> *report) :
        report(report)
    {}

    template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor)
    {
        report->push_back(DIM_X);
        report->push_back(DIM_Y);
        report->push_back(DIM_Z);
        report->push_back(INDEX);
    }

    template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(const_soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor)
    {
            throw std::logic_error("this should not have been called");
    }

private:
    std::vector<long> *report;
};

ADD_TEST(TestSelectSizesDefault)
{
    char data[1024 * 1024];
    std::vector<long> actual;
    std::vector<long> expected;
    TestFunctor functor(&actual);
    typedef api_traits::select_sizes<CellDefaultSizes> selector;

    selector()(data, functor, 10, 20, 30);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 30, 30, 30);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 32, 32, 32);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 40, 32, 32);
    expected.push_back(64);
    expected.push_back(64);
    expected.push_back(64);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 40, 100, 32);
    expected.push_back(128);
    expected.push_back(128);
    expected.push_back(128);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();
}

ADD_TEST(TestSelectSizesDefault2D)
{
    char data[1024 * 1024];
    std::vector<long> actual;
    std::vector<long> expected;
    TestFunctor functor(&actual);
    typedef api_traits::select_sizes<CellDefault2DSizes> selector;

    selector()(data, functor, 10, 20, 1);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(1);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 30, 30, 1);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(1);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 32, 32, 1);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(1);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 40, 32, 1);
    expected.push_back(64);
    expected.push_back(32);
    expected.push_back(1);
    expected.push_back(0);
    std::cout << actual[0] << ", " << actual[1] << ", " << actual[2] << "\n";
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 65, 32, 1);
    expected.push_back(128);
    expected.push_back(32);
    expected.push_back(1);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 32, 33, 1);
    expected.push_back(32);
    expected.push_back(64);
    expected.push_back(1);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 66, 66, 1);
    expected.push_back(128);
    expected.push_back(128);
    expected.push_back(1);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();
}

ADD_TEST(TestSelectSizesDefault3D)
{
    char data[1024 * 1024];
    std::vector<long> actual;
    std::vector<long> expected;
    TestFunctor functor(&actual);
    typedef api_traits::select_sizes<CellDefault3DSizes> selector;

    selector()(data, functor, 10, 20, 30);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 30, 30, 30);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 32, 32, 32);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 40, 32, 32);
    expected.push_back(64);
    expected.push_back(32);
    expected.push_back(32);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 40, 250, 32);
    expected.push_back(64);
    expected.push_back(256);
    expected.push_back(32);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 40, 250, 70);
    expected.push_back(64);
    expected.push_back(256);
    expected.push_back(128);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();
}

ADD_TEST(TestSelectSizesCustom)
{
    char data[1024 * 1024];
    std::vector<long> actual;
    std::vector<long> expected;
    TestFunctor functor(&actual);
    typedef api_traits::select_sizes<CellCustomSizes> selector;

    selector()(data, functor, 10, 10, 30);
    expected.push_back(10);
    expected.push_back(11);
    expected.push_back(47);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    // scale x-axis
    selector()(data, functor, 11, 20, 30);
    expected.push_back(20);
    expected.push_back(44);
    expected.push_back(47);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 20, 20, 30);
    expected.push_back(20);
    expected.push_back(44);
    expected.push_back(47);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 21, 20, 30);
    expected.push_back(30);
    expected.push_back(44);
    expected.push_back(47);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 30, 20, 30);
    expected.push_back(30);
    expected.push_back(44);
    expected.push_back(47);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    // scale y-axis
    selector()(data, functor, 10, 20, 30);
    expected.push_back(10);
    expected.push_back(44);
    expected.push_back(47);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 10, 80, 30);
    expected.push_back(10);
    expected.push_back(88);
    expected.push_back(47);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 10, 90, 30);
    expected.push_back(10);
    expected.push_back(99);
    expected.push_back(47);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    // scale z-axis
    selector()(data, functor, 10, 10, 50);
    expected.push_back(10);
    expected.push_back(11);
    expected.push_back(53);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();
}

ADD_TEST(TestSelectSizesCustomUniform)
{
    char data[1024 * 1024];
    std::vector<long> actual;
    std::vector<long> expected;
    TestFunctor functor(&actual);
    typedef api_traits::select_sizes<CellCustomSizesUniform> selector;

    selector()(data, functor, 10, 10, 30);
    expected.push_back(30);
    expected.push_back(30);
    expected.push_back(30);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    // scale x-axis
    selector()(data, functor, 40, 20, 30);
    expected.push_back(40);
    expected.push_back(40);
    expected.push_back(40);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 55, 20, 30);
    expected.push_back(60);
    expected.push_back(60);
    expected.push_back(60);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 90, 20, 30);
    expected.push_back(90);
    expected.push_back(90);
    expected.push_back(90);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    // scale y-axis
    selector()(data, functor, 10, 20, 30);
    expected.push_back(30);
    expected.push_back(30);
    expected.push_back(30);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 10, 80, 30);
    expected.push_back(80);
    expected.push_back(80);
    expected.push_back(80);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 10, 90, 30);
    expected.push_back(90);
    expected.push_back(90);
    expected.push_back(90);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    // scale z-axis
    selector()(data, functor, 10, 10, 50);
    expected.push_back(50);
    expected.push_back(50);
    expected.push_back(50);
    expected.push_back(0);
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();
}

int main(int argc, char **argv)
{
    return 0;
}
