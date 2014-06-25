/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/assign/std/vector.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <iostream>
#include <vector>
#include <libflatarray/api_traits.hpp>
#include <libflatarray/soa_grid.hpp>
#include <libflatarray/macros.hpp>

#include "test.h"

using namespace boost::assign;
using namespace LibFlatArray;

class Cell
{
public:
    double memberA;
    double memberB;
    double memberC;
};

LIBFLATARRAY_REGISTER_SOA(Cell, ((double)(memberA))((double)(memberB))((double)(memberC)))

class CellDefaultSizes
{
public:
    double memberA;
    double memberB;
    double memberC;
};

LIBFLATARRAY_REGISTER_SOA(CellDefaultSizes, ((double)(memberA))((double)(memberB))((double)(memberC)))

// class CellDefault2DSizes
// {
// public:
//     class API : public LibFlatArray::api_traits::has_default_2d_sizes
//     {};
// };

class CellDefault3DSizes
{
public:
    class API : public api_traits::has_default_3d_sizes
    {};
};

class TestFunctor
{
public:
    TestFunctor(std::vector<int> *report) :
        report(report)
    {}

    template<typename CELL, int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void operator()(soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor, int *unused)
    {
        *report += DIM_X, DIM_Y, DIM_Z, INDEX;
    }

    template<typename CELL, int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void operator()(const_soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor, int *unused)
    {
            throw std::logic_error("this should not have been called");
    }

private:
    std::vector<int> *report;
};

ADD_TEST(TestSelectSizes)
{
    char data[1024 * 1024];
    std::vector<int> actual;
    std::vector<int> expected;
    TestFunctor functor(&actual);
    typedef api_traits::select_sizes<CellDefaultSizes> selector;

    selector()(data, functor, 10, 20, 30);
    expected += 32, 32, 32, 0;
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 30, 30, 30);
    expected += 32, 32, 32, 0;
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 32, 32, 32);
    expected += 32, 32, 32, 0;
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    selector()(data, functor, 40, 32, 32);
    expected += 128, 128, 32, 0;
    BOOST_TEST(actual == expected);
    actual.clear();
    expected.clear();

    // for (int i = 0; i < actual.size(); ++i) {
    //     std::cout << "actual[" << i << "] = " << actual[i] << "\n";
    // }
}

int main(int argc, char **argv)
{
    return 0;
}
