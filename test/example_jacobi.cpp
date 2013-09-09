/**
 * Copyright 2012-2013 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <iostream>
#include <libflatarray/flat_array.hpp>

class Cell
{
public:
    // fixme: would be nicer to have this as a macro.
    /**
     * This operator is required to give the operator<<(CELL_TYPE,
     * LibFlatArray::soa_accessor) access to our private members.
     */
    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    friend void operator<<(Cell&, const LibFlatArray::soa_accessor<Cell, DIM_X, DIM_Y, DIM_Z, INDEX>);

    Cell(double temp) :
        temp(temp)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood)
    {
        temp = 123;
    }

// private:
    double temp;
};

LIBFLATARRAY_REGISTER_SOA(Cell, ((double)(temp)))

int main(int argc, char **argv)
{
    std::cout << "go\n";

    std::cout << "boomer " << LibFlatArray::detail::flat_array::offset<Cell, 1>::OFFSET << "\n";

    return 0;
}
