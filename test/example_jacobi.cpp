#include <iostream>
#include <libflatarray/flat_array.hpp>

class Cell
{
public:
    Cell(double temp) :
        temp(temp)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood)
    {
        temp = 123;
    }

private:
    double temp;
};

LIBFLATARRAY_REGISTER_SOA(Cell, ((double)(temp)))

int main(int argc, char **argv)
{
    std::cout << "go\n";

    std::cout << "boomer " << LibFlatArray::detail::flat_array::offset<Cell, 1>::OFFSET << "\n";

    return 0;
}
