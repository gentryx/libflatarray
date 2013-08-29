/**
 * Copyright 2013 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/detail/lightweight_test.hpp>
#include <iostream>
#include <typeinfo>
#include <libflatarray/flat_array.hpp>
#include <vector>

using namespace LibFlatArray;

class HeatedGameOfLifeCell
{
public:
    inline HeatedGameOfLifeCell(double temperature=0.0, bool alive=false) :
        temperature(temperature),
        alive(alive)
    {}

    inline bool operator==(const HeatedGameOfLifeCell& other) const
    {
	return
	    (temperature == other.temperature) &&
	    (alive == other.alive);
    }

    double temperature;
    bool alive;
};

class InvertTemperature
{
public:
    InvertTemperature(int dimX, int dimY, int dimZ) :
	dimX(dimX),
	dimY(dimY),
	dimZ(dimZ)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR accessor, int *index)
    {
        for (int z = 0; z < dimZ; ++z) {
            for (int y = 0; y < dimY; ++y) {
                for (int x = 0; x < dimX; ++x) {
                    *index =
                        ACCESSOR::DIM_X * ACCESSOR::DIM_Y * z +
                        ACCESSOR::DIM_X * y +
                        x;
		    accessor.temperature() = -accessor.temperature();
		}
	    }
	}
    }

private:
    int dimX;
    int dimY;
    int dimZ;
};

class CopyTemperatureNativeStyle
{
public:
    CopyTemperatureNativeStyle(
        int startX,
        int startY,
        int startZ,
        int endX,
        int endY,
        int endZ) :
        startX(startX),
        startY(startY),
        startZ(startZ),
        endX(endX),
        endY(endY),
        endZ(endZ)
    {}

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(const ACCESSOR1& accessor1, int *index1, ACCESSOR2& accessor2, int *index2) const
    {
        for (int z = startZ; z < endZ; ++z) {
            for (int y = startY; y < endY; ++y) {
                for (int x = startX; x < endX; ++x) {
                    int index =
                        ACCESSOR1::DIM_X * ACCESSOR1::DIM_Y * z +
                        ACCESSOR1::DIM_X * y +
                        x;
                    *index1 = index;
                    *index2 = index;
                    accessor2.temperature() = accessor1.temperature();
                }
            }
        }
    }

private:
    int startX;
    int startY;
    int startZ;
    int endX;
    int endY;
    int endZ;
};

class CopyTemperatureCactusStyle
{
public:
    CopyTemperatureCactusStyle(
        int startX,
        int startY,
        int startZ,
        int endX,
        int endY,
        int endZ) :
        startX(startX),
        startY(startY),
        startZ(startZ),
        endX(endX),
        endY(endY),
        endZ(endZ)
    {}

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(const ACCESSOR1& accessor1, int *index1, ACCESSOR2& accessor2, int *index2) const
    {
        for (int z = startZ; z < endZ; ++z) {
            for (int y = startY; y < endY; ++y) {
                for (int x = startX; x < endX; ++x) {
                    int index =
                        ACCESSOR1::DIM_X * ACCESSOR1::DIM_Y * z +
                        ACCESSOR1::DIM_X * y +
                        x;
                    (&accessor2.temperature())[index] = (&accessor1.temperature())[index];
                }
            }
        }
    }

private:
    int startX;
    int startY;
    int startZ;
    int endX;
    int endY;
    int endZ;
};

LIBFLATARRAY_REGISTER_SOA(HeatedGameOfLifeCell, ((double)(temperature))((bool)(alive)))

void testSingleGetSet()
{
    int dimX = 5;
    int dimY = 3;
    int dimZ = 10;

    soa_grid<HeatedGameOfLifeCell> grid(dimX, dimY, dimZ);
    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
		double temp = z * 100 + y + x * 0.01;
                grid.set(x, y, z, HeatedGameOfLifeCell(temp, false));
	    }
	}
    }

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
		double temp = z * 100 + y + x * 0.01;
		HeatedGameOfLifeCell cell(temp, false);
		BOOST_TEST(cell == grid.get(x, y, z));
	    }
	}
    }
}

void testArrayGetSet()
{
    int dimX = 15;
    int dimY = 3;
    int dimZ = 10;

    soa_grid<HeatedGameOfLifeCell> grid(dimX, dimY, dimZ);
    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
	    std::vector<HeatedGameOfLifeCell> cells(dimX);

	    for (int x = 0; x < dimX; ++x) {
		double temp = z * 100 + y + x * 0.01;
		cells[x] = HeatedGameOfLifeCell(temp, false);
	    }

	    grid.set(0, y, z, &cells[0], dimX);
	}
    }

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
	    std::vector<HeatedGameOfLifeCell> cells(dimX);

	    grid.get(0, y, z, &cells[0], dimX);

            for (int x = 0; x < dimX; ++x) {
		double temp = z * 100 + y + x * 0.01;
		HeatedGameOfLifeCell cell(temp, false);
		BOOST_TEST(cell == cells[x]);
	    }
	}
    }
}

void testResizeAndByteSize()
{
    int dimX = 2;
    int dimY = 2;
    int dimZ = 2;

    soa_grid<HeatedGameOfLifeCell> grid(dimX, dimY, dimZ);
    BOOST_TEST(grid.byte_size() == (32 * 32 * 32 * sizeof(HeatedGameOfLifeCell)));

    dimX = 10;
    dimY = 20;
    dimZ = 40;
    grid.resize(dimX, dimY, dimZ);
    grid.set(dimX - 1, dimY - 1, dimZ - 1, HeatedGameOfLifeCell(4711));
    BOOST_TEST(grid.get(dimX - 1, dimY - 1, dimZ - 1) == HeatedGameOfLifeCell(4711));
    BOOST_TEST(grid.byte_size() == (32 * 32 * 256 * sizeof(HeatedGameOfLifeCell)));

}

void testSingleCallback()
{
    int dimX = 5;
    int dimY = 3;
    int dimZ = 10;

    soa_grid<HeatedGameOfLifeCell> grid(dimX, dimY, dimZ);
    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
		double temp = z * 100 + y + x * 0.01;
                grid.set(x, y, z, HeatedGameOfLifeCell(temp, false));
	    }
	}
    }

    int index = 0;
    grid.callback(InvertTemperature(dimX, dimY, dimZ), &index);

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
		double temp = z * 100 + y + x * 0.01;
		HeatedGameOfLifeCell cell(-temp, false);
		BOOST_TEST(cell == grid.get(x, y, z));
	    }
	}
    }
}

template<typename COPY_FUNCTOR>
void testDualCallback()
{
    int dimX = 5;
    int dimY = 3;
    int dimZ = 2;

    soa_grid<HeatedGameOfLifeCell> gridOld(dimX, dimY, dimZ);
    soa_grid<HeatedGameOfLifeCell> gridNew(dimX, dimY, dimZ);

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
		double temp = z * 100 + y + x * 0.01;
                gridOld.set(x, y, z, HeatedGameOfLifeCell(temp, false));
            }
        }
    }

    COPY_FUNCTOR functor(0, 0, 0, dimX, dimY, dimZ);
    gridOld.callback(&gridNew, functor);

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                HeatedGameOfLifeCell cell = gridOld.get(x, y, z);
		double temp = z * 100 + y + x * 0.01;
		BOOST_TEST(cell.temperature == temp);
		BOOST_TEST(cell.alive == false);
            }
        }
    }
}

int main(int argc, char **argv)
{
    testSingleGetSet();
    testArrayGetSet();
    testResizeAndByteSize();
    testSingleCallback();
    testDualCallback<CopyTemperatureCactusStyle>();
    testDualCallback<CopyTemperatureNativeStyle>();
    return 0;
}
