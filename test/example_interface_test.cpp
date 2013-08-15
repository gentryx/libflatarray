/**
 * Copyright 2013 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <iostream>
#include <typeinfo>
#include <libflatarray/flat_array.hpp>

using namespace LibFlatArray;

class HeatedGameOfLifeCell
{
public:
    inline HeatedGameOfLifeCell(double temperature=0.0, bool alive=false) :
        temperature(temperature),
        alive(alive)
    {}

    double temperature;
    bool alive;
};

class HeatedGameOfLifeCellUpdateFunctor
{
public:
    template<typename SOA_ACCESSOR1, typename SOA_ACCESSOR2>
    // fixme: use size_t instead of int?
    void operator()(SOA_ACCESSOR1 gridOld, SOA_ACCESSOR2 gridNew, int startX, int endX)
    {
        for (int x = startX; x < endX; ++x) {
            gridNew[x].temperature() = gridOld[x].temperature() * 2;
            gridNew[x].alive() = ! gridOld[x].alive();

            // // soa code:
            // gridNew[x].temperature() = gridOld[x].temperature() * 2;
            // // cactus code:
            // temperature[x] = temperature_p[x] * 2;
        }
    }
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

int main(int argc, char **argv)
{
    int dimX = 5;
    int dimY = 3;
    int dimZ = 2;

    soa_grid<HeatedGameOfLifeCell> gridOld(dimX, dimY, dimZ);
    soa_grid<HeatedGameOfLifeCell> gridNew(dimX, dimY, dimZ);

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                gridOld.set(x, y, z, HeatedGameOfLifeCell(z * 100 + y + x * 0.01, false));
            }
        }
    }

    CopyTemperatureCactusStyle functor(0, 0, 0, dimX, dimY, dimZ);
    // CopyTemperatureNativeStyle functor(0, 0, 0, dimX, dimY, dimZ);
    gridOld.callback(&gridNew, functor);

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                HeatedGameOfLifeCell cell = gridOld.get(x, y, z);
                std::cout << "(" << x << ", " << y << ", " << z << ") == (" << cell.temperature
                          << ", " << cell.alive << ")\n";
            }
        }
    }
}
