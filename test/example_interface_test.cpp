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

class BingoBongo
{
public:
    BingoBongo(int startX,
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
    void operator()(const ACCESSOR1& accessor1, const ACCESSOR2& accessor2) const
    {
        for (int z = startZ; z < endZ; ++z) {
            for (int y = startY; y < endY; ++y) {
                for (int x = startX; x < endX; ++x) {
                    int index =
                        ACCESSOR1::DIM_X * ACCESSOR1::DIM_Y * z +
                        ACCESSOR1::DIM_X * y +
                        x;
                    accessor2[index].temperature() = accessor1[index].temperature();
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
    int endZ;;
};

LIBFLATARRAY_REGISTER_SOA(HeatedGameOfLifeCell, ((double)(temperature))((bool)(alive)))

template<typename ACCESSOR1, typename UPDATE_FUNCTOR>
class UpdateFunctorHelper2
{
public:
    UpdateFunctorHelper2(const ACCESSOR1& accessor1, UPDATE_FUNCTOR *functor) :
        accessor1(accessor1),
        functor(functor)
    {}

    template<typename ACCESSOR2>
    void operator()(const ACCESSOR2& accessor2) const
    {
        return (*functor)(accessor1, accessor2);
    }

private:
    const ACCESSOR1 accessor1;
    const UPDATE_FUNCTOR *functor;
};

template<typename GRID2, typename UPDATE_FUNCTOR>
class UpdateFunctorHelper1
{
public:
    UpdateFunctorHelper1(const GRID2 *grid2, UPDATE_FUNCTOR *functor) :
        grid2(grid2),
        functor(functor)
    {}

    template<typename ACCESSOR1>
    void operator()(const ACCESSOR1& accessor1) const
    {
        UpdateFunctorHelper2<ACCESSOR1, UPDATE_FUNCTOR> helper(accessor1, functor);
        grid2->callback(helper);
    }

private:
    const GRID2 *grid2;
    UPDATE_FUNCTOR *functor;
};

class UpdateFunctor
{
public:
    template<typename GRID1, typename GRID2, typename UPDATE_FUNCTOR>
    void operator()(const GRID1& grid1, const GRID2& grid2, UPDATE_FUNCTOR *functor) const
    {
        UpdateFunctorHelper1<GRID2, UPDATE_FUNCTOR> helper(&grid2, functor);
        grid1.callback(helper);
    }
};

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


    BingoBongo functor(0, 0, 0, dimX, dimY, dimZ);
    UpdateFunctor()(gridOld, gridNew, &functor);

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                HeatedGameOfLifeCell cell = gridNew.get(x, y, z);
                std::cout << "(" << x << ", " << y << ", " << z << ") == (" << cell.temperature
                          << ", " << cell.alive << ")\n";
            }
        }
    }
}
