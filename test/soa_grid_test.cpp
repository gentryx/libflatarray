/**
 * Copyright 2013-2017 Andreas Schäfer
 * Copyright 2017 Google
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <libflatarray/flat_array.hpp>

// globally disable some warnings with MSVC, that are issued not for a
// specific header, but rather for the interaction of system headers
// and LibFlatArray source.  Also disable overly eager sign conversion
// and overflow warnings:
#ifdef _MSC_BUILD
#pragma warning( disable : 4244 4305 4307 4365 4456 4514 4710 4800 )
#endif

// Don't warn about these functions being stripped from an executable
// as they're not being used, that's actually expected behavior.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <iostream>
#include <typeinfo>
#include <map>
#include <vector>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#include "test.hpp"

// padding is fine:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

class HeatedGameOfLifeCell
{
public:
    inline
    explicit HeatedGameOfLifeCell(double temperature=0.0, bool alive=false) :
        temperature(temperature),
        alive(alive)
    {}

    inline bool operator==(const HeatedGameOfLifeCell& other) const
    {
        return
            (temperature == other.temperature) &&
            (alive == other.alive);
    }

    inline bool operator!=(const HeatedGameOfLifeCell& other) const
    {
        return !(*this == other);
    }

    double temperature;
    bool alive;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

LIBFLATARRAY_REGISTER_SOA(
    HeatedGameOfLifeCell,
    ((double)(temperature))
    ((bool)(alive)))

class CellWithMultipleMembersOfSameType
{
public:
    LIBFLATARRAY_ACCESS

    double memberA;
    double memberB;

    static double CellWithMultipleMembersOfSameType:: *getMemberCPointer()
    {
        return &CellWithMultipleMembersOfSameType::memberC;
    }

private:
    double memberC;
};

LIBFLATARRAY_REGISTER_SOA(
    CellWithMultipleMembersOfSameType,
    ((double)(memberA))
    ((double)(memberB))
    ((double)(memberC)))

class CellWithArrayMember
{
public:

    double temp[40];
};


LIBFLATARRAY_REGISTER_SOA(
    CellWithArrayMember,
    ((double)(temp)(40)))

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           const HeatedGameOfLifeCell& c)
{
    os << "(" << c.temperature << ", " << c.alive << ")";
    return os;
}

class MemberAccessChecker1
{
public:
    MemberAccessChecker1(int dim_x, int dim_y, int dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR& accessor) const
    {
        for (long z = 0; z < dim_z; ++z) {
            for (long y = 0; y < dim_y; ++y) {
                for (long x = 0; x < dim_x; ++x) {
                    accessor.index() = accessor.gen_index(x, y, z);

                    double actualA = accessor.template access_member<double, 0>();
                    bool actualB = accessor.template access_member<bool, 1>();

                    double expectedA = x * 1000.0 + y + z * 0.001;
                    bool expectedB = (x % 2 == 0);

                    BOOST_TEST(actualA == expectedA);
                    BOOST_TEST(actualB == expectedB);
                }
            }
        }
    }

private:
    long dim_x;
    long dim_y;
    long dim_z;
};

class MemberAccessChecker2
{
public:
    MemberAccessChecker2(long dim_x, long dim_y, long dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR& accessor) const
    {
        for (long z = 0; z < dim_z; ++z) {
            for (long y = 0; y < dim_y; ++y) {
                for (long x = 0; x < dim_x; ++x) {
                    accessor.index() = accessor.gen_index(x, y, z);

                    double actualA = *reinterpret_cast<double*>(accessor.access_member(8, 0));
                    bool actualB = *reinterpret_cast<bool*>(accessor.access_member(1, 8));

                    double expectedA = x * 1000.0 + y + z * 0.001;
                    bool expectedB = (x % 2 == 0);

                    BOOST_TEST(actualA == expectedA);
                    BOOST_TEST(actualB == expectedB);
                }
            }
        }
    }

private:
    long dim_x;
    long dim_y;
    long dim_z;
};

class InvertTemperature
{
public:
    InvertTemperature(long dim_x, long dim_y, long dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR& accessor) const
    {
        for (long z = 0; z < dim_z; ++z) {
            for (long y = 0; y < dim_y; ++y) {
                for (long x = 0; x < dim_x; ++x) {
                    accessor.index() =
                        ACCESSOR::DIM_X * ACCESSOR::DIM_Y * z +
                        ACCESSOR::DIM_X * y +
                        x;
                    accessor.temperature() = -accessor.temperature();
                }
            }
        }
    }

private:
    long dim_x;
    long dim_y;
    long dim_z;
};

class CopyTemperatureNativeStyle
{
public:
    CopyTemperatureNativeStyle(
        long startX,
        long startY,
        long startZ,
        long endX,
        long endY,
        long endZ) :
        startX(startX),
        startY(startY),
        startZ(startZ),
        endX(endX),
        endY(endY),
        endZ(endZ)
    {}

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(ACCESSOR1& accessor1,
                    ACCESSOR2& accessor2) const
    {
        long index = 0;

        for (long z = startZ; z < endZ; ++z) {
            for (long y = startY; y < endY; ++y) {
                for (long x = startX; x < endX; ++x) {
                    index =
                        ACCESSOR1::DIM_X * ACCESSOR1::DIM_Y * z +
                        ACCESSOR1::DIM_X * y +
                        x;
                    accessor1.index() = index;
                    accessor2.index() = index;
                    accessor2.temperature() = accessor1.temperature();
                }
            }
        }
    }

private:
    long startX;
    long startY;
    long startZ;
    long endX;
    long endY;
    long endZ;
};

class CopyTemperatureCactusStyle
{
public:
    CopyTemperatureCactusStyle(
        long startX,
        long startY,
        long startZ,
        long endX,
        long endY,
        long endZ) :
        startX(startX),
        startY(startY),
        startZ(startZ),
        endX(endX),
        endY(endY),
        endZ(endZ)
    {}

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(const ACCESSOR1& accessor1, ACCESSOR2& accessor2) const
    {
        for (long z = startZ; z < endZ; ++z) {
            for (long y = startY; y < endY; ++y) {
                for (long x = startX; x < endX; ++x) {
                    long index =
                        ACCESSOR1::DIM_X * ACCESSOR1::DIM_Y * z +
                        ACCESSOR1::DIM_X * y +
                        x;
                    (&accessor2.temperature())[index] = (&accessor1.temperature())[index];
                }
            }
        }
    }

private:
    long startX;
    long startY;
    long startZ;
    long endX;
    long endY;
    long endZ;
};

class ArrayParticle
{
public:
    explicit ArrayParticle(
        float mass = 0,
        float charge = 0,
        float pos0 = 0,
        float pos1 = 0,
        float pos2 = 0,
        float vel0 = 0,
        float vel1 = 0,
        float vel2 = 0,
        int state = 0) :
        mass(mass),
        charge(charge),
        state(state)
    {
        pos[0] = pos0;
        pos[1] = pos1;
        pos[2] = pos2;
        vel[0] = vel0;
        vel[1] = vel1;
        vel[2] = vel2;
    }

    float mass;
    float charge;
    float pos[3];
    float vel[3];
    int state;
};

LIBFLATARRAY_REGISTER_SOA(
    ArrayParticle,
    ((float)(mass))
    ((float)(charge))
    ((float)(pos)(3))
    ((float)(vel)(3))
    ((int)(state)))

class DestructionCounterClass
{
public:
    static std::size_t count;

    ~DestructionCounterClass()
    {
        ++count;
    }
};

std::size_t DestructionCounterClass::count = 0;

// padding is fine:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4714 4820 )
#endif

class CellWithNonTrivialMembers
{
public:
    typedef std::map<int, std::vector<double> > MapType;
    int id;
    MapType map;
    MapType maps[4];
    DestructionCounterClass destructCounter;
};

LIBFLATARRAY_REGISTER_SOA(
    CellWithNonTrivialMembers,
    ((int)(id))
    ((CellWithNonTrivialMembers::MapType)(map))
    ((CellWithNonTrivialMembers::MapType)(maps)(4))
    ((DestructionCounterClass)(destructCounter)))

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

class MultiplyVelocityArrayStyle
{
public:
    MultiplyVelocityArrayStyle(long dim_x, long dim_y, long dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR& accessor) const
    {
        for (long z = 0; z < dim_z; ++z) {
            for (long y = 0; y < dim_y; ++y) {
                for (long x = 0; x < dim_x; ++x) {
                    accessor.index() =
                        ACCESSOR::DIM_X * ACCESSOR::DIM_Y * z +
                        ACCESSOR::DIM_X * y +
                        x;
                    for (int i = 0; i < 3; ++i) {
                        accessor.vel()[i] *= (3 + i);
                    }
                }
            }
        }
    }

private:
    long dim_x;
    long dim_y;
    long dim_z;
};

class MultiplyVelocityFunctionStyle
{
public:
    MultiplyVelocityFunctionStyle(long dim_x, long dim_y, long dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR& accessor) const
    {
        for (long z = 0; z < dim_z; ++z) {
            for (long y = 0; y < dim_y; ++y) {
                for (long x = 0; x < dim_x; ++x) {
                    accessor.index() =
                        ACCESSOR::DIM_X * ACCESSOR::DIM_Y * z +
                        ACCESSOR::DIM_X * y +
                        x;
                    accessor.template vel<0>() *= 6;
                    accessor.template vel<1>() *= 7;
                    accessor.template vel<2>() *= 8;
                }
            }
        }
    }

private:
    long dim_x;
    long dim_y;
    long dim_z;
};

class OffsetPositionArrayStyle
{
public:
    OffsetPositionArrayStyle(long dim_x, long dim_y, long dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR& accessor) const
    {
        for (long z = 0; z < dim_z; ++z) {
            for (long y = 0; y < dim_y; ++y) {
                for (long x = 0; x < dim_x; ++x) {
                    accessor.index() =
                        ACCESSOR::DIM_X * ACCESSOR::DIM_Y * z +
                        ACCESSOR::DIM_X * y +
                        x;
                    accessor.pos()[0] += x * 1000;
                    accessor.pos()[1] += y * 2000;
                    accessor.pos()[2] += z * 3000;
                }
            }
        }
    }

private:
    long dim_x;
    long dim_y;
    long dim_z;
};

class OffsetPositionFunctionStyle
{
public:
    OffsetPositionFunctionStyle(long dim_x, long dim_y, long dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR& accessor) const
    {
        for (long z = 0; z < dim_z; ++z) {
            for (long y = 0; y < dim_y; ++y) {
                for (long x = 0; x < dim_x; ++x) {
                    accessor.index() =
                        ACCESSOR::DIM_X * ACCESSOR::DIM_Y * z +
                        ACCESSOR::DIM_X * y +
                        x;
                    accessor.template pos<0>() += x * 1001;
                    accessor.template pos<1>() += y * 2001;
                    accessor.template pos<2>() += z * 3001;
                }
            }
        }
    }

private:
    long dim_x;
    long dim_y;
    long dim_z;
};

namespace LibFlatArray {

ADD_TEST(TestSingleGetSet)
{
    long dim_x = 5;
    long dim_y = 3;
    long dim_z = 10;

    soa_grid<HeatedGameOfLifeCell> grid(dim_x, dim_y, dim_z);
    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                double temp = z * 100 + y + x * 0.01;
                grid.set(x, y, z, HeatedGameOfLifeCell(temp, false));
            }
        }
    }

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                double temp = z * 100 + y + x * 0.01;
                HeatedGameOfLifeCell cell(temp, false);
                BOOST_TEST(cell == grid.get(x, y, z));
            }
        }
    }
}

ADD_TEST(TestArrayGetSet)
{
    long dim_x = 15;
    long dim_y = 3;
    long dim_z = 10;

    soa_grid<HeatedGameOfLifeCell> grid(dim_x, dim_y, dim_z);
    std::vector<double> temp_buffer;

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            std::vector<HeatedGameOfLifeCell> cells(dim_x);

            for (long x = 0; x < dim_x; ++x) {
                double temp = z * 100 + y + x * 0.01;
                temp_buffer.push_back(temp);
                cells[x] = HeatedGameOfLifeCell(temp, false);
            }

            grid.set(0, y, z, &cells[0], dim_x);
        }
    }

    std::size_t index = 0;

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            std::vector<HeatedGameOfLifeCell> cells(dim_x);

            grid.get(0, y, z, &cells[0], dim_x);

            for (long x = 0; x < dim_x; ++x) {
                double temp = temp_buffer[index++];
                HeatedGameOfLifeCell cell(temp, false);
                BOOST_TEST(cell == cells[x]);
            }
        }
    }
}

ADD_TEST(TestResizeAndByteSize)
{
    long dim_x = 2;
    long dim_y = 2;
    long dim_z = 2;
    long cellSize = 9; // 1 double, 1 bool
    soa_grid<HeatedGameOfLifeCell> grid(dim_x, dim_y, dim_z);
    BOOST_TEST(grid.byte_size() == static_cast<unsigned long>((32 * 32 * 32 * cellSize)));

    dim_x = 10;
    dim_y = 20;
    dim_z = 40;
    grid.resize(dim_x, dim_y, dim_z);
    grid.set(dim_x - 1, dim_y - 1, dim_z - 1, HeatedGameOfLifeCell(4711));
    BOOST_TEST(grid.get(dim_x - 1, dim_y - 1, dim_z - 1) == HeatedGameOfLifeCell(4711));
    BOOST_TEST(grid.byte_size() == static_cast<unsigned long>((64 * 64 * 64 * cellSize)));

}

ADD_TEST(TestSingleCallback)
{
    long dim_x = 5;
    long dim_y = 3;
    long dim_z = 10;

    soa_grid<HeatedGameOfLifeCell> grid(dim_x, dim_y, dim_z);
    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                double temp = z * 100 + y + x * 0.01;
                grid.set(x, y, z, HeatedGameOfLifeCell(temp, false));
            }
        }
    }

    grid.callback(InvertTemperature(dim_x, dim_y, dim_z));

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
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
    long dim_x = 5;
    long dim_y = 3;
    long dim_z = 2;

    soa_grid<HeatedGameOfLifeCell> gridOld(dim_x, dim_y, dim_z);
    soa_grid<HeatedGameOfLifeCell> gridNew(dim_x, dim_y, dim_z);

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                double temp = z * 100 + y + x * 0.01;
                gridOld.set(x, y, z, HeatedGameOfLifeCell(temp, false));
            }
        }
    }

    COPY_FUNCTOR functor(0, 0, 0, dim_x, dim_y, dim_z);
    gridOld.callback(&gridNew, functor);

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                HeatedGameOfLifeCell cell = gridOld.get(x, y, z);
                double temp = z * 100 + y + x * 0.01;
                BOOST_TEST(cell.temperature == temp);
                BOOST_TEST(cell.alive == false);
            }
        }
    }
}

ADD_TEST(TestDualCallback)
{
    testDualCallback<CopyTemperatureCactusStyle>();
    testDualCallback<CopyTemperatureNativeStyle>();
}

ADD_TEST(TestAssignment1)
{
    soa_grid<HeatedGameOfLifeCell> gridOld(20, 30, 40);
    soa_grid<HeatedGameOfLifeCell> gridNew(70, 60, 50);

    BOOST_TEST(gridOld.data() != gridNew.data());
    BOOST_TEST(gridOld.dim_x()  != gridNew.dim_x());
    BOOST_TEST(gridOld.dim_y()  != gridNew.dim_y());
    BOOST_TEST(gridOld.dim_z()  != gridNew.dim_z());
    BOOST_TEST(gridOld.my_byte_size != gridNew.my_byte_size);

    gridOld = gridNew;

    BOOST_TEST(gridOld.data() != gridNew.data());
    BOOST_TEST(gridOld.dim_x()  == gridNew.dim_x());
    BOOST_TEST(gridOld.dim_y()  == gridNew.dim_y());
    BOOST_TEST(gridOld.dim_z()  == gridNew.dim_z());
    BOOST_TEST(gridOld.my_byte_size == gridNew.my_byte_size);
}

ADD_TEST(TestAssignment2)
{
    soa_grid<HeatedGameOfLifeCell> grid1(20, 10, 1);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            grid1.set(x, y, 0, HeatedGameOfLifeCell(y * 100 + x, true));
        }
    }

    soa_grid<HeatedGameOfLifeCell> grid2;
    grid2 = grid1;

    // overwrite old grid to ensure both are still separate
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            grid1.set(x, y, 0, HeatedGameOfLifeCell(-1, false));
        }
    }

    BOOST_TEST(grid1.dim_x() == 20);
    BOOST_TEST(grid1.dim_y() == 10);
    BOOST_TEST(grid1.dim_z() ==  1);

    BOOST_TEST(grid2.dim_x() == 20);
    BOOST_TEST(grid2.dim_y() == 10);
    BOOST_TEST(grid2.dim_z() ==  1);

    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            HeatedGameOfLifeCell cell = grid2.get(x, y, 0);
            BOOST_TEST(cell == HeatedGameOfLifeCell(y * 100 + x, true));
            cell = grid1.get(x, y, 0);
            BOOST_TEST(cell == HeatedGameOfLifeCell(-1, false));
        }
    }
}

ADD_TEST(TestAssignment3)
{
    soa_grid<HeatedGameOfLifeCell> grid1(20, 10, 1);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            grid1.set(x, y, 0, HeatedGameOfLifeCell(y * 100 + x, true));
        }
    }

    const soa_grid<HeatedGameOfLifeCell>& grid_reference(grid1);
    soa_grid<HeatedGameOfLifeCell> grid2;
    grid2 = grid_reference;

    // overwrite old grid to ensure both are still separate
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            grid1.set(x, y, 0, HeatedGameOfLifeCell(-1, false));
        }
    }

    BOOST_TEST(grid1.dim_x() == 20);
    BOOST_TEST(grid1.dim_y() == 10);
    BOOST_TEST(grid1.dim_z() ==  1);

    BOOST_TEST(grid2.dim_x() == 20);
    BOOST_TEST(grid2.dim_y() == 10);
    BOOST_TEST(grid2.dim_z() ==  1);

    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            HeatedGameOfLifeCell cell = grid2.get(x, y, 0);
            BOOST_TEST(cell == HeatedGameOfLifeCell(y * 100 + x, true));
            cell = grid1.get(x, y, 0);
            BOOST_TEST(cell == HeatedGameOfLifeCell(-1, false));
        }
    }
}

ADD_TEST(TestAssignment4)
{
    soa_grid<CellWithNonTrivialMembers> grid1(100, 50, 1);
    for (int y = 0; y < 50; ++y) {
        for (int x = 0; x < 100; ++x) {
            CellWithNonTrivialMembers dummy;
            dummy.map[y].push_back(x);
            grid1.set(x, y, 0, dummy);
        }
    }

    soa_grid<CellWithNonTrivialMembers> grid2(10, 20, 1);
    grid2 = grid1;

    // overwrite old grid to ensure both are still separate
    for (int y = 0; y < 50; ++y) {
        for (int x = 0; x < 100; ++x) {
            CellWithNonTrivialMembers dummy;
            dummy.map[y].push_back(-1);
            dummy.map[y].push_back(-2);
            grid1.set(x, y, 0, dummy);
        }
    }

    BOOST_TEST(grid1.dim_x() == 100);
    BOOST_TEST(grid1.dim_y() ==  50);
    BOOST_TEST(grid1.dim_z() ==   1);

    BOOST_TEST(grid2.dim_x() == 100);
    BOOST_TEST(grid2.dim_y() ==  50);
    BOOST_TEST(grid2.dim_z() ==   1);

    for (int y = 0; y < 50; ++y) {
        for (int x = 0; x < 100; ++x) {
            CellWithNonTrivialMembers cell = grid2.get(x, y, 0);
            BOOST_TEST(cell.map[y].size() == 1);
            BOOST_TEST(cell.map[y][0] == x);
            cell = grid1.get(x, y, 0);
            BOOST_TEST(cell.map[y].size() == 2);
            BOOST_TEST(cell.map[y][0] == -1);
            BOOST_TEST(cell.map[y][1] == -2);
        }
    }
}

ADD_TEST(TestSwap)
{
    long dim_x = 5;
    long dim_y = 3;
    long dim_z = 2;

    soa_grid<HeatedGameOfLifeCell> gridOld(dim_x, dim_y, dim_z);
    soa_grid<HeatedGameOfLifeCell> gridNew(dim_x, dim_y, dim_z);

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                gridOld.set(x, y, z, HeatedGameOfLifeCell(4711));
                gridNew.set(x, y, z, HeatedGameOfLifeCell(666));
            }
        }
    }

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                BOOST_TEST(gridOld.get(x, y, z).temperature == 4711);
                BOOST_TEST(gridNew.get(x, y, z).temperature == 666);
            }
        }
    }

    using std::swap;
    swap(gridOld, gridNew);

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                BOOST_TEST(gridOld.get(x, y, z).temperature == 666);
                BOOST_TEST(gridNew.get(x, y, z).temperature == 4711);
            }
        }
    }
}

ADD_TEST(TestCopyArrayIn)
{
    long dim_x = 5;
    long dim_y = 3;
    long dim_z = 2;

    soa_grid<HeatedGameOfLifeCell> grid(dim_x, dim_y, dim_z);
    std::vector<char> store0(1024);
    double *store1 = reinterpret_cast<double*>(&store0[0]);
    store1[ 0] = 47.11;
    store1[ 1] = 1.234;
    store1[ 2] = 666.1;
    store0[24] = true;
    store0[25] = false;
    store0[26] = true;

    grid.load(0, 0, 0, &store0[0], 3);
    BOOST_TEST(grid.get(0, 0, 0) == HeatedGameOfLifeCell(47.11, true));
    BOOST_TEST(grid.get(1, 0, 0) == HeatedGameOfLifeCell(1.234, false));
    BOOST_TEST(grid.get(2, 0, 0) == HeatedGameOfLifeCell(666.1, true));

    store1[ 0] = 2.345;
    store1[ 1] = 987.6;
    store0[16] = false;
    store0[17] = true;

    grid.load(3, 2, 1, &store0[0], 2);
    BOOST_TEST(grid.get(3, 2, 1) == HeatedGameOfLifeCell(2.345, false));
    BOOST_TEST(grid.get(4, 2, 1) == HeatedGameOfLifeCell(987.6, true));
}

ADD_TEST(TestCopyArrayOut)
{
    long dim_x = 5;
    long dim_y = 3;
    long dim_z = 2;

    std::vector<char> store0(1024);
    double *store1 = reinterpret_cast<double*>(&store0[0]);
    soa_grid<HeatedGameOfLifeCell> grid(dim_x, dim_y, dim_z);

    grid.set(0, 0, 0, HeatedGameOfLifeCell(47.11, true));
    grid.set(1, 0, 0, HeatedGameOfLifeCell(1.234, false));
    grid.set(2, 0, 0, HeatedGameOfLifeCell(666.1, true));
    grid.save(0, 0, 0, &store0[0], 3);

    BOOST_TEST(store1[ 0] == 47.11);
    BOOST_TEST(store1[ 1] == 1.234);
    BOOST_TEST(store1[ 2] == 666.1);
    BOOST_TEST(static_cast<bool>(store0[24]) == true);
    BOOST_TEST(static_cast<bool>(store0[25]) == false);
    BOOST_TEST(static_cast<bool>(store0[26]) == true);

    grid.set(3, 2, 1, HeatedGameOfLifeCell(2.345, false));
    grid.set(4, 2, 1, HeatedGameOfLifeCell(987.6, true));
    grid.save(3, 2, 1, &store0[0], 2);

    BOOST_TEST(store1[ 0] == 2.345);
    BOOST_TEST(store1[ 1] == 987.6);
    BOOST_TEST(static_cast<bool>(store0[16]) == false);
    BOOST_TEST(static_cast<bool>(store0[17]) == true);
}

ADD_TEST(TestNumberOfMembers)
{
// Don't warn about const expressions not being flagged as such: we
// don't have a suitable macro for such comparisons.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4127 )
#endif
    BOOST_TEST(number_of_members<HeatedGameOfLifeCell>::VALUE == 2);
#ifdef _MSC_BUILD
#pragma warning( pop )
#endif
}

ADD_TEST(TestAggregatedMemberSize)
{
// Don't warn about const expressions not being flagged as such: we
// don't have a suitable macro for such comparisons.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4127 )
#endif
    BOOST_TEST(sizeof(HeatedGameOfLifeCell) == 16);
    BOOST_TEST(aggregated_member_size<HeatedGameOfLifeCell>::VALUE == 9);
#ifdef _MSC_BUILD
#pragma warning( pop )
#endif
}

ADD_TEST(TestAccessMember)
{
    long dim_x = 15;
    long dim_y = 13;
    long dim_z = 19;

    soa_grid<HeatedGameOfLifeCell> grid(dim_x, dim_y, dim_z);

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                grid.set(x, y, z, HeatedGameOfLifeCell(x * 1000.0 + y + z * 0.001, (x % 2 == 0)));
            }
        }
    }

    BOOST_TEST(grid.get(11, 12, 13).temperature == 11012.013);
    BOOST_TEST(grid.get(10, 12, 13).alive == true);
    BOOST_TEST(grid.get(11, 12, 13).alive == false);
    BOOST_TEST(grid.get(12, 12, 13).alive == true);

    grid.callback(MemberAccessChecker1(dim_x, dim_y, dim_z));
    grid.callback(MemberAccessChecker2(dim_x, dim_y, dim_z));
}

ADD_TEST(TestMemberPtrToOffset)
{
    BOOST_TEST( 0 == member_ptr_to_offset()(&HeatedGameOfLifeCell::temperature));
    BOOST_TEST( 8 == member_ptr_to_offset()(&HeatedGameOfLifeCell::alive));

    BOOST_TEST( 0 == member_ptr_to_offset()(&CellWithMultipleMembersOfSameType::memberA));
    BOOST_TEST( 8 == member_ptr_to_offset()(&CellWithMultipleMembersOfSameType::memberB));
    BOOST_TEST(16 == member_ptr_to_offset()(CellWithMultipleMembersOfSameType::getMemberCPointer()));
}

ADD_TEST(TestArrayMember)
{
    long dim_x = 40;
    long dim_y = 15;
    long dim_z = 10;

    soa_grid<ArrayParticle> grid(dim_x, dim_y, dim_z);
    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                ArrayParticle particle(
                    0.1 + x,
                    0.2 + x,
                    0.3 + x,
                    0.4 + y,
                    0.5 + z,
                    0.6 + x,
                    0.7 + y,
                    0.9 + z,
                    x + y + z);
                grid.set(x, y, z, particle);
            }
        }
    }

    grid.callback(MultiplyVelocityArrayStyle(dim_x, dim_y, dim_z));

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                ArrayParticle particle = grid.get(x, y, z);

                BOOST_TEST(particle.mass   == float(0.1 + x));
                BOOST_TEST(particle.charge == (float(0.2 + x)));
                BOOST_TEST(particle.pos[0] == (float(0.3 + x)));
                BOOST_TEST(particle.pos[1] == (float(0.4 + y)));
                BOOST_TEST(particle.pos[2] == (float(0.5 + z)));
                BOOST_TEST(particle.vel[0] == (float(0.6 + x) * 3));
                BOOST_TEST(particle.vel[1] == (float(0.7 + y) * 4));
                BOOST_TEST(particle.vel[2] == (float(0.9 + z) * 5));
                BOOST_TEST(particle.state  == int(x + y + z));
            }
        }
    }

    grid.callback(MultiplyVelocityFunctionStyle(dim_x, dim_y, dim_z));

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                ArrayParticle particle = grid.get(x, y, z);

                BOOST_TEST(particle.mass   == float(0.1 + x));
                BOOST_TEST(particle.charge == (float(0.2 + x)));
                BOOST_TEST(particle.pos[0] == (float(0.3 + x)));
                BOOST_TEST(particle.pos[1] == (float(0.4 + y)));
                BOOST_TEST(particle.pos[2] == (float(0.5 + z)));
                TEST_REAL_ACCURACY(particle.vel[0], (float(0.6 + x) * 3 * 6), 0.00001);
                TEST_REAL_ACCURACY(particle.vel[1], (float(0.7 + y) * 4 * 7), 0.00001);
                TEST_REAL_ACCURACY(particle.vel[2], (float(0.9 + z) * 5 * 8), 0.00001);
                BOOST_TEST(particle.state  == int(x + y + z));
            }
        }
    }

    grid.callback(OffsetPositionArrayStyle(dim_x, dim_y, dim_z));

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                ArrayParticle particle = grid.get(x, y, z);

                BOOST_TEST(particle.mass   == float(0.1 + x));
                BOOST_TEST(particle.charge == (float(0.2 + x)));
                BOOST_TEST(particle.pos[0] == (float(0.3 + x) + x * 1000));
                BOOST_TEST(particle.pos[1] == (float(0.4 + y) + y * 2000));
                BOOST_TEST(particle.pos[2] == (float(0.5 + z) + z * 3000));
                TEST_REAL_ACCURACY(particle.vel[0], (float(0.6 + x) * 3 * 6), 0.00001);
                TEST_REAL_ACCURACY(particle.vel[1], (float(0.7 + y) * 4 * 7), 0.00001);
                TEST_REAL_ACCURACY(particle.vel[2], (float(0.9 + z) * 5 * 8), 0.00001);
                BOOST_TEST(particle.state  == int(x + y + z));
            }
        }
    }

    grid.callback(OffsetPositionFunctionStyle(dim_x, dim_y, dim_z));

    for (long z = 0; z < dim_z; ++z) {
        for (long y = 0; y < dim_y; ++y) {
            for (long x = 0; x < dim_x; ++x) {
                ArrayParticle particle = grid.get(x, y, z);

                BOOST_TEST(particle.mass   == float(0.1 + x));
                BOOST_TEST(particle.charge == (float(0.2 + x)));
                BOOST_TEST(particle.pos[0] == (float(0.3 + x) + x * 1000 + x * 1001));
                BOOST_TEST(particle.pos[1] == (float(0.4 + y) + y * 2000 + y * 2001));
                BOOST_TEST(particle.pos[2] == (float(0.5 + z) + z * 3000 + z * 3001));
                TEST_REAL_ACCURACY(particle.vel[0], (float(0.6 + x) * 3 * 6), 0.00001);
                TEST_REAL_ACCURACY(particle.vel[1], (float(0.7 + y) * 4 * 7), 0.00001);
                TEST_REAL_ACCURACY(particle.vel[2], (float(0.9 + z) * 5 * 8), 0.00001);
                BOOST_TEST(particle.state  == int(x + y + z));
            }
        }
    }
}

ADD_TEST(TestArrayMemberLoadSave)
{
    soa_grid<CellWithArrayMember> grid(10, 10, 10);
    std::vector<char> store0(3 * 40 * sizeof(double));
    std::vector<char> store1 = store0;
    BOOST_TEST(store0 == store1);

    double *storeA = reinterpret_cast<double*>(&store0[0]);
    for (int j = 0; j < 40; ++j) {
        for (int i = 0; i < 3; ++i) {
            storeA[j * 3 + i] = j * 1000 + i;
        }
    }
    BOOST_TEST(store0 != store1);

    grid.load(5, 2, 1, &store0[0], 3);
    grid.save(5, 2, 1, &store1[0], 3);
    BOOST_TEST(store0 == store1);

    storeA = reinterpret_cast<double*>(&store1[0]);
    for (int j = 0; j < 40; ++j) {
        for (int i = 0; i < 3; ++i) {
            double expected = j * 1000 + i;
            double actual = storeA[j * 3 + i];
            BOOST_TEST(expected == actual);
        }
    }
}

ADD_TEST(TestNonTrivialMembers)
{
    CellWithNonTrivialMembers cell1;
    CellWithNonTrivialMembers cell2;
    cell1.map[5] = std::vector<double>(4711, 47.11);
    cell2.map[7] = std::vector<double>( 666, 66.66);
    {
        // fill memory with non-zero values...
        soa_grid<HeatedGameOfLifeCell> grid1(63, 63, 120);
        std::fill(grid1.data(), grid1.data() + grid1.byte_size(), char(1));
    }
    int counter = DestructionCounterClass::count;
    {
        soa_grid<CellWithNonTrivialMembers> grid1(3, 4, 5);
        // ...so that deallocation of memory upon assignment of maps
        // here will fail. Memory initialized to 0 might make the maps
        // inside not run free() at all). The effect would be that the
        // code "accidentally" works.
        grid1.set(1, 1, 1, cell1);
        grid1.set(1, 1, 1, cell2);
    }
    // ensure d-tor got called
    size_t expected = 3 * 4 * 5 + counter;
    BOOST_TEST(expected == DestructionCounterClass::count);
}

ADD_TEST(TestNonTrivialMembers2)
{
    CellWithNonTrivialMembers cell1;
    cell1.map[5] = std::vector<double>(4711, 47.11);
    CellWithNonTrivialMembers cell2;
    cell1.map[7] = std::vector<double>(666, 1.1);
    {
        soa_grid<CellWithNonTrivialMembers> grid1(3, 3, 3);
        soa_grid<CellWithNonTrivialMembers> grid2(3, 3, 3);

        grid1.set(1, 1, 1, cell1);
        grid2 = grid1;
        // this ensures no bit-wise copy was done in the assignment
        // above. It it had been done then the two copy assignments
        // below would cause a double free error below:
        grid1.set(1, 1, 1, cell2);
        grid2.set(1, 1, 1, cell2);
    }
}

ADD_TEST(TestNonTrivialMembers3)
{
    CellWithNonTrivialMembers cell1;
    cell1.map[5] = std::vector<double>(4711, 47.11);
    CellWithNonTrivialMembers cell2;
    cell1.map[7] = std::vector<double>(666, 1.1);
    {
        soa_grid<CellWithNonTrivialMembers> grid1(3, 3, 3);
        grid1.set(1, 1, 1, cell1);
        soa_grid<CellWithNonTrivialMembers> grid2(grid1);

        // this ensures no bit-wise copy was done in the assignment
        // above. It it had been done then the two copy assignments
        // below would cause a double free error below:
        grid1.set(1, 1, 1, cell2);
        grid2.set(1, 1, 1, cell2);
    }
}

ADD_TEST(TestNonTrivialMembers4)
{
    CellWithNonTrivialMembers cell1;
    cell1.map[5] = std::vector<double>(4711, 47.11);
    CellWithNonTrivialMembers cell2;
    cell1.map[7] = std::vector<double>(666, 1.1);
    {
        soa_grid<CellWithNonTrivialMembers> grid1(3, 3, 3);
        const soa_grid<CellWithNonTrivialMembers>& grid_const_ref(grid1);
        grid1.set(1, 1, 1, cell1);
        soa_grid<CellWithNonTrivialMembers> grid2(grid_const_ref);

        // this ensures no bit-wise copy was done in the assignment
        // above. It it had been done then the two copy assignments
        // below would cause a double free error below:
        grid1.set(1, 1, 1, cell2);
        grid2.set(1, 1, 1, cell2);
    }
}

ADD_TEST(TestCopyConstructor1)
{
    soa_grid<HeatedGameOfLifeCell> grid1(20, 10, 1);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            grid1.set(x, y, 0, HeatedGameOfLifeCell(y * 100 + x, true));
        }
    }

    soa_grid<HeatedGameOfLifeCell> grid2(grid1);

    // overwrite old grid to ensure both are still separate
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            grid1.set(x, y, 0, HeatedGameOfLifeCell(-1, false));
        }
    }

    BOOST_TEST(grid1.dim_x() == 20);
    BOOST_TEST(grid1.dim_y() == 10);
    BOOST_TEST(grid1.dim_z() ==  1);

    BOOST_TEST_EQ(grid1.extent_x(), 32);
    BOOST_TEST_EQ(grid1.extent_y(), 32);
    BOOST_TEST_EQ(grid1.extent_z(),  1);

    BOOST_TEST(grid2.dim_x() == 20);
    BOOST_TEST(grid2.dim_y() == 10);
    BOOST_TEST(grid2.dim_z() ==  1);

    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            HeatedGameOfLifeCell cell = grid2.get(x, y, 0);
            BOOST_TEST(cell == HeatedGameOfLifeCell(y * 100 + x, true));
            cell = grid1.get(x, y, 0);
            BOOST_TEST(cell == HeatedGameOfLifeCell(-1, false));
        }
    }
}

ADD_TEST(TestCopyConstructor2)
{
    soa_grid<HeatedGameOfLifeCell> grid1(20, 10, 1);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            grid1.set(x, y, 0, HeatedGameOfLifeCell(y * 100 + x, true));
        }
    }

    const soa_grid<HeatedGameOfLifeCell>& grid_temp(grid1);
    soa_grid<HeatedGameOfLifeCell> grid2(grid_temp);

    // overwrite old grid to ensure both are still separate
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            grid1.set(x, y, 0, HeatedGameOfLifeCell(-1, false));
        }
    }

    BOOST_TEST(grid1.dim_x() == 20);
    BOOST_TEST(grid1.dim_y() == 10);
    BOOST_TEST(grid1.dim_z() ==  1);

    BOOST_TEST(grid2.dim_x() == 20);
    BOOST_TEST(grid2.dim_y() == 10);
    BOOST_TEST(grid2.dim_z() ==  1);

    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            HeatedGameOfLifeCell cell = grid2.get(x, y, 0);
            BOOST_TEST(cell == HeatedGameOfLifeCell(y * 100 + x, true));
            cell = grid1.get(x, y, 0);
            BOOST_TEST(cell == HeatedGameOfLifeCell(-1, false));
        }
    }
}

ADD_TEST(TestBroadcast)
{
    soa_grid<HeatedGameOfLifeCell> grid(20, 10, 1);
    HeatedGameOfLifeCell hotCell(200);

    grid.broadcast(5, 7, 0, hotCell, 10);

    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 20; ++x) {
            HeatedGameOfLifeCell actual = grid.get(x, y, 0);
            HeatedGameOfLifeCell expected(0, false);
            if ((y == 7) && (x >= 5) && (x < 15)) {
                expected.temperature = 200;
            }

            BOOST_TEST_EQ(actual, expected);
        }
    }
}

ADD_TEST(TestDefaultSizesFor1D2D3D)
{
    {
        // should require approx. 1 GB RAM:
        soa_grid<HeatedGameOfLifeCell> grid3D(500, 500, 500);
    }

    {
        // should require approx. 1 GB RAM, but would fail if allocated as
        // 3D grid (in that case memory requirement would be 9 TB).
        soa_grid<HeatedGameOfLifeCell> grid2D(10000, 10000, 1);
    }

    {
        // should require approx. 1 GB RAM, but would fail if allocated as
        // 2D grid (in that case memory requirement would be 80 PB). Worse
        // for 3D.
        soa_grid<HeatedGameOfLifeCell> grid1D(100000000, 1, 1);
    }
}

}

int main(int /* argc */, char ** /* argv */)
{
    return 0;
}
