/**
 * Copyright 2013 - 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/detail/lightweight_test.hpp>
#include <iostream>
#include <typeinfo>
#include <libflatarray/flat_array.hpp>
#include <vector>

#include "test.h"

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

LIBFLATARRAY_REGISTER_SOA(HeatedGameOfLifeCell, ((double)(temperature))((bool)(alive)))

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

LIBFLATARRAY_REGISTER_SOA(CellWithMultipleMembersOfSameType, ((double)(memberA))((double)(memberB))((double)(memberC)))

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
    MemberAccessChecker1(int dimX, int dimY, int dimZ) :
	dimX(dimX),
	dimY(dimY),
	dimZ(dimZ)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR& accessor, int *index) const
    {
        for (int z = 0; z < dimZ; ++z) {
            for (int y = 0; y < dimY; ++y) {
                for (int x = 0; x < dimX; ++x) {
                    *index =
                        ACCESSOR::DIM_X * ACCESSOR::DIM_Y * z +
                        ACCESSOR::DIM_X * y +
                        x;

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
    int dimX;
    int dimY;
    int dimZ;
};

class MemberAccessChecker2
{
public:
    MemberAccessChecker2(int dimX, int dimY, int dimZ) :
	dimX(dimX),
	dimY(dimY),
	dimZ(dimZ)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR& accessor, int *index) const
    {
        for (int z = 0; z < dimZ; ++z) {
            for (int y = 0; y < dimY; ++y) {
                for (int x = 0; x < dimX; ++x) {
                    *index =
                        ACCESSOR::DIM_X * ACCESSOR::DIM_Y * z +
                        ACCESSOR::DIM_X * y +
                        x;

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
    int dimX;
    int dimY;
    int dimZ;
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
    void operator()(ACCESSOR& accessor, int *index) const
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

namespace LibFlatArray {

ADD_TEST(TestSingleGetSet)
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

ADD_TEST(TestArrayGetSet)
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

ADD_TEST(TestResizeAndByteSize)
{
    int dimX = 2;
    int dimY = 2;
    int dimZ = 2;
    int cellSize = 9; // 1 double, 1 bool
    soa_grid<HeatedGameOfLifeCell> grid(dimX, dimY, dimZ);
    BOOST_TEST(grid.byte_size() == (32 * 32 * 32 * cellSize));

    dimX = 10;
    dimY = 20;
    dimZ = 40;
    grid.resize(dimX, dimY, dimZ);
    grid.set(dimX - 1, dimY - 1, dimZ - 1, HeatedGameOfLifeCell(4711));
    BOOST_TEST(grid.get(dimX - 1, dimY - 1, dimZ - 1) == HeatedGameOfLifeCell(4711));
    BOOST_TEST(grid.byte_size() == (32 * 32 * 256 * cellSize));

}

ADD_TEST(TestSingleCallback)
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
    grid.callback(InvertTemperature(dimX, dimY, dimZ), index);

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

ADD_TEST(TestDualCallback)
{
    testDualCallback<CopyTemperatureCactusStyle>();
    testDualCallback<CopyTemperatureNativeStyle>();
}

ADD_TEST(TestAssignment)
{
    soa_grid<HeatedGameOfLifeCell> gridOld(20, 30, 40);
    soa_grid<HeatedGameOfLifeCell> gridNew(70, 60, 50);

    BOOST_TEST(gridOld.data != gridNew.data);
    BOOST_TEST(gridOld.dim_x != gridNew.dim_x);
    BOOST_TEST(gridOld.dim_y != gridNew.dim_y);
    BOOST_TEST(gridOld.dim_z != gridNew.dim_z);
    BOOST_TEST(gridOld.my_byte_size != gridNew.my_byte_size);

    gridOld = gridNew;

    BOOST_TEST(gridOld.data != gridNew.data);
    BOOST_TEST(gridOld.dim_x == gridNew.dim_x);
    BOOST_TEST(gridOld.dim_y == gridNew.dim_y);
    BOOST_TEST(gridOld.dim_z == gridNew.dim_z);
    BOOST_TEST(gridOld.my_byte_size == gridNew.my_byte_size);
}

ADD_TEST(TestSwap)
{
    int dimX = 5;
    int dimY = 3;
    int dimZ = 2;

    soa_grid<HeatedGameOfLifeCell> gridOld(dimX, dimY, dimZ);
    soa_grid<HeatedGameOfLifeCell> gridNew(dimX, dimY, dimZ);

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                gridOld.set(x, y, z, HeatedGameOfLifeCell(4711));
                gridNew.set(x, y, z, HeatedGameOfLifeCell(666));
            }
        }
    }

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
		double temp = z * 100 + y + x * 0.01;
		BOOST_TEST(gridOld.get(x, y, z).temperature == 4711);
		BOOST_TEST(gridNew.get(x, y, z).temperature == 666);
            }
        }
    }

    std::swap(gridOld, gridNew);

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
		double temp = z * 100 + y + x * 0.01;
		BOOST_TEST(gridOld.get(x, y, z).temperature == 666);
		BOOST_TEST(gridNew.get(x, y, z).temperature == 4711);
            }
        }
    }
}

ADD_TEST(TestCopyArrayIn)
{
    int dimX = 5;
    int dimY = 3;
    int dimZ = 2;

    soa_grid<HeatedGameOfLifeCell> grid(dimX, dimY, dimZ);
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
    int dimX = 5;
    int dimY = 3;
    int dimZ = 2;

    std::vector<char> store0(1024);
    double *store1 = reinterpret_cast<double*>(&store0[0]);
    soa_grid<HeatedGameOfLifeCell> grid(dimX, dimY, dimZ);

    grid.set(0, 0, 0, HeatedGameOfLifeCell(47.11, true));
    grid.set(1, 0, 0, HeatedGameOfLifeCell(1.234, false));
    grid.set(2, 0, 0, HeatedGameOfLifeCell(666.1, true));
    grid.save(0, 0, 0, &store0[0], 3);

    BOOST_TEST(store1[ 0] == 47.11);
    BOOST_TEST(store1[ 1] == 1.234);
    BOOST_TEST(store1[ 2] == 666.1);
    BOOST_TEST(store0[24] == true);
    BOOST_TEST(store0[25] == false);
    BOOST_TEST(store0[26] == true);

    grid.set(3, 2, 1, HeatedGameOfLifeCell(2.345, false));
    grid.set(4, 2, 1, HeatedGameOfLifeCell(987.6, true));
    grid.save(3, 2, 1, &store0[0], 2);

    BOOST_TEST(store1[ 0] == 2.345);
    BOOST_TEST(store1[ 1] == 987.6);
    BOOST_TEST(store0[16] == false);
    BOOST_TEST(store0[17] == true);
}

ADD_TEST(TestNumberOfMembers)
{
    BOOST_TEST(number_of_members<HeatedGameOfLifeCell>::VALUE == 2);
}

ADD_TEST(TestAggregatedMemberSize)
{
    BOOST_TEST(sizeof(HeatedGameOfLifeCell) == 16);
    BOOST_TEST(aggregated_member_size<HeatedGameOfLifeCell>::VALUE == 9);
}

ADD_TEST(TestAccessMember)
{
    int dimX = 15;
    int dimY = 13;
    int dimZ = 19;

    soa_grid<HeatedGameOfLifeCell> grid(dimX, dimY, dimZ);

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                grid.set(x, y, z, HeatedGameOfLifeCell(x * 1000.0 + y + z * 0.001, (x % 2 == 0)));
            }
        }
    }

    BOOST_TEST(grid.get(11, 12, 13).temperature == 11012.013);
    BOOST_TEST(grid.get(10, 12, 13).alive == true);
    BOOST_TEST(grid.get(11, 12, 13).alive == false);
    BOOST_TEST(grid.get(12, 12, 13).alive == true);

    grid.callback(MemberAccessChecker1(dimX, dimY, dimZ));
    grid.callback(MemberAccessChecker2(dimX, dimY, dimZ));
}

ADD_TEST(TestMemberPtrToOffset)
{
    BOOST_TEST( 0 == member_ptr_to_offset()(&HeatedGameOfLifeCell::temperature));
    BOOST_TEST( 8 == member_ptr_to_offset()(&HeatedGameOfLifeCell::alive));

    BOOST_TEST( 0 == member_ptr_to_offset()(&CellWithMultipleMembersOfSameType::memberA));
    BOOST_TEST( 8 == member_ptr_to_offset()(&CellWithMultipleMembersOfSameType::memberB));
    BOOST_TEST(16 == member_ptr_to_offset()(CellWithMultipleMembersOfSameType::getMemberCPointer()));
}

}

int main(int argc, char **argv)
{
    return 0;
}
