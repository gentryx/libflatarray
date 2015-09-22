/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <iostream>
#include <libflatarray/flat_array.hpp>
#include <libflatarray/short_vec.hpp>
#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

#define WEIGHT_S 0.11
#define WEIGHT_T 0.12
#define WEIGHT_W 0.13
#define WEIGHT_C 0.20
#define WEIGHT_E 0.15
#define WEIGHT_B 0.16
#define WEIGHT_N 0.17
#define DELTA_T 0.001
#define SOFTENING 0.1

using namespace LibFlatArray;

class JacobiD3Q7 : public cpu_benchmark
{
public:
    std::string family()
    {
        return "JacobiD3Q7";
    }

    std::string unit()
    {
        return "GLUPS";
    }

    double glups(std::vector<int> dim, int steps, double time) const
    {
        double updates = steps;
        for (std::size_t i = 0; i < dim.size(); ++i) {
            updates *= dim[i] - 2;
        }

        double gigaLatticeUpdatesPerSecond = updates / time * 1e-9;
        return gigaLatticeUpdatesPerSecond;
    }
};

class JacobiD3Q7Vanilla : public JacobiD3Q7
{
public:
    std::string species()
    {
        return "vanilla";
    }

    double performance(std::vector<int> dim)
    {
        int dim_x = dim[0];
        int dim_y = dim[1];
        int dim_z = dim[2];
        int maxT = 200000000 / dim_x / dim_y / dim_z;
        maxT = std::max(16, maxT);

        int offsetZ = dim_x * dim_y;
        int gridVolume = dim_x * dim_y * dim_z;
        std::vector<double> compressedGrid(2 * gridVolume);
        double *gridOld = &compressedGrid[0];
        double *gridNew = &compressedGrid[gridVolume];

        for (int z = 0; z < dim_z; ++z) {
            for (int y = 0; y < dim_y; ++y) {
                for (int x = 0; x < dim_x; ++x) {
                    gridOld[z * offsetZ + y * dim_y + x] = x + y + z;
                    gridNew[z * offsetZ + y * dim_y + x] = x + y + z;
                }
            }
        }

        double tStart = time();

        for (int t = 0; t < maxT; ++t) {
            for (int z = 1; z < (dim_z - 1); ++z) {
                for (int y = 1; y < (dim_y - 1); ++y) {
                    updateLine(gridOld, gridNew, 1, y, z, dim_x - 1, dim_x, offsetZ);
                }
            }
        }

        double tEnd = time();

        if (gridOld[1 * offsetZ + 1 * dim_y + 1] ==
            gridNew[1 * offsetZ + 1 * dim_y + 1]) {
            std::cout << "this is a debug statement to prevent the compiler from optimizing away the update routine\n";
        }

        return glups(dim, maxT, tEnd - tStart);
    }

private:
    void updateLine(double *gridOld, double *gridNew,
                    const int xStart, const int y,       const int z,
                    const int xEnd,   const int offsetY, const int offsetZ) const
    {
        for (int x = xStart; x < xEnd; ++x) {
            gridNew[x + y * offsetY + z * offsetZ] =
                gridOld[x + y * offsetY + z * offsetZ - 1 * offsetZ] * WEIGHT_S +
                gridOld[x + y * offsetY + z * offsetZ - 1 * offsetY] * WEIGHT_T +
                gridOld[x + y * offsetY + z * offsetZ - 1          ] * WEIGHT_W +
                gridOld[x + y * offsetY + z * offsetZ + 0          ] * WEIGHT_C +
                gridOld[x + y * offsetY + z * offsetZ + 1          ] * WEIGHT_E +
                gridOld[x + y * offsetY + z * offsetZ + 1 * offsetY] * WEIGHT_B +
                gridOld[x + y * offsetY + z * offsetZ + 1 * offsetZ] * WEIGHT_N;
        }
    }
};

#ifdef __SSE__

class JacobiD3Q7Pepper : public JacobiD3Q7
{
public:
    std::string species()
    {
        return "pepper";
    }

    double performance(std::vector<int> dim)
    {
        int dim_x = dim[0];
        int dim_y = dim[1];
        int dim_z = dim[2];
        int maxT = 200000000 / dim_x / dim_y / dim_z;
        maxT = std::max(16, maxT);

        int offsetZ = dim_x * dim_y;
        int gridVolume = dim_x * dim_y * dim_z;
        std::vector<double> compressedGrid(2 * gridVolume);
        double *gridOld = &compressedGrid[0];
        double *gridNew = &compressedGrid[gridVolume];

        for (int z = 0; z < dim_z; ++z) {
            for (int y = 0; y < dim_y; ++y) {
                for (int x = 0; x < dim_x; ++x) {
                    gridOld[z * offsetZ + y * dim_y + x] = x + y + z;
                    gridNew[z * offsetZ + y * dim_y + x] = x + y + z;
                }
            }
        }

        double tStart = time();

        for (int t = 0; t < maxT; ++t) {
            for (int z = 1; z < (dim_z - 1); ++z) {
                for (int y = 1; y < (dim_y - 1); ++y) {
                    updateLine(gridOld, gridNew, 1, y, z, dim_x - 1, dim_x, offsetZ);
                }
            }
        }

        double tEnd = time();

        if (gridOld[1 * offsetZ + 1 * dim_y + 1] ==
            gridNew[1 * offsetZ + 1 * dim_y + 1]) {
            std::cout << "this is a debug statement to prevent the compiler from optimizing away the update routine\n";
        }

        return glups(dim, maxT, tEnd - tStart);
    }

private:
    void updateLine(double *gridOld, double *gridNew,
                    const int xStart, const int y,       const int z,
                    const int xEnd,   const int offsetY, const int offsetZ) const
    {
        __m128d factorS = _mm_set1_pd(WEIGHT_S);
        __m128d factorT = _mm_set1_pd(WEIGHT_T);
        __m128d factorW = _mm_set1_pd(WEIGHT_W);
        __m128d factorE = _mm_set1_pd(WEIGHT_E);
        __m128d factorB = _mm_set1_pd(WEIGHT_B);
        __m128d factorN = _mm_set1_pd(WEIGHT_N);

        int x = xStart;

        if (x % 2) {
            gridNew[x + y * offsetY + z * offsetZ] =
                gridOld[x + y * offsetY + z * offsetZ - 1 * offsetZ] * WEIGHT_S +
                gridOld[x + y * offsetY + z * offsetZ - 1 * offsetY] * WEIGHT_N +
                gridOld[x + y * offsetY + z * offsetZ - 1          ] * WEIGHT_W +
                gridOld[x + y * offsetY + z * offsetZ + 0          ] * WEIGHT_C +
                gridOld[x + y * offsetY + z * offsetZ + 1          ] * WEIGHT_E +
                gridOld[x + y * offsetY + z * offsetZ + 1 * offsetY] * WEIGHT_B +
                gridOld[x + y * offsetY + z * offsetZ + 1 * offsetZ] * WEIGHT_N;
            ++x;
        }

        for (; x < (xEnd - 7); x += 8) {
            // load south row:
            __m128d bufA = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ - 1 * offsetZ + 0);
            __m128d bufB = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ - 1 * offsetZ + 2);
            __m128d bufC = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ - 1 * offsetZ + 4);
            __m128d bufD = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ - 1 * offsetZ + 6);
            __m128d bufE;

            bufA = _mm_mul_pd(bufA, factorS);
            bufB = _mm_mul_pd(bufB, factorS);
            bufC = _mm_mul_pd(bufC, factorS);
            bufD = _mm_mul_pd(bufD, factorS);

            // load top row:
            __m128d sumA = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ - 1 * offsetY + 0);
            __m128d sumB = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ - 1 * offsetY + 2);
            __m128d sumC = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ - 1 * offsetY + 4);
            __m128d sumD = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ - 1 * offsetY + 6);

            sumA = _mm_mul_pd(sumA, factorT);
            sumB = _mm_mul_pd(sumB, factorT);
            sumC = _mm_mul_pd(sumC, factorT);
            sumD = _mm_mul_pd(sumD, factorT);

            sumA = _mm_add_pd(sumA, bufA);
            sumB = _mm_add_pd(sumB, bufB);
            sumC = _mm_add_pd(sumC, bufC);
            sumD = _mm_add_pd(sumD, bufD);

            // load left/right row:
            bufA = _mm_loadu_pd(gridOld + x + y * offsetY + z * offsetZ + 0 * offsetZ - 1);
            bufB = _mm_loadu_pd(gridOld + x + y * offsetY + z * offsetZ + 0 * offsetZ + 1);
            bufC = _mm_loadu_pd(gridOld + x + y * offsetY + z * offsetZ + 0 * offsetZ + 3);
            bufD = _mm_loadu_pd(gridOld + x + y * offsetY + z * offsetZ + 0 * offsetZ + 5);
            bufE = _mm_loadu_pd(gridOld + x + y * offsetY + z * offsetZ + 0 * offsetZ + 7);

            sumA = _mm_add_pd(sumA, _mm_mul_pd(bufA, factorW));
            sumB = _mm_add_pd(sumB, _mm_mul_pd(bufB, factorW));
            sumC = _mm_add_pd(sumC, _mm_mul_pd(bufC, factorW));
            sumD = _mm_add_pd(sumD, _mm_mul_pd(bufD, factorW));

            sumA = _mm_add_pd(sumA, _mm_mul_pd(bufB, factorE));
            sumB = _mm_add_pd(sumB, _mm_mul_pd(bufC, factorE));
            sumC = _mm_add_pd(sumC, _mm_mul_pd(bufD, factorE));
            sumD = _mm_add_pd(sumD, _mm_mul_pd(bufE, factorE));

            // load bottom row:
            bufA = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ + 1 * offsetY + 0);
            bufB = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ + 1 * offsetY + 2);
            bufC = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ + 1 * offsetY + 4);
            bufD = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ + 1 * offsetY + 6);

            bufA = _mm_mul_pd(bufA, factorB);
            bufB = _mm_mul_pd(bufB, factorB);
            bufC = _mm_mul_pd(bufC, factorB);
            bufD = _mm_mul_pd(bufD, factorB);

            sumA = _mm_add_pd(sumA, bufA);
            sumB = _mm_add_pd(sumB, bufB);
            sumC = _mm_add_pd(sumC, bufC);
            sumD = _mm_add_pd(sumD, bufD);

            // load north row:
            bufA = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ + 1 * offsetZ + 0);
            bufB = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ + 1 * offsetZ + 2);
            bufC = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ + 1 * offsetZ + 4);
            bufD = _mm_load_pd(gridOld + x + y * offsetY + z * offsetZ + 1 * offsetZ + 6);

            bufA = _mm_mul_pd(bufA, factorN);
            bufB = _mm_mul_pd(bufB, factorN);
            bufC = _mm_mul_pd(bufC, factorN);
            bufD = _mm_mul_pd(bufD, factorN);

            sumA = _mm_add_pd(sumA, bufA);
            sumB = _mm_add_pd(sumB, bufB);
            sumC = _mm_add_pd(sumC, bufC);
            sumD = _mm_add_pd(sumD, bufD);

            _mm_stream_pd(gridNew + x + y * offsetY + z * offsetZ + 0, sumA);
            _mm_stream_pd(gridNew + x + y * offsetY + z * offsetZ + 2, sumB);
            _mm_stream_pd(gridNew + x + y * offsetY + z * offsetZ + 4, sumC);
            _mm_stream_pd(gridNew + x + y * offsetY + z * offsetZ + 6, sumD);
        }

        for (; x < xEnd; ++x) {
            gridNew[x + y * offsetY + z * offsetZ] =
                gridOld[x + y * offsetY + z * offsetZ - 1 * offsetZ] * WEIGHT_S +
                gridOld[x + y * offsetY + z * offsetZ - 1 * offsetY] * WEIGHT_N +
                gridOld[x + y * offsetY + z * offsetZ - 1          ] * WEIGHT_W +
                gridOld[x + y * offsetY + z * offsetZ + 0          ] * WEIGHT_C +
                gridOld[x + y * offsetY + z * offsetZ + 1          ] * WEIGHT_E +
                gridOld[x + y * offsetY + z * offsetZ + 1 * offsetY] * WEIGHT_B +
                gridOld[x + y * offsetY + z * offsetZ + 1 * offsetZ] * WEIGHT_N;
        }
    }
};

#endif

class JacobiCell
{
public:
    JacobiCell(double temp = 0) :
        temp(temp)
    {}

    double temp;
};

LIBFLATARRAY_REGISTER_SOA(JacobiCell,
                          ((double)(temp)))

class JacobiD3Q7Bronze : public JacobiD3Q7
{
public:
    class UpdateFunctor
    {
    public:
        UpdateFunctor(long dim_x, long dim_y, long dim_z) :
            dim_x(dim_x),
            dim_y(dim_y),
            dim_z(dim_z)
        {}

        template<typename accessor_type1, typename accessor_type2>
        void operator()(accessor_type1& accessor1, accessor_type2& accessor2) const
        {
            for (long z = 1; z < (dim_z - 1); ++z) {
                for (long y = 1; y < (dim_y - 1); ++y) {
                    long indexStart = accessor1.gen_index(1,        y, z);
                    long indexEnd   = accessor1.gen_index(dim_x - 1, y, z);

                    for (accessor1.index = indexStart, accessor2.index = indexStart;
                         accessor1.index < indexEnd;
                         accessor1 += 1, accessor2 += 1) {

                        accessor2.temp() =
                            accessor1[coord< 0,  0, -1>()].temp() * WEIGHT_S +
                            accessor1[coord< 0, -1,  0>()].temp() * WEIGHT_T +
                            accessor1[coord<-1,  0,  0>()].temp() * WEIGHT_W +
                            accessor1[coord< 0,  0,  0>()].temp() * WEIGHT_C +
                            accessor1[coord< 1,  0,  0>()].temp() * WEIGHT_E +
                            accessor1[coord< 0,  1,  0>()].temp() * WEIGHT_B +
                            accessor1[coord< 0,  0,  1>()].temp() * WEIGHT_N;
                    }
                }
            }
        }

    private:
        long dim_x;
        long dim_y;
        long dim_z;
    };

    std::string species()
    {
        return "bronze";
    }

    double performance(std::vector<int> dim)
    {
        long dim_x = dim[0];
        long dim_y = dim[1];
        long dim_z = dim[2];
        int maxT = 200000000 / dim_x / dim_y / dim_z;
        maxT = std::max(16, maxT);

        soa_grid<JacobiCell> gridOld(dim_x, dim_y, dim_z);
        soa_grid<JacobiCell> gridNew(dim_x, dim_y, dim_z);

        for (long z = 0; z < dim_z; ++z) {
            for (long y = 0; y < dim_y; ++y) {
                for (long x = 0; x < dim_x; ++x) {
                    gridOld.set(x, y, z, x + y + z);
                    gridNew.set(x, y, z, x + y + z);
                }
            }
        }

        double tStart = time();

        UpdateFunctor functor(dim_x, dim_y, dim_z);
        for (int t = 0; t < maxT; ++t) {
            gridOld.callback(&gridNew, functor);
            std::swap(gridOld, gridNew);
        }

        double tEnd = time();

        if (gridOld.get(1, 1, 1).temp ==
            gridNew.get(1, 1, 1).temp) {
            std::cout << gridOld.get(1, 1, 1).temp << "\n";
            std::cout << gridNew.get(1, 1, 1).temp << "\n";
            std::cout << "this is a debug statement to prevent the compiler from optimizing away the update routine\n";
        }

        return glups(dim, maxT, tEnd - tStart);
    }

private:
    void updateLine(double *gridOld, double *gridNew,
                    const long xStart, const long y,       const long z,
                    const long xEnd,   const long offsetY, const long offsetZ) const
    {
        for (long x = xStart; x < xEnd; ++x) {
            gridNew[x + y * offsetY + z * offsetZ] =
                gridOld[x + y * offsetY + z * offsetZ - 1 * offsetZ] * WEIGHT_S +
                gridOld[x + y * offsetY + z * offsetZ - 1 * offsetY] * WEIGHT_T +
                gridOld[x + y * offsetY + z * offsetZ - 1          ] * WEIGHT_W +
                gridOld[x + y * offsetY + z * offsetZ + 0          ] * WEIGHT_C +
                gridOld[x + y * offsetY + z * offsetZ + 1          ] * WEIGHT_E +
                gridOld[x + y * offsetY + z * offsetZ + 1 * offsetY] * WEIGHT_B +
                gridOld[x + y * offsetY + z * offsetZ + 1 * offsetZ] * WEIGHT_N;
        }
    }
};

#ifdef __SSE__

class JacobiD3Q7Silver : public JacobiD3Q7
{
public:
    class UpdateFunctor
    {
    public:
        UpdateFunctor(long dim_x, long dim_y, long dim_z) :
            dim_x(dim_x),
            dim_y(dim_y),
            dim_z(dim_z)
        {}

        template<typename accessor_type1, typename accessor_type2>
        void operator()(accessor_type1& accessor1,
                        accessor_type2& accessor2) const
        {
            __m128d factorS = _mm_set1_pd(WEIGHT_S);
            __m128d factorT = _mm_set1_pd(WEIGHT_T);
            __m128d factorW = _mm_set1_pd(WEIGHT_W);
            __m128d factorE = _mm_set1_pd(WEIGHT_E);
            __m128d factorB = _mm_set1_pd(WEIGHT_B);
            __m128d factorN = _mm_set1_pd(WEIGHT_N);

            for (long z = 1; z < (dim_z - 1); ++z) {
                for (long y = 1; y < (dim_y - 1); ++y) {
                    long indexStart = accessor1.gen_index(1,         y, z);
                    long indexEnd   = accessor1.gen_index(dim_x - 1, y, z);

                    accessor1.index = indexStart;
                    accessor2.index = indexStart;

                    accessor2.temp() =
                        accessor1[coord< 0,  0, -1>()].temp() * WEIGHT_S +
                        accessor1[coord< 0, -1,  0>()].temp() * WEIGHT_T +
                        accessor1[coord<-1,  0,  0>()].temp() * WEIGHT_W +
                        accessor1[coord< 0,  0,  0>()].temp() * WEIGHT_C +
                        accessor1[coord< 1,  0,  0>()].temp() * WEIGHT_E +
                        accessor1[coord< 0,  1,  0>()].temp() * WEIGHT_B +
                        accessor1[coord< 0,  0,  1>()].temp() * WEIGHT_N;

                    accessor1.index += 1;
                    accessor2.index += 1;

                    for (;
                         accessor1.index < (indexEnd - 7);
                         accessor1.index += 8, accessor2.index += 8) {

                        // load south row:
                        __m128d bufA = _mm_load_pd(&accessor1[coord<0, 0, -1>()].temp() + 0);
                        __m128d bufB = _mm_load_pd(&accessor1[coord<0, 0, -1>()].temp() + 2);
                        __m128d bufC = _mm_load_pd(&accessor1[coord<0, 0, -1>()].temp() + 4);
                        __m128d bufD = _mm_load_pd(&accessor1[coord<0, 0, -1>()].temp() + 6);
                        __m128d bufE;

                        bufA = _mm_mul_pd(bufA, factorS);
                        bufB = _mm_mul_pd(bufB, factorS);
                        bufC = _mm_mul_pd(bufC, factorS);
                        bufD = _mm_mul_pd(bufD, factorS);

                        // load top row:
                        __m128d sumA = _mm_load_pd(&accessor1[coord<0, -1, 0>()].temp() + 0);
                        __m128d sumB = _mm_load_pd(&accessor1[coord<0, -1, 0>()].temp() + 2);
                        __m128d sumC = _mm_load_pd(&accessor1[coord<0, -1, 0>()].temp() + 4);
                        __m128d sumD = _mm_load_pd(&accessor1[coord<0, -1, 0>()].temp() + 6);

                        sumA = _mm_mul_pd(sumA, factorT);
                        sumB = _mm_mul_pd(sumB, factorT);
                        sumC = _mm_mul_pd(sumC, factorT);
                        sumD = _mm_mul_pd(sumD, factorT);

                        sumA = _mm_add_pd(sumA, bufA);
                        sumB = _mm_add_pd(sumB, bufB);
                        sumC = _mm_add_pd(sumC, bufC);
                        sumD = _mm_add_pd(sumD, bufD);

                        // load left/right row:
                        bufA = _mm_loadu_pd(&accessor1[coord<0, 0, 0>()].temp() - 1);
                        bufB = _mm_loadu_pd(&accessor1[coord<0, 0, 0>()].temp() + 1);
                        bufC = _mm_loadu_pd(&accessor1[coord<0, 0, 0>()].temp() + 3);
                        bufD = _mm_loadu_pd(&accessor1[coord<0, 0, 0>()].temp() + 5);
                        bufE = _mm_loadu_pd(&accessor1[coord<0, 0, 0>()].temp() + 7);

                        sumA = _mm_add_pd(sumA, _mm_mul_pd(bufA, factorW));
                        sumB = _mm_add_pd(sumB, _mm_mul_pd(bufB, factorW));
                        sumC = _mm_add_pd(sumC, _mm_mul_pd(bufC, factorW));
                        sumD = _mm_add_pd(sumD, _mm_mul_pd(bufD, factorW));

                        sumA = _mm_add_pd(sumA, _mm_mul_pd(bufB, factorE));
                        sumB = _mm_add_pd(sumB, _mm_mul_pd(bufC, factorE));
                        sumC = _mm_add_pd(sumC, _mm_mul_pd(bufD, factorE));
                        sumD = _mm_add_pd(sumD, _mm_mul_pd(bufE, factorE));

                        // load bottom row:
                        bufA = _mm_load_pd(&accessor1[coord<0, 1, 0>()].temp() + 0);
                        bufB = _mm_load_pd(&accessor1[coord<0, 1, 0>()].temp() + 2);
                        bufC = _mm_load_pd(&accessor1[coord<0, 1, 0>()].temp() + 4);
                        bufD = _mm_load_pd(&accessor1[coord<0, 1, 0>()].temp() + 6);

                        bufA = _mm_mul_pd(bufA, factorB);
                        bufB = _mm_mul_pd(bufB, factorB);
                        bufC = _mm_mul_pd(bufC, factorB);
                        bufD = _mm_mul_pd(bufD, factorB);

                        sumA = _mm_add_pd(sumA, bufA);
                        sumB = _mm_add_pd(sumB, bufB);
                        sumC = _mm_add_pd(sumC, bufC);
                        sumD = _mm_add_pd(sumD, bufD);

                        // load north row:
                        bufA = _mm_load_pd(&accessor1[coord<0, 0, 1>()].temp() + 0);
                        bufB = _mm_load_pd(&accessor1[coord<0, 0, 1>()].temp() + 2);
                        bufC = _mm_load_pd(&accessor1[coord<0, 0, 1>()].temp() + 4);
                        bufD = _mm_load_pd(&accessor1[coord<0, 0, 1>()].temp() + 6);

                        bufA = _mm_mul_pd(bufA, factorN);
                        bufB = _mm_mul_pd(bufB, factorN);
                        bufC = _mm_mul_pd(bufC, factorN);
                        bufD = _mm_mul_pd(bufD, factorN);

                        sumA = _mm_add_pd(sumA, bufA);
                        sumB = _mm_add_pd(sumB, bufB);
                        sumC = _mm_add_pd(sumC, bufC);
                        sumD = _mm_add_pd(sumD, bufD);

                        _mm_stream_pd(&accessor2[coord<0, 0, 0>()].temp() + 0, sumA);
                        _mm_stream_pd(&accessor2[coord<0, 0, 0>()].temp() + 2, sumB);
                        _mm_stream_pd(&accessor2[coord<0, 0, 0>()].temp() + 4, sumC);
                        _mm_stream_pd(&accessor2[coord<0, 0, 0>()].temp() + 6, sumD);
                    }


                    for (;
                         accessor1.index < (indexEnd - 1);
                         accessor1.index += 1, accessor2.index += 1) {
                        accessor2.temp() =
                            accessor1[coord< 0,  0, -1>()].temp() * WEIGHT_S +
                            accessor1[coord< 0, -1,  0>()].temp() * WEIGHT_T +
                            accessor1[coord<-1,  0,  0>()].temp() * WEIGHT_W +
                            accessor1[coord< 0,  0,  0>()].temp() * WEIGHT_C +
                            accessor1[coord< 1,  0,  0>()].temp() * WEIGHT_E +
                            accessor1[coord< 0,  1,  0>()].temp() * WEIGHT_B +
                            accessor1[coord< 0,  0,  1>()].temp() * WEIGHT_N;

                    }
                }
            }
        }

    private:
        long dim_x;
        long dim_y;
        long dim_z;
    };

    std::string species()
    {
        return "silver";
    }

    double performance(std::vector<int> dim)
    {
        long dim_x = dim[0];
        long dim_y = dim[1];
        long dim_z = dim[2];
        int maxT = 200000000 / dim_x / dim_y / dim_z;
        maxT = std::max(16, maxT);

        soa_grid<JacobiCell> gridOld(dim_x, dim_y, dim_z);
        soa_grid<JacobiCell> gridNew(dim_x, dim_y, dim_z);

        for (long z = 0; z < dim_z; ++z) {
            for (long y = 0; y < dim_y; ++y) {
                for (long x = 0; x < dim_x; ++x) {
                    gridOld.set(x, y, z, x + y + z);
                    gridNew.set(x, y, z, x + y + z);
                }
            }
        }

        double tStart = time();

        UpdateFunctor functor(dim_x, dim_y, dim_z);
        for (int t = 0; t < maxT; ++t) {
            gridOld.callback(&gridNew, functor);
            std::swap(gridOld, gridNew);
        }

        double tEnd = time();

        if (gridOld.get(20, 20, 20).temp ==
            gridNew.get(10, 10, 10).temp) {
            std::cout << "this is a debug statement to prevent the compiler from optimizing away the update routine\n";
        }

        return glups(dim, maxT, tEnd - tStart);
    }

private:
    void updateLine(double *gridOld, double *gridNew,
                    const long xStart, const long y,       const long z,
                    const long xEnd,   const long offsetY, const long offsetZ) const
    {
        for (long x = xStart; x < xEnd; ++x) {
            gridNew[x + y * offsetY + z * offsetZ] =
                gridOld[x + y * offsetY + z * offsetZ - 1 * offsetZ] * WEIGHT_S +
                gridOld[x + y * offsetY + z * offsetZ - 1 * offsetY] * WEIGHT_T +
                gridOld[x + y * offsetY + z * offsetZ - 1          ] * WEIGHT_W +
                gridOld[x + y * offsetY + z * offsetZ + 0          ] * WEIGHT_C +
                gridOld[x + y * offsetY + z * offsetZ + 1          ] * WEIGHT_E +
                gridOld[x + y * offsetY + z * offsetZ + 1 * offsetY] * WEIGHT_B +
                gridOld[x + y * offsetY + z * offsetZ + 1 * offsetZ] * WEIGHT_N;
        }
    }
};

#endif

class Particle
{
public:
    inline Particle(
        const float posX = 0,
        const float posY = 0,
        const float posZ = 0,
        const float velX = 0,
        const float velY = 0,
        const float velZ = 0,
        const float charge = 0) :
        posX(posX),
        posY(posY),
        posZ(posZ),
        velX(velX),
        velY(velY),
        velZ(velZ),
        charge(charge)
    {}

    float posX;
    float posY;
    float posZ;
    float velX;
    float velY;
    float velZ;
    float charge;
};

LIBFLATARRAY_REGISTER_SOA(Particle,
                          ((float)(posX))
                          ((float)(posY))
                          ((float)(posZ))
                          ((float)(velX))
                          ((float)(velY))
                          ((float)(velZ))
                          ((float)(charge)))

class ArrayParticle
{
public:
    inline ArrayParticle(
        const float posX = 0,
        const float posY = 0,
        const float posZ = 0,
        const float velX = 0,
        const float velY = 0,
        const float velZ = 0,
        const float charge = 0) :
        charge(charge)
    {
        pos[0] = posX;
        pos[1] = posY;
        pos[2] = posZ;
        vel[0] = velX;
        vel[1] = velY;
        vel[2] = velZ;
    }

    float pos[3];
    float vel[3];
    float charge;
};

LIBFLATARRAY_REGISTER_SOA(ArrayParticle,
                          ((float)(pos)(3))
                          ((float)(vel)(3))
                          ((float)(charge)))

class NBody : public cpu_benchmark
{
public:
    std::string family()
    {
        return "NBody";
    }

    std::string unit()
    {
        return "GFLOPS";
    }

    double gflops(double numParticles, double repeats, double tStart, double tEnd)
    {
        double flops = repeats * numParticles * (9 + numParticles * (3 + 6 + 5 + 3 + 3));
        double gflops = flops / (tEnd - tStart) * 1e-9;
        return gflops;
    }
};

class NBodyVanilla : public NBody
{
public:
    std::string species()
    {
        return "vanilla";
    }

    double performance(std::vector<int> dim)
    {
        int numParticles = dim[0];
        int repeats = dim[1];

        std::vector<Particle> particlesA;
        std::vector<Particle> particlesB;
        particlesA.reserve(numParticles);
        particlesB.reserve(numParticles);

        for (int i = 0; i < numParticles; ++i) {
            Particle p(
                i, i * i, sin(i),
                i % 11, i % 13, i % 19,
                10 + cos(2 * i));

            particlesA.push_back(p);
            particlesB.push_back(p);
        }

        double tStart = time();

        for (int t = 0; t < repeats; ++t) {
            for (int i = 0; i < numParticles; ++i) {
                float posX = particlesA[i].posX;
                float posY = particlesA[i].posY;
                float posZ = particlesA[i].posZ;

                float velX = particlesA[i].velX;
                float velY = particlesA[i].velY;
                float velZ = particlesA[i].velZ;

                float charge = particlesA[i].charge;

                float accelerationX = 0;
                float accelerationY = 0;
                float accelerationZ = 0;

                for (int j = 0; j < numParticles; ++j) {
                    float deltaX = posX - particlesA[j].posX;
                    float deltaY = posY - particlesA[j].posY;
                    float deltaZ = posZ - particlesA[j].posZ;
                    float distance2 = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ + SOFTENING;
                    float factor = charge * particlesA[j].charge * DELTA_T / distance2 / sqrt(distance2);
                    float forceX = deltaX * factor;
                    float forceY = deltaY * factor;
                    float forceZ = deltaZ * factor;

                    accelerationX += forceX;
                    accelerationY += forceY;
                    accelerationZ += forceZ;
                }

                particlesB[i].posX = posX + velX * DELTA_T;
                particlesB[i].posY = posY + velY * DELTA_T;
                particlesB[i].posZ = posZ + velZ * DELTA_T;

                particlesB[i].velX = velX + accelerationX;
                particlesB[i].velY = velY + accelerationY;
                particlesB[i].velZ = velZ + accelerationZ;

                particlesB[i].charge = charge;
            }

            std::swap(particlesA, particlesB);
        }

        double tEnd = time();

        if (particlesA[0].posX == 0.12345) {
            std::cout << "this is a debug statement to prevent the compiler from optimizing away the update routine\n";
        }

        return gflops(numParticles, repeats, tStart, tEnd);
    }
};

#ifdef __AVX__

class NBodyPepper : public NBody
{
public:
    std::string species()
    {
        return "pepper";
    }

    double performance(std::vector<int> dim)
    {
        int numParticles = dim[0];
        int repeats = dim[1];

        std::vector<float> posXA;
        std::vector<float> posYA;
        std::vector<float> posZA;
        std::vector<float> posXB;
        std::vector<float> posYB;
        std::vector<float> posZB;

        std::vector<float> velXA;
        std::vector<float> velYA;
        std::vector<float> velZA;
        std::vector<float> velXB;
        std::vector<float> velYB;
        std::vector<float> velZB;

        std::vector<float> chargeA;
        std::vector<float> chargeB;

        for (int i = 0; i < numParticles; ++i) {
            Particle p(
                i, i * i, sin(i),
                i % 11, i % 13, i % 19,
                10 + cos(2 * i));

            posXA.push_back(p.posX);
            posXB.push_back(p.posX);
            posYA.push_back(p.posY);
            posYB.push_back(p.posY);
            posZA.push_back(p.posZ);
            posZB.push_back(p.posZ);

            velXA.push_back(p.velX);
            velXB.push_back(p.velX);
            velYA.push_back(p.velY);
            velYB.push_back(p.velY);
            velZA.push_back(p.velZ);
            velZB.push_back(p.velZ);

            chargeA.push_back(p.charge);
            chargeB.push_back(p.charge);
        }

        double tStart = time();

        for (int t = 0; t < repeats; ++t) {
            for (int i = 0; i < numParticles; ++i) {
                int j;

                float posX = posXA[i];
                float posY = posYA[i];
                float posZ = posZA[i];
                __m256 posXV = _mm256_set1_ps(posX);
                __m256 posYV = _mm256_set1_ps(posY);
                __m256 posZV = _mm256_set1_ps(posZ);

                float velX = velXA[i];
                float velY = velYA[i];
                float velZ = velZA[i];

                float charge = chargeA[i];
                __m256 chargeV = _mm256_set1_ps(charge);

                __m256 accelerationXV = _mm256_set1_ps(0);
                __m256 accelerationYV = _mm256_set1_ps(0);
                __m256 accelerationZV = _mm256_set1_ps(0);

                for (j = 0; j < (numParticles - 7); j += 8) {
                    __m256 deltaX = _mm256_sub_ps(posXV, _mm256_loadu_ps(&posXA[j]));
                    __m256 deltaY = _mm256_sub_ps(posYV, _mm256_loadu_ps(&posYA[j]));
                    __m256 deltaZ = _mm256_sub_ps(posZV, _mm256_loadu_ps(&posZA[j]));
                    __m256 distance2 =
                        _mm256_add_ps(_mm256_mul_ps(deltaX, deltaX),
                                      _mm256_mul_ps(deltaY, deltaY));
                    distance2 =
                        _mm256_add_ps(_mm256_mul_ps(deltaZ, deltaZ),
                                      distance2);
                    distance2 =
                        _mm256_add_ps(_mm256_set1_ps(SOFTENING),
                                      distance2);

                    __m256 factor = _mm256_mul_ps(chargeV, _mm256_loadu_ps(&chargeA[j]));
                    factor = _mm256_mul_ps(factor, _mm256_set1_ps(DELTA_T));
                    factor = _mm256_mul_ps(factor, _mm256_rcp_ps(distance2));
                    factor = _mm256_mul_ps(factor, _mm256_rsqrt_ps(distance2));

                    __m256 forceX = _mm256_mul_ps(deltaX, factor);
                    __m256 forceY = _mm256_mul_ps(deltaY, factor);
                    __m256 forceZ = _mm256_mul_ps(deltaZ, factor);

                    accelerationXV = _mm256_add_ps(accelerationXV, forceX);
                    accelerationYV = _mm256_add_ps(accelerationYV, forceY);
                    accelerationZV = _mm256_add_ps(accelerationZV, forceZ);
                }

                float buf[8];
                _mm256_store_ps(buf, accelerationXV);
                float accelerationX = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
                _mm256_store_ps(buf, accelerationYV);
                float accelerationY = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
                _mm256_store_ps(buf, accelerationZV);
                float accelerationZ = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

                for (; j < numParticles; ++j) {
                    float deltaX = posX - posXA[j];
                    float deltaY = posY - posYA[j];
                    float deltaZ = posZ - posZA[j];
                    float distance2 = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ + SOFTENING;
                    float factor = charge * chargeA[j] * DELTA_T / distance2 / sqrt(distance2);
                    float forceX = deltaX * factor;
                    float forceY = deltaY * factor;
                    float forceZ = deltaZ * factor;

                    accelerationX += forceX;
                    accelerationY += forceY;
                    accelerationZ += forceZ;
                }

                posXB[i] = posX + velX * DELTA_T;
                posYB[i] = posY + velY * DELTA_T;
                posZB[i] = posZ + velZ * DELTA_T;

                velXB[i] = velX + accelerationX;
                velYB[i] = velY + accelerationY;
                velZB[i] = velZ + accelerationZ;

                chargeB[i] = charge;
            }

            std::swap(posXA, posXB);
            std::swap(posYA, posYB);
            std::swap(posZA, posZB);

            std::swap(velXA, velXB);
            std::swap(velYA, velYB);
            std::swap(velZA, velZB);

            std::swap(chargeA, chargeB);
        }

        double tEnd = time();

        if (posXA[0] == 0.12345) {
            std::cout << "this is a debug statement to prevent the compiler from optimizing away the update routine\n";
        }

        return gflops(numParticles, repeats, tStart, tEnd);
    }
};

class NBodyCurry : public NBody
{
public:
    std::string species()
    {
        return "curry";
    }

    double performance(std::vector<int> dim)
    {
        int numParticles = dim[0];
        int repeats = dim[1];

        std::vector<float> posXA;
        std::vector<float> posYA;
        std::vector<float> posZA;
        std::vector<float> posXB;
        std::vector<float> posYB;
        std::vector<float> posZB;

        std::vector<float> velXA;
        std::vector<float> velYA;
        std::vector<float> velZA;
        std::vector<float> velXB;
        std::vector<float> velYB;
        std::vector<float> velZB;

        std::vector<float> chargeA;
        std::vector<float> chargeB;

        for (int i = 0; i < numParticles; ++i) {
            Particle p(
                i, i * i, sin(i),
                i % 11, i % 13, i % 19,
                10 + cos(2 * i));

            posXA.push_back(p.posX);
            posXB.push_back(p.posX);
            posYA.push_back(p.posY);
            posYB.push_back(p.posY);
            posZA.push_back(p.posZ);
            posZB.push_back(p.posZ);

            velXA.push_back(p.velX);
            velXB.push_back(p.velX);
            velYA.push_back(p.velY);
            velYB.push_back(p.velY);
            velZA.push_back(p.velZ);
            velZB.push_back(p.velZ);

            chargeA.push_back(p.charge);
            chargeB.push_back(p.charge);
        }

        double tStart = time();

        for (int t = 0; t < repeats; ++t) {
            for (int i = 0; i < (numParticles - 7); i += 8) {
                int j;

                __m256 posXV = _mm256_loadu_ps(&posXA[i]);
                __m256 posYV = _mm256_loadu_ps(&posYA[i]);
                __m256 posZV = _mm256_loadu_ps(&posZA[i]);

                __m256 velXV = _mm256_loadu_ps(&velXA[i]);
                __m256 velYV = _mm256_loadu_ps(&velYA[i]);
                __m256 velZV = _mm256_loadu_ps(&velZA[i]);

                __m256 chargeV = _mm256_loadu_ps(&chargeA[i]);

                __m256 accelerationXV = _mm256_set1_ps(0);
                __m256 accelerationYV = _mm256_set1_ps(0);
                __m256 accelerationZV = _mm256_set1_ps(0);

                __m256 deltaT = _mm256_set1_ps(DELTA_T);

                for (j = 0; j < numParticles; ++j) {
                    __m256 deltaX = _mm256_sub_ps(posXV, _mm256_broadcast_ss(&posXA[j]));
                    __m256 deltaY = _mm256_sub_ps(posYV, _mm256_broadcast_ss(&posYA[j]));
                    __m256 deltaZ = _mm256_sub_ps(posZV, _mm256_broadcast_ss(&posZA[j]));
                    __m256 distance2 =
                        _mm256_add_ps(_mm256_mul_ps(deltaX, deltaX),
                                      _mm256_mul_ps(deltaY, deltaY));
                    distance2 =
                        _mm256_add_ps(_mm256_mul_ps(deltaZ, deltaZ),
                                      distance2);
                    distance2 =
                        _mm256_add_ps(_mm256_set1_ps(SOFTENING),
                                      distance2);

                    __m256 factor = _mm256_mul_ps(chargeV, _mm256_broadcast_ss(&chargeA[j]));
                    factor = _mm256_mul_ps(factor, deltaT);
                    factor = _mm256_mul_ps(factor, _mm256_rcp_ps(distance2));
                    factor = _mm256_mul_ps(factor, _mm256_rsqrt_ps(distance2));

                    __m256 forceX = _mm256_mul_ps(deltaX, factor);
                    __m256 forceY = _mm256_mul_ps(deltaY, factor);
                    __m256 forceZ = _mm256_mul_ps(deltaZ, factor);

                    accelerationXV = _mm256_add_ps(accelerationXV, forceX);
                    accelerationYV = _mm256_add_ps(accelerationYV, forceY);
                    accelerationZV = _mm256_add_ps(accelerationZV, forceZ);
                }

                posXV = _mm256_add_ps(posXV, _mm256_mul_ps(velXV, deltaT));
                posYV = _mm256_add_ps(posYV, _mm256_mul_ps(velYV, deltaT));
                posZV = _mm256_add_ps(posZV, _mm256_mul_ps(velZV, deltaT));

                _mm256_storeu_ps(&posXB[i], posXV);
                _mm256_storeu_ps(&posYB[i], posYV);
                _mm256_storeu_ps(&posZB[i], posZV);

                velXV = _mm256_add_ps(velXV, accelerationXV);
                velYV = _mm256_add_ps(velYV, accelerationYV);
                velZV = _mm256_add_ps(velZV, accelerationZV);

                _mm256_storeu_ps(&velXB[i], velXV);
                _mm256_storeu_ps(&velYB[i], velYV);
                _mm256_storeu_ps(&velZB[i], velZV);

                _mm256_storeu_ps(&chargeB[i], chargeV);
            }

            std::swap(posXA, posXB);
            std::swap(posYA, posYB);
            std::swap(posZA, posZB);

            std::swap(velXA, velXB);
            std::swap(velYA, velYB);
            std::swap(velZA, velZB);

            std::swap(chargeA, chargeB);
        }

        double tEnd = time();

        if (posXA[0] == 0.12345) {
            std::cout << "this is a debug statement to prevent the compiler from optimizing away the update routine\n";
        }

        return gflops(numParticles, repeats, tStart, tEnd);
    }
};

#endif

class NBodyIron : public NBody
{
public:
    std::string species()
    {
        return "iron";
    }

    double performance(std::vector<int> dim)
    {
        using namespace LibFlatArray;

        int numParticles = dim[0];
        int repeats = dim[1];

        soa_array<Particle, 8192> particlesA;
        soa_array<Particle, 8192> particlesB;

        for (int i = 0; i < numParticles; ++i) {
            Particle p(
                i, i * i, sin(i),
                i % 11, i % 13, i % 19,
                10 + cos(2 * i));

            particlesA.push_back(p);
            particlesB.push_back(p);
        }

        double tStart = time();

        for (int t = 0; t < repeats; ++t) {
            soa_accessor<Particle, 8192, 1, 1, 0> accessorA = particlesA[0];
            soa_accessor<Particle, 8192, 1, 1, 0> accessorB = particlesB[0];

            for (; accessorA.index < numParticles; ++accessorA, ++accessorB ) {
                float posX = accessorA.posX();
                float posY = accessorA.posY();
                float posZ = accessorA.posZ();

                float velX = accessorA.velX();
                float velY = accessorA.velY();
                float velZ = accessorA.velZ();

                float charge = accessorA.charge();

                float accelerationX = 0;
                float accelerationY = 0;
                float accelerationZ = 0;

                soa_accessor<Particle, 8192, 1, 1, 0> accessorA2 = particlesA[0];

                for (accessorA2.index = 0; accessorA2.index < numParticles; ++accessorA2) {
                    float deltaX = posX - accessorA2.posX();
                    float deltaY = posY - accessorA2.posY();
                    float deltaZ = posZ - accessorA2.posZ();
                    float distance2 = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ + SOFTENING;
                    float factor = charge * accessorA2.charge() * DELTA_T / distance2 / sqrt(distance2);
                    float forceX = deltaX * factor;
                    float forceY = deltaY * factor;
                    float forceZ = deltaZ * factor;

                    accelerationX += forceX;
                    accelerationY += forceY;
                    accelerationZ += forceZ;
                }

                accessorB.posX() = posX + velX * DELTA_T;
                accessorB.posY() = posY + velY * DELTA_T;
                accessorB.posZ() = posZ + velZ * DELTA_T;

                accessorB.velX() = velX + accelerationX;
                accessorB.velY() = velY + accelerationY;
                accessorB.velZ() = velZ + accelerationZ;

                accessorB.charge() = charge;
            }

            std::swap(particlesA, particlesB);
        }

        double tEnd = time();

        if (particlesA[0].posX() == 0.12345) {
            std::cout << "this is a debug statement to prevent the compiler from optimizing away the update routine\n";
        }

        return gflops(numParticles, repeats, tStart, tEnd);
    }
};

#ifdef __AVX__

class NBodyBronze : public NBody
{
public:
    std::string species()
    {
        return "bronze";
    }

    double performance(std::vector<int> dim)
    {
        if (dim[0] <= 128) {
            return performance<128>(dim);
        }
        if (dim[0] <= 256) {
            return performance<256>(dim);
        }
        if (dim[0] <= 512) {
            return performance<512>(dim);
        }
        if (dim[0] <= 1024) {
            return performance<1024>(dim);
        }
        if (dim[0] <= 2048) {
            return performance<2048>(dim);
        }
        if (dim[0] <= 4096) {
            return performance<4096>(dim);
        }
        if (dim[0] <= 8192) {
            return performance<8192>(dim);
        }

        throw std::out_of_range("could not run test NBodyBronze as grid dimension X was too large");
    }

    template<int DIM>
    double performance(std::vector<int> dim)
    {
        using namespace LibFlatArray;

        int numParticles = dim[0];
        int repeats = dim[1];

        soa_array<Particle, DIM> particlesA;
        soa_array<Particle, DIM> particlesB;

        for (int i = 0; i < numParticles; ++i) {
            Particle p(
                i, i * i, sin(i),
                i % 11, i % 13, i % 19,
                10 + cos(2 * i));

            particlesA.push_back(p);
            particlesB.push_back(p);
        }

        double tStart = time();

        for (int t = 0; t < repeats; ++t) {
            soa_accessor<Particle, DIM, 1, 1, 0> accessorA = particlesA[0];
            soa_accessor<Particle, DIM, 1, 1, 0> accessorB = particlesB[0];
            soa_accessor<Particle, DIM, 1, 1, 0> accessorA2 = particlesA[0];

            for (; accessorA.index < (numParticles - 7); accessorA += 8, accessorB += 8) {
                __m256 posX = _mm256_loadu_ps(&accessorA.posX());
                __m256 posY = _mm256_loadu_ps(&accessorA.posY());
                __m256 posZ = _mm256_loadu_ps(&accessorA.posZ());

                __m256 velX = _mm256_loadu_ps(&accessorA.velX());
                __m256 velY = _mm256_loadu_ps(&accessorA.velY());
                __m256 velZ = _mm256_loadu_ps(&accessorA.velZ());

                __m256 charge = _mm256_loadu_ps(&accessorA.charge());

                __m256 accelerationX = _mm256_set1_ps(0.0);
                __m256 accelerationY = _mm256_set1_ps(0.0);
                __m256 accelerationZ = _mm256_set1_ps(0.0);

                __m256 deltaT = _mm256_set1_ps(DELTA_T);

                for (accessorA2.index = 0; accessorA2.index < numParticles; ++accessorA2) {
                    __m256 deltaX = _mm256_sub_ps(posX, _mm256_broadcast_ss(&accessorA2.posX()));
                    __m256 deltaY = _mm256_sub_ps(posY, _mm256_broadcast_ss(&accessorA2.posY()));
                    __m256 deltaZ = _mm256_sub_ps(posZ, _mm256_broadcast_ss(&accessorA2.posZ()));
                    __m256 distance2 =
                        _mm256_add_ps(_mm256_mul_ps(deltaX, deltaX),
                                      _mm256_mul_ps(deltaY, deltaY));
                    distance2 =
                        _mm256_add_ps(_mm256_mul_ps(deltaZ, deltaZ),
                                      distance2);
                    distance2 =
                        _mm256_add_ps(_mm256_set1_ps(SOFTENING),
                                      distance2);

                    __m256 factor = _mm256_mul_ps(charge, _mm256_broadcast_ss(&accessorA2.charge()));
                    factor = _mm256_mul_ps(factor, _mm256_set1_ps(DELTA_T));
                    factor = _mm256_mul_ps(factor, _mm256_rcp_ps(distance2));
                    factor = _mm256_mul_ps(factor, _mm256_rsqrt_ps(distance2));

                    __m256 forceX = _mm256_mul_ps(deltaX, factor);
                    __m256 forceY = _mm256_mul_ps(deltaY, factor);
                    __m256 forceZ = _mm256_mul_ps(deltaZ, factor);

                    accelerationX = _mm256_add_ps(accelerationX, forceX);
                    accelerationY = _mm256_add_ps(accelerationY, forceY);
                    accelerationZ = _mm256_add_ps(accelerationZ, forceZ);
                }

                posX = _mm256_add_ps(posX, _mm256_mul_ps(velX, deltaT));
                posY = _mm256_add_ps(posY, _mm256_mul_ps(velY, deltaT));
                posZ = _mm256_add_ps(posZ, _mm256_mul_ps(velZ, deltaT));

                _mm256_storeu_ps(&accessorB.posX(), posX);
                _mm256_storeu_ps(&accessorB.posY(), posY);
                _mm256_storeu_ps(&accessorB.posZ(), posZ);

                velX = _mm256_add_ps(velX, accelerationX);
                velY = _mm256_add_ps(velY, accelerationY);
                velZ = _mm256_add_ps(velZ, accelerationZ);

                _mm256_storeu_ps(&accessorB.velX(), velX);
                _mm256_storeu_ps(&accessorB.velY(), velY);
                _mm256_storeu_ps(&accessorB.velZ(), velZ);

                _mm256_storeu_ps(&accessorB.charge(), charge);
            }

            std::swap(particlesA, particlesB);
        }

        double tEnd = time();

        if (particlesA[0].posX() == 0.12345) {
            std::cout << "this is a debug statement to prevent the compiler from optimizing away the update routine\n";
        }

        return gflops(numParticles, repeats, tStart, tEnd);
    }
};

#endif

class NBodySilver : public NBody
{
public:
    std::string species()
    {
        return "silver";
    }

    double performance(std::vector<int> dim)
    {
        if (dim[0] <= 128) {
            return performance<128,  short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 256) {
            return performance<256,  short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 512) {
            return performance<512,  short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 1024) {
            return performance<1024, short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 2048) {
            return performance<2048, short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 4096) {
            return performance<4096, short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 8192) {
            return performance<8192, short_vec<float, 8> >(dim);
        }

        throw std::out_of_range("could not run test NBodySilver as grid dimension X was too large");
    }

    template<int DIM, typename REAL>
    double performance(std::vector<int> dim)
    {
        using namespace LibFlatArray;

        int numParticles = dim[0];
        int repeats = dim[1];

        soa_array<Particle, DIM> particlesA;
        soa_array<Particle, DIM> particlesB;

        for (int i = 0; i < numParticles; ++i) {
            Particle p(
                i, i * i, sin(i),
                i % 11, i % 13, i % 19,
                10 + cos(2 * i));

            particlesA.push_back(p);
            particlesB.push_back(p);
        }

        double tStart = time();

        for (int t = 0; t < repeats; ++t) {
            soa_accessor<Particle, DIM, 1, 1, 0> accessorA = particlesA[0];
            soa_accessor<Particle, DIM, 1, 1, 0> accessorB = particlesB[0];
            soa_accessor<Particle, DIM, 1, 1, 0> accessorA2 = particlesA[0];

            for (; accessorA.index < (numParticles - REAL::ARITY + 1); accessorA += REAL::ARITY, accessorB += REAL::ARITY) {
                REAL posX = &accessorA.posX();
                REAL posY = &accessorA.posY();
                REAL posZ = &accessorA.posZ();

                REAL velX = &accessorA.velX();
                REAL velY = &accessorA.velY();
                REAL velZ = &accessorA.velZ();

                REAL charge = &accessorA.charge();

                REAL accelerationX = 0.0;
                REAL accelerationY = 0.0;
                REAL accelerationZ = 0.0;

                for (accessorA2.index = 0; accessorA2.index < numParticles; ++accessorA2) {
                    REAL deltaX = posX - REAL(accessorA2.posX());
                    REAL deltaY = posY - REAL(accessorA2.posY());
                    REAL deltaZ = posZ - REAL(accessorA2.posZ());
                    REAL distance2 = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ + SOFTENING;

                    REAL factor = charge * accessorA2.charge() * DELTA_T / distance2 / sqrt(distance2);
                    REAL forceX = deltaX * factor;
                    REAL forceY = deltaY * factor;
                    REAL forceZ = deltaZ * factor;

                    accelerationX += forceX;
                    accelerationY += forceY;
                    accelerationZ += forceZ;
                }

                &accessorB.posX() << (posX + velX * DELTA_T);
                &accessorB.posY() << (posY + velY * DELTA_T);
                &accessorB.posZ() << (posZ + velZ * DELTA_T);

                &accessorB.velX() << (velX + accelerationX);
                &accessorB.velY() << (velY + accelerationY);
                &accessorB.velZ() << (velZ + accelerationZ);

                &accessorB.charge() << charge;
            }

            std::swap(particlesA, particlesB);
        }

        double tEnd = time();

        if (particlesA[0].posX() == 0.12345) {
            std::cout << "this is a debug statement to prevent the compiler from optimizing away the update routine\n";
        }

        return gflops(numParticles, repeats, tStart, tEnd);
    }
};

class NBodyGold : public NBody
{
public:
    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> dim)
    {
        if (dim[0] <= 128) {
            return performance<128,  short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 256) {
            return performance<256,  short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 512) {
            return performance<512,  short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 1024) {
            return performance<1024, short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 2048) {
            return performance<2048, short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 4096) {
            return performance<4096, short_vec<float, 8> >(dim);
        }
        if (dim[0] <= 8192) {
            return performance<8192, short_vec<float, 8> >(dim);
        }

        throw std::out_of_range("could not run test NBodySilver as grid dimension X was too large");
    }

    template<int DIM, typename REAL>
    double performance(std::vector<int> dim)
    {
        using namespace LibFlatArray;

        int numParticles = dim[0];
        int repeats = dim[1];

        soa_array<ArrayParticle, DIM> particlesA;
        soa_array<ArrayParticle, DIM> particlesB;

        for (int i = 0; i < numParticles; ++i) {
            ArrayParticle p(
                i, i * i, sin(i),
                i % 11, i % 13, i % 19,
                10 + cos(2 * i));

            particlesA.push_back(p);
            particlesB.push_back(p);
        }

        double tStart = time();

        for (int t = 0; t < repeats; ++t) {
            soa_accessor<ArrayParticle, DIM, 1, 1, 0> accessorA = particlesA[0];
            soa_accessor<ArrayParticle, DIM, 1, 1, 0> accessorB = particlesB[0];
            soa_accessor<ArrayParticle, DIM, 1, 1, 0> accessorA2 = particlesA[0];

            for (; accessorA.index < (numParticles - REAL::ARITY + 1); accessorA += REAL::ARITY, accessorB += REAL::ARITY) {
                REAL posX = &accessorA.pos()[0];
                REAL posY = &accessorA.pos()[1];
                REAL posZ = &accessorA.pos()[2];

                REAL velX = &accessorA.vel()[0];
                REAL velY = &accessorA.vel()[1];
                REAL velZ = &accessorA.vel()[2];

                REAL charge = &accessorA.charge();

                REAL accelerationX = 0.0;
                REAL accelerationY = 0.0;
                REAL accelerationZ = 0.0;

                for (accessorA2.index = 0; accessorA2.index < numParticles; ++accessorA2) {
                    REAL deltaX = posX - REAL(accessorA2.pos()[0]);
                    REAL deltaY = posY - REAL(accessorA2.pos()[1]);
                    REAL deltaZ = posZ - REAL(accessorA2.pos()[2]);
                    REAL distance2 = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ + SOFTENING;

                    REAL factor = charge * accessorA2.charge() * DELTA_T / distance2 / sqrt(distance2);
                    REAL forceX = deltaX * factor;
                    REAL forceY = deltaY * factor;
                    REAL forceZ = deltaZ * factor;

                    accelerationX += forceX;
                    accelerationY += forceY;
                    accelerationZ += forceZ;
                }

                &accessorB.pos()[0] << (posX + velX * DELTA_T);
                &accessorB.pos()[1] << (posY + velY * DELTA_T);
                &accessorB.pos()[2] << (posZ + velZ * DELTA_T);

                &accessorB.vel()[0] << (velX + accelerationX);
                &accessorB.vel()[1] << (velY + accelerationY);
                &accessorB.vel()[2] << (velZ + accelerationZ);

                &accessorB.charge() << charge;
            }

            std::swap(particlesA, particlesB);
        }

        double tEnd = time();

        if (particlesA[0].pos()[0] == 0.12345) {
            std::cout << "this is a debug statement to prevent the compiler from optimizing away the update routine\n";
        }

        return gflops(numParticles, repeats, tStart, tEnd);
    }
};

int main(int argc, char **argv)
{
    if ((argc < 3) || (argc == 4) || (argc > 5)) {
        std::cerr << "usage: " << argv[0] << " [-n,--name SUBSTRING] REVISION CUDA_DEVICE \n"
                  << "  - optional: only run tests whose name contains a SUBSTRING,\n"
                  << "  - REVISION is purely for output reasons,\n"
                  << "  - CUDA_DEVICE causes CUDA tests to run on the device with the given ID.\n";
        return 1;
    }
    std::string name = "";
    int argumentIndex = 1;
    if (argc == 5) {
        if ((std::string(argv[1]) == "-n") ||
            (std::string(argv[1]) == "--name")) {
            name = std::string(argv[2]);
        }
        argumentIndex = 3;
    }
    std::string revision = argv[argumentIndex + 0];

    std::stringstream s;
    s << argv[argumentIndex + 1];
    int cudaDevice;
    s >> cudaDevice;

    evaluate eval(name, revision);
    eval.print_header();

    std::vector<std::vector<int> > sizes;
    for (int d = 32; d <= 544; d += 4) {
        std::vector<int> dim(3, d);

        sizes.push_back(dim);
    }

    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(JacobiD3Q7Vanilla(), *i);
    }

#ifdef __SSE__
    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(JacobiD3Q7Pepper(), *i);
    }
#endif

    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(JacobiD3Q7Bronze(), *i);
    }

#ifdef __SSE__
    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(JacobiD3Q7Silver(), *i);
    }
#endif

    sizes.clear();

    for (int n = 128; n <= 8192; n *= 2) {
        std::vector<int> dim(3);
        dim[0] = n;
        dim[1] = std::size_t(4) * 512 * 1024 * 1024 / n / n;
        dim[2] = 0;

        sizes.push_back(dim);
    }

    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(NBodyVanilla(), *i);
    }

#ifdef __AVX__
    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(NBodyPepper(),  *i);
    }

    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(NBodyCurry(),  *i);
    }
#endif

    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(NBodyIron(),  *i);
    }

#ifdef __AVX__
    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(NBodyBronze(),  *i);
    }
#endif

    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(NBodySilver(),  *i);
    }

    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(NBodyGold(),  *i);
    }

    return 0;
}
