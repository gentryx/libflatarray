#include <iostream>
#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>

using namespace LibFlatArray;

class JacobiD3Q6 : public cpu_benchmark
{
public:
    std::string family()
    {
        return "JacobiD3Q6";
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

class JacobiD3Q6Iron : public JacobiD3Q6
{
public:
    std::string species()
    {
        return "iron";
    }

    double performance(std::vector<int> dim)
    {
        int dimX = dim[0];
        int dimY = dim[1];
        int dimZ = dim[2];
        int maxT = 20;

        int offsetZ = dimX * dimY;
        int gridVolume = dimX * dimY * dimZ;
        std::vector<double> compressedGrid(2 * gridVolume);
        double *gridOld = &compressedGrid[0];
        double *gridNew = &compressedGrid[gridVolume];

        for (int z = 0; z < dimZ; ++z) {
            for (int y = 0; y < dimY; ++y) {
                for (int x = 0; x < dimX; ++x) {
                    gridOld[z * offsetZ + y * dimY + x] = x + y + z;
                    gridNew[z * offsetZ + y * dimY + x] = x + y + z;
                }
            }
        }

        double tStart = time();

        for (int t = 0; t < maxT; ++t) {
            for (int z = 1; z < (dimZ - 1); ++z) {
                for (int y = 1; y < (dimY - 1); ++y) {
                    updateLine(gridOld, gridNew, 1, y, z, dimX - 1, dimX, offsetZ);
                }
            }
        }

        double tEnd = time();

        if (gridOld[1 * offsetZ + 1 * dimY + 1] ==
            gridNew[1 * offsetZ + 1 * dimY + 1]) {
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
            gridNew[x + y * offsetY + z * offsetZ] = 1 +
                (gridOld[x + y * offsetY + z * offsetZ - 1 * offsetZ] +
                 gridOld[x + y * offsetY + z * offsetZ - 1 * offsetY] +
                 gridOld[x + y * offsetY + z * offsetZ - 1          ] +
                 gridOld[x + y * offsetY + z * offsetZ + 0          ] +
                 gridOld[x + y * offsetY + z * offsetZ + 1          ] +
                 gridOld[x + y * offsetY + z * offsetZ + 1 * offsetY] +
                 gridOld[x + y * offsetY + z * offsetZ + 1 * offsetZ]) * (1.0 / 7.0);
        }
    }
};

// class JacobiD3Q6Cell
// {
// public:
//     double temp;
// };

// LIBFLATARRAY_REGISTER_SOA(JacobiD3Q6Cell, (((double)(temp))))

// class JacobiD3Q6Vanilla : public JacobiD3Q6
// {
// public:
//     std::string species()
//     {
//         return "vanilla";
//     }

//     double performance(std::vector<int> dim)
//     {
//         int dimX = dim[0];
//         int dimY = dim[1];
//         int dimZ = dim[2];
//         int maxT = 20;

//         soa_grid<
//     }
// };

int main(int argc, char **argv)
{
    if ((argc < 3) || (argc > 4)) {
        std::cerr << "usage: " << argv[0] << "[-q,--quick] REVISION CUDA_DEVICE\n";
        return 1;
    }

    bool quick = false;
    int argumentIndex = 1;
    if (argc == 4) {
        if ((std::string(argv[1]) == "-q") ||
            (std::string(argv[1]) == "--quick")) {
            quick = true;
        }
        argumentIndex = 2;
    }
    std::string revision = argv[argumentIndex + 0];

    std::stringstream s;
    s << argv[argumentIndex + 1];
    int cudaDevice;
    s >> cudaDevice;

    evaluate eval(revision);
    eval.print_header();

    for (int d = 32; d <= 544; d += 4) {
        std::vector<int> dim(3, d);
         eval(JacobiD3Q6Iron(), dim);
    }

    return 0;
}
