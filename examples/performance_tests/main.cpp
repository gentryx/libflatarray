#include <iostream>
#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>
#include <immintrin.h>

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
        int dimX = dim[0];
        int dimY = dim[1];
        int dimZ = dim[2];
        int maxT = 200000000 / dimX / dimY / dimZ;
        maxT = std::max(16, maxT);

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

class JacobiD3Q7Pepper : public JacobiD3Q7
{
public:
    std::string species()
    {
        return "pepper";
    }

    double performance(std::vector<int> dim)
    {
        int dimX = dim[0];
        int dimY = dim[1];
        int dimZ = dim[2];
        int maxT = 200000000 / dimX / dimY / dimZ;
        maxT = std::max(16, maxT);

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
        __m128d factorS = _mm_set1_pd(WEIGHT_S);
        __m128d factorT = _mm_set1_pd(WEIGHT_T);
        __m128d factorW = _mm_set1_pd(WEIGHT_W);
        __m128d factorC = _mm_set1_pd(WEIGHT_C);
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

class Particle
{
public:
    inline Particle(
        const float posX,
        const float posY,
        const float posZ,
        const float velX,
        const float velY,
        const float velZ,
        const float charge) :
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
        double flops = repeats * numParticles * (9 + numParticles * (3 + 6 + 4 + 3 + 3));
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
                    float factor = charge * particlesA[j].charge * DELTA_T / sqrt(distance2);
                    float forceX = deltaX / factor;
                    float forceY = deltaY / factor;
                    float forceZ = deltaZ / factor;

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
                    factor = _mm256_mul_ps(factor, _mm256_rsqrt_ps(distance2));
                    factor = _mm256_rcp_ps(factor);

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
                    float factor = charge * chargeA[j] * DELTA_T / sqrt(distance2);
                    float forceX = deltaX / factor;
                    float forceY = deltaY / factor;
                    float forceZ = deltaZ / factor;

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

    std::vector<std::vector<int> > sizes;
    for (int d = 32; d <= 544; d += 4) {
        std::vector<int> dim(3);
        dim[0] = d;
        dim[1] = d;
        dim[2] = 3;

        sizes.push_back(dim);
    }

    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(JacobiD3Q7Vanilla(), *i);
    }

    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(JacobiD3Q7Pepper(), *i);
    }

    sizes.clear();

    for (int n = 128; n <= 8192; n *= 2) {
        std::vector<int> dim(3);
        dim[0] = n;
        dim[1] = 1024 * 1024 * 1024 / n / n;
        dim[2] = 0;

        sizes.push_back(dim);
    }

    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(NBodyVanilla(), *i);
    }

    for (std::vector<std::vector<int> >::iterator i = sizes.begin(); i != sizes.end(); ++i) {
        eval(NBodyPepper(),  *i);
    }

    return 0;
}
