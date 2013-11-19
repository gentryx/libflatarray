/**
 * Copyright 2013 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <libflatarray/flat_array.hpp>
#include <vector>

class Particle
{
public:
    inline
    Particle(
        float posX = 0.0,
        float posY = 0.0,
        float posZ = 0.0,
        float velX = 0.0,
        float velY = 0.0,
        float velZ = 0.0,
        float charge = 0.0,
        float mass = 1.0) :
        posX(posX),
        posY(posY),
        posZ(posZ),
        velX(velX),
        velY(velY),
        velZ(velZ),
        charge(charge),
        mass(mass)
    {}

    float posX;
    float posY;
    float posZ;
    float velX;
    float velY;
    float velZ;
    float charge;
    float mass;
};

LIBFLATARRAY_REGISTER_SOA(Particle, ((float)(posX))((float)(posY))((float)(posZ))((float)(velX))((float)(velY))((float)(velZ))((float)(charge))((float)(mass)))

using namespace LibFlatArray;

template<int SIZE>
class Benchmark
{
public:
    void evaluate(int maxSteps)
    {
        soa_array<Particle, SIZE> particlesOld;
        soa_array<Particle, SIZE> particlesNew;

        for (int i = 0; i < SIZE; ++i) {
            particlesOld[i].posX() = rand() % 1000;
            particlesOld[i].posY() = rand() % 1000;
            particlesOld[i].posZ() = rand() % 1000;

            particlesOld[i].velX() = (rand() % 2000 - 1000) * 1e-4;
            particlesOld[i].velY() = (rand() % 2000 - 1000) * 1e-4;
            particlesOld[i].velZ() = (rand() % 2000 - 1000) * 1e-4;

            particlesOld[i].charge() = 1.0 + (rand() % 100) * 1e-2;
            particlesOld[i].mass()   = 1.0 + (rand() % 900) * 1e-2;
        }

        printStats(particlesOld, 0);

        for (int t = 0; t < maxSteps; ++t) {
            update(particlesOld, &particlesNew);
            std::swap(particlesOld, particlesNew);
        }

        printStats(particlesOld, maxSteps);
    }

    void printStats(const soa_array<Particle, SIZE>& particles, int time) const
    {
        float totalEnergy = 0;

        for (int i = 0; i < SIZE; ++i) {
            float vel = sqrt(
                particles[i].velX() * particles[i].velX() +
                particles[i].velY() * particles[i].velY() +
                particles[i].velZ() * particles[i].velZ());
            totalEnergy += vel;
            totalEnergy += vel * particles[i].mass();
        }

        std::cout << "time: " << time << " total energy: " << totalEnergy << "\n";
    }

    void update(const soa_array<Particle, SIZE>& particlesOld,
                soa_array<Particle, SIZE> *particlesNew)
    {
        for (int i = 0; i < SIZE; ++i) {
            float velX = particlesOld[i].velX();
            float velY = particlesOld[i].velY();
            float velZ = particlesOld[i].velZ();

            for (int j = 0; j < SIZE; ++j) {
                float deltaX = particlesOld[i].posX() - particlesOld[j].posX();
                float deltaY = particlesOld[i].posY() - particlesOld[j].posY();
                float deltaZ = particlesOld[i].posZ() - particlesOld[j].posZ();

                float deltaLength = sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
                float acceleration = particlesOld[i].charge() * particlesOld[j].charge() /
                    deltaLength / particlesOld[i].mass();

                velX +=  deltaX * acceleration;
                velY +=  deltaY * acceleration;
                velZ +=  deltaZ * acceleration;
            }

            (*particlesNew)[i].posX() = particlesOld[i].posX() + velX;
            (*particlesNew)[i].posY() = particlesOld[i].posX() + velY;
            (*particlesNew)[i].posZ() = particlesOld[i].posX() + velZ;

            (*particlesNew)[i].velX() = velX;
            (*particlesNew)[i].velY() = velY;
            (*particlesNew)[i].velZ() = velZ;

            (*particlesNew)[i].charge() = particlesOld[i].charge();
            (*particlesNew)[i].mass()   = particlesOld[i].mass();
        }
    }
};

int main(int argc, char **argv)
{
    Benchmark<1024> bench;
    bench.evaluate(10);

    return 0;
}
