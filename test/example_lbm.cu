#include <iostream>
#include <cuda.h>
#include <libflatarray/flat_array.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <stdexcept>

long long time_usec()
{
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    return now.time_of_day().total_microseconds();
}

void check_cuda_error()
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(error) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

class CellLBM
{
public:
    double C;
    double N;
    double E;
    double W;
    double S;
    double T;
    double B;
    double NW;
    double NE;
    double SW;
    double SE;
    double TW;
    double BW;
    double TE;
    double BE;
    double TN;
    double BN;
    double TS;
    double BS;

#define GET_COMP(X, Y, Z, DIR)                                  \
    accessorOld[LibFlatArray::FixedCoord<X, Y, Z>()].DIR()

#define SET_COMP(DIR)                           \
    accessorNew.DIR()

    // fixme: refactor interface so that all wire-up in Cell can be summarized?
    template<typename ACCESSOR1, typename ACCESSOR2>
    __device__
    __host__
    static void updateLine(ACCESSOR1 accessorOld, int *indexOld, ACCESSOR2 accessorNew, int *indexNew, int startZ, int endZ)
    {
        int global_x = blockIdx.x * blockDim.x + threadIdx.x + 2;
        int global_y = blockIdx.y * blockDim.y + threadIdx.y + 2;
        int global_z = startZ;
        *indexOld =
            global_z * ACCESSOR1::DIM_X * ACCESSOR1::DIM_Y +
            global_y * ACCESSOR1::DIM_X +
            global_x;
        *indexNew =
            global_z * ACCESSOR2::DIM_X * ACCESSOR2::DIM_Y +
            global_y * ACCESSOR2::DIM_X +
            global_x;
        const int planeSizeOld = ACCESSOR1::DIM_X * ACCESSOR1::DIM_Y;
        const int planeSizeNew = ACCESSOR1::DIM_X * ACCESSOR2::DIM_Y;

#pragma unroll 10
        for (; global_z < endZ; global_z += 1) {
// #define SQR(X) ((X)*(X))
//             const double omega = 1.0/1.7;
//             const double omega_trm = 1.0 - omega;
//             const double omega_w0 = 3.0 * 1.0 / 3.0 * omega;
//             const double omega_w1 = 3.0*1.0/18.0*omega;
//             const double omega_w2 = 3.0*1.0/36.0*omega;
//             const double one_third = 1.0 / 3.0;
//             double velX, velY, velZ;

//             velX  =
//                 GET_COMP(-1,0,0,E) + GET_COMP(x-1,y-1,0,NE) +
//                 GET_COMP(-1,1,0,SE) + GET_COMP(x-1,y,z-1,TE) +
//                 GET_COMP(-1,0,1,BE);
//             velY  = GET_COMP(x,y-1,0,N) + GET_COMP(x+1,y-1,0,NW) +
//                 GET_COMP(x,y-1,z-1,TN) + GET_COMP(x,y-1,1,BN);
//             velZ  = GET_COMP(x,y,z-1,T) + GET_COMP(x,y+1,z-1,TS) +
//                 GET_COMP(x+1,y,z-1,TW);

//             const double rho =
//                 GET_COMP(x,y,0,C) + GET_COMP(x,y+1,0,S) +
//                 GET_COMP(x+1,y,0,W) + GET_COMP(x,y,1,B) +
//                 GET_COMP(x+1,y+1,0,SW) + GET_COMP(x,y+1,1,BS) +
//                 GET_COMP(x+1,y,1,BW) + velX + velY + velZ;
//             velX  = velX
//                 - GET_COMP(x+1,y,0,W)    - GET_COMP(x+1,y-1,0,NW)
//                 - GET_COMP(x+1,y+1,0,SW) - GET_COMP(x+1,y,z-1,TW)
//                 - GET_COMP(x+1,y,1,BW);
//             velY  = velY
//                 + GET_COMP(x-1,y-1,0,NE) - GET_COMP(x,y+1,0,S)
//                 - GET_COMP(x+1,y+1,0,SW) - GET_COMP(x-1,y+1,0,SE)
//                 - GET_COMP(x,y+1,z-1,TS) - GET_COMP(x,y+1,1,BS);
//             velZ  = velGET_COMP(x,y-1,z-1,TN) + GET_COMP(x-1,y,z-1,TE) - GET_COMP(x,y,1,B) - GET_COMP(x,y-1,1,BN) - GET_COMP(x,y+1,1,BS) - GET_COMP(x+1,y,1,BW) - GET_COMP(x-1,y,1,BE);

//             // density = rho;
//             // velocityX = velX;
//             // velocityY = velY;
//             // velocityZ = velZ;

//             const double dir_indep_trm = one_third*rho - 0.5*( velX*velX + velY*velY + velZ*velZ );

//             SET_COMP(C)=omega_trm * GET_COMP(x,y,0,C) + omega_w0*( dir_indep_trm );

//             SET_COMP(NW)=omega_trm * GET_COMP(x+1,y-1,0,NW) +
//                 omega_w2*( dir_indep_trm - ( velX-velY ) + 1.5*SQR( velX-velY ) );
//             SET_COMP(SE)=omega_trm * GET_COMP(x-1,y+1,0,SE) +
//                 omega_w2*( dir_indep_trm + ( velX-velY ) + 1.5*SQR( velX-velY ) );
//             SET_COMP(NE)=omega_trm * GET_COMP(x-1,y-1,0,NE) +
//                 omega_w2*( dir_indep_trm + ( velX+velY ) + 1.5*SQR( velX+velY ) );
//             SET_COMP(SW)=omega_trm * GET_COMP(x+1,y+1,0,SW) +
//                 omega_w2*( dir_indep_trm - ( velX+velY ) + 1.5*SQR( velX+velY ) );

//             SET_COMP(TW)=omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
//             SET_COMP(BE)=omega_trm * GET_COMP(x-1,y,1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
//             SET_COMP(TE)=omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + 1.5*SQR( velX+velZ ) );
//             SET_COMP(BW)=omega_trm * GET_COMP(x+1,y,1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + 1.5*SQR( velX+velZ ) );

//             SET_COMP(TS)=omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
//             SET_COMP(BN)=omega_trm * GET_COMP(x,y-1,1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
//             SET_COMP(TN)=omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + 1.5*SQR( velY+velZ ) );
//             SET_COMP(BS)=omega_trm * GET_COMP(x,y+1,1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + 1.5*SQR( velY+velZ ) );

//             SET_COMP(N)=omega_trm * GET_COMP(x,y-1,0,N) + omega_w1*( dir_indep_trm + velY + 1.5*SQR(velY));
//             SET_COMP(S)=omega_trm * GET_COMP(x,y+1,0,S) + omega_w1*( dir_indep_trm - velY + 1.5*SQR(velY));
//             SET_COMP(E)=omega_trm * GET_COMP(x-1,y,0,E) + omega_w1*( dir_indep_trm + velX + 1.5*SQR(velX));
//             SET_COMP(W)=omega_trm * GET_COMP(x+1,y,0,W) + omega_w1*( dir_indep_trm - velX + 1.5*SQR(velX));
//             SET_COMP(T)=omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + 1.5*SQR(velZ));
//             SET_COMP(B)=omega_trm * GET_COMP(x,y,1,B) + omega_w1*( dir_indep_trm - velZ + 1.5*SQR(velZ));


            SET_COMP(C) = GET_COMP(0, 0, 0, C);
            // if ((x == 2) && (y == 2)) {
            //     printf("going strong: %d, %d, %d\n", z, planeSizeOld, planeSizeNew);
            // }
            *indexOld += planeSizeOld;
            *indexNew += planeSizeNew;
        }
    }

#undef GET_COMP
#undef SET_COMP
#undef SQR

};

LIBFLATARRAY_REGISTER_SOA(CellLBM, ((double)(C))((double)(N))((double)(E))((double)(W))((double)(S))((double)(T))((double)(B))((double)(NW))((double)(SW))((double)(NE))((double)(SE))((double)(TW))((double)(BW))((double)(TE))((double)(BE))((double)(TN))((double)(BN))((double)(TS))((double)(BS)))

#define C 0
#define N 1
#define E 2
#define W 3
#define S 4
#define T 5
#define B 6

#define NW 7
#define SW 8
#define NE 9
#define SE 10

#define TW 11
#define BW 12
#define TE 13
#define BE 14

#define TN 15
#define BN 16
#define TS 17
#define BS 18

#define GET_COMP(X, Y, Z, DIR)                                          \
    gridOld[(Z) * dimX * dimY + (Y) * dimX + (X) + (DIR) * dimX * dimY * dimZ]

#define SET_COMP(DIR)                                                   \
    gridNew[z   * dimX * dimY +   y * dimX +   x + (DIR) * dimX * dimY * dimZ]

template<int UNUSED_X, int UNUSED_Y, int UNUSED_Z>
__global__ void update_lbm_classic(int dimX, int dimY, int dimZ, double *gridOld, double *gridNew)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 2;
    int z = 2;

#pragma unroll 10
    for (; z < (dimZ - 2); z += 1) {

#define SQR(X) ((X)*(X))
        const double omega = 1.0/1.7;
        const double omega_trm = 1.0 - omega;
        const double omega_w0 = 3.0 * 1.0 / 3.0 * omega;
        const double omega_w1 = 3.0*1.0/18.0*omega;
        const double omega_w2 = 3.0*1.0/36.0*omega;
        const double one_third = 1.0 / 3.0;
        double velX, velY, velZ;

        velX  =
            GET_COMP(x-1,y,z,E) + GET_COMP(x-1,y-1,z,NE) +
            GET_COMP(x-1,y+1,z,SE) + GET_COMP(x-1,y,z-1,TE) +
            GET_COMP(x-1,y,z+1,BE);
        velY  = GET_COMP(x,y-1,z,N) + GET_COMP(x+1,y-1,z,NW) +
            GET_COMP(x,y-1,z-1,TN) + GET_COMP(x,y-1,z+1,BN);
        velZ  = GET_COMP(x,y,z-1,T) + GET_COMP(x,y+1,z-1,TS) +
            GET_COMP(x+1,y,z-1,TW);

        const double rho =
            GET_COMP(x,y,z,C) + GET_COMP(x,y+1,z,S) +
            GET_COMP(x+1,y,z,W) + GET_COMP(x,y,z+1,B) +
            GET_COMP(x+1,y+1,z,SW) + GET_COMP(x,y+1,z+1,BS) +
            GET_COMP(x+1,y,z+1,BW) + velX + velY + velZ;
        velX  = velX
            - GET_COMP(x+1,y,z,W)    - GET_COMP(x+1,y-1,z,NW)
            - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x+1,y,z-1,TW)
            - GET_COMP(x+1,y,z+1,BW);
        velY  = velY
            + GET_COMP(x-1,y-1,z,NE) - GET_COMP(x,y+1,z,S)
            - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x-1,y+1,z,SE)
            - GET_COMP(x,y+1,z-1,TS) - GET_COMP(x,y+1,z+1,BS);
        velZ  = velZ+GET_COMP(x,y-1,z-1,TN) + GET_COMP(x-1,y,z-1,TE) - GET_COMP(x,y,z+1,B) - GET_COMP(x,y-1,z+1,BN) - GET_COMP(x,y+1,z+1,BS) - GET_COMP(x+1,y,z+1,BW) - GET_COMP(x-1,y,z+1,BE);

        // density = rho;
        // velocityX = velX;
        // velocityY = velY;
        // velocityZ = velZ;

        const double dir_indep_trm = one_third*rho - 0.5*( velX*velX + velY*velY + velZ*velZ );

        SET_COMP(C)=omega_trm * GET_COMP(x,y,z,C) + omega_w0*( dir_indep_trm );

        SET_COMP(NW)=omega_trm * GET_COMP(x+1,y-1,z,NW) +
            omega_w2*( dir_indep_trm - ( velX-velY ) + 1.5*SQR( velX-velY ) );
        SET_COMP(SE)=omega_trm * GET_COMP(x-1,y+1,z,SE) +
            omega_w2*( dir_indep_trm + ( velX-velY ) + 1.5*SQR( velX-velY ) );
        SET_COMP(NE)=omega_trm * GET_COMP(x-1,y-1,z,NE) +
            omega_w2*( dir_indep_trm + ( velX+velY ) + 1.5*SQR( velX+velY ) );
        SET_COMP(SW)=omega_trm * GET_COMP(x+1,y+1,z,SW) +
            omega_w2*( dir_indep_trm - ( velX+velY ) + 1.5*SQR( velX+velY ) );

        SET_COMP(TW)=omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        SET_COMP(BE)=omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        SET_COMP(TE)=omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + 1.5*SQR( velX+velZ ) );
        SET_COMP(BW)=omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + 1.5*SQR( velX+velZ ) );

        SET_COMP(TS)=omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        SET_COMP(BN)=omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        SET_COMP(TN)=omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + 1.5*SQR( velY+velZ ) );
        SET_COMP(BS)=omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + 1.5*SQR( velY+velZ ) );

        SET_COMP(N)=omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + 1.5*SQR(velY));
        SET_COMP(S)=omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + 1.5*SQR(velY));
        SET_COMP(E)=omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + 1.5*SQR(velX));
        SET_COMP(W)=omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + 1.5*SQR(velX));
        SET_COMP(T)=omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + 1.5*SQR(velZ));
        SET_COMP(B)=omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + 1.5*SQR(velZ));
    }
}

#undef GET_COMP
#undef SET_COMP
#undef SQR

#undef C
#undef N
#undef E
#undef W
#undef S
#undef T
#undef B

#undef NW
#undef SW
#undef NE
#undef SE

#undef TW
#undef BW
#undef TE
#undef BE

#undef TN
#undef BN
#undef TS
#undef BS

#define GET_COMP(X, Y, Z, DIR)                  \
    hoodOld[LibFlatArray::FixedCoord<X, Y, Z>()].DIR()

#define SET_COMP(DIR)                           \
    hoodNew.DIR()

template<int DIM_X, int DIM_Y, int DIM_Z>
__global__ void update_lbm_flat_array(int dimX, int dimY, int dimZ, double *gridOld, double *gridNew)
{
    int myX = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int myY = blockIdx.y * blockDim.y + threadIdx.y + 2;
    int myZ = 2;

    int index = myZ * DIM_X * DIM_Y + myY * DIM_X + myX;
    int offset = DIM_X * DIM_Y;
    int end = DIM_X * DIM_Y * (dimZ - 2);

    LibFlatArray::soa_accessor<CellLBM, DIM_X, DIM_Y, DIM_Z, 0> hoodNew((char*)gridNew, &index);
    LibFlatArray::soa_accessor<CellLBM, DIM_X, DIM_Y, DIM_Z, 0> hoodOld((char*)gridOld, &index);

#pragma unroll 10
    for (; index < end; index += offset) {
#define SQR(X) ((X)*(X))
        const double omega = 1.0/1.7;
        const double omega_trm = 1.0 - omega;
        const double omega_w0 = 3.0 * 1.0 / 3.0 * omega;
        const double omega_w1 = 3.0*1.0/18.0*omega;
        const double omega_w2 = 3.0*1.0/36.0*omega;
        const double one_third = 1.0 / 3.0;
        const int x = 0;
        const int y = 0;
        const int z = 0;
        double velX, velY, velZ;

        velX  =
            GET_COMP(x-1,y,z,E) + GET_COMP(x-1,y-1,z,NE) +
            GET_COMP(x-1,y+1,z,SE) + GET_COMP(x-1,y,z-1,TE) +
            GET_COMP(x-1,y,z+1,BE);
        velY  = GET_COMP(x,y-1,z,N) + GET_COMP(x+1,y-1,z,NW) +
            GET_COMP(x,y-1,z-1,TN) + GET_COMP(x,y-1,z+1,BN);
        velZ  = GET_COMP(x,y,z-1,T) + GET_COMP(x,y+1,z-1,TS) +
            GET_COMP(x+1,y,z-1,TW);

        const double rho =
            GET_COMP(x,y,z,C) + GET_COMP(x,y+1,z,S) +
            GET_COMP(x+1,y,z,W) + GET_COMP(x,y,z+1,B) +
            GET_COMP(x+1,y+1,z,SW) + GET_COMP(x,y+1,z+1,BS) +
            GET_COMP(x+1,y,z+1,BW) + velX + velY + velZ;
        velX  = velX
            - GET_COMP(x+1,y,z,W)    - GET_COMP(x+1,y-1,z,NW)
            - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x+1,y,z-1,TW)
            - GET_COMP(x+1,y,z+1,BW);
        velY  = velY
            + GET_COMP(x-1,y-1,z,NE) - GET_COMP(x,y+1,z,S)
            - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x-1,y+1,z,SE)
            - GET_COMP(x,y+1,z-1,TS) - GET_COMP(x,y+1,z+1,BS);
        velZ  = velZ+GET_COMP(x,y-1,z-1,TN) + GET_COMP(x-1,y,z-1,TE) - GET_COMP(x,y,z+1,B) - GET_COMP(x,y-1,z+1,BN) - GET_COMP(x,y+1,z+1,BS) - GET_COMP(x+1,y,z+1,BW) - GET_COMP(x-1,y,z+1,BE);

        // density = rho;
        // velocityX = velX;
        // velocityY = velY;
        // velocityZ = velZ;

        const double dir_indep_trm = one_third*rho - 0.5*( velX*velX + velY*velY + velZ*velZ );

        SET_COMP(C)=omega_trm * GET_COMP(x,y,z,C) + omega_w0*( dir_indep_trm );

        SET_COMP(NW)=omega_trm * GET_COMP(x+1,y-1,z,NW) +
            omega_w2*( dir_indep_trm - ( velX-velY ) + 1.5*SQR( velX-velY ) );
        SET_COMP(SE)=omega_trm * GET_COMP(x-1,y+1,z,SE) +
            omega_w2*( dir_indep_trm + ( velX-velY ) + 1.5*SQR( velX-velY ) );
        SET_COMP(NE)=omega_trm * GET_COMP(x-1,y-1,z,NE) +
            omega_w2*( dir_indep_trm + ( velX+velY ) + 1.5*SQR( velX+velY ) );
        SET_COMP(SW)=omega_trm * GET_COMP(x+1,y+1,z,SW) +
            omega_w2*( dir_indep_trm - ( velX+velY ) + 1.5*SQR( velX+velY ) );

        SET_COMP(TW)=omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        SET_COMP(BE)=omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        SET_COMP(TE)=omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + 1.5*SQR( velX+velZ ) );
        SET_COMP(BW)=omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + 1.5*SQR( velX+velZ ) );

        SET_COMP(TS)=omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        SET_COMP(BN)=omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        SET_COMP(TN)=omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + 1.5*SQR( velY+velZ ) );
        SET_COMP(BS)=omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + 1.5*SQR( velY+velZ ) );

        SET_COMP(N)=omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + 1.5*SQR(velY));
        SET_COMP(S)=omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + 1.5*SQR(velY));
        SET_COMP(E)=omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + 1.5*SQR(velX));
        SET_COMP(W)=omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + 1.5*SQR(velX));
        SET_COMP(T)=omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + 1.5*SQR(velZ));
        SET_COMP(B)=omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + 1.5*SQR(velZ));
    }
}

#undef GET_COMP
#undef SET_COMP

#define GET_COMP(X, Y, Z, DIR)                  \
    gridOld[(Z) * dimX * dimY + (Y) * dimX + (X)].DIR

#define SET_COMP(DIR)                           \
    gridNew[(z) * dimX * dimY + (y) * dimX + (x)].DIR

__global__ void update_lbm_object_oriented(int dimX, int dimY, int dimZ, CellLBM *gridOld, CellLBM *gridNew)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 2;
    int z = 2;

#pragma unroll 10
    for (; z < (dimZ - 2); z += 1) {

#define SQR(X) ((X)*(X))
        const double omega = 1.0/1.7;
        const double omega_trm = 1.0 - omega;
        const double omega_w0 = 3.0 * 1.0 / 3.0 * omega;
        const double omega_w1 = 3.0*1.0/18.0*omega;
        const double omega_w2 = 3.0*1.0/36.0*omega;
        const double one_third = 1.0 / 3.0;
        double velX, velY, velZ;

        velX  =
            GET_COMP(x-1,y,z,E) + GET_COMP(x-1,y-1,z,NE) +
            GET_COMP(x-1,y+1,z,SE) + GET_COMP(x-1,y,z-1,TE) +
            GET_COMP(x-1,y,z+1,BE);
        velY  = GET_COMP(x,y-1,z,N) + GET_COMP(x+1,y-1,z,NW) +
            GET_COMP(x,y-1,z-1,TN) + GET_COMP(x,y-1,z+1,BN);
        velZ  = GET_COMP(x,y,z-1,T) + GET_COMP(x,y+1,z-1,TS) +
            GET_COMP(x+1,y,z-1,TW);

        const double rho =
            GET_COMP(x,y,z,C) + GET_COMP(x,y+1,z,S) +
            GET_COMP(x+1,y,z,W) + GET_COMP(x,y,z+1,B) +
            GET_COMP(x+1,y+1,z,SW) + GET_COMP(x,y+1,z+1,BS) +
            GET_COMP(x+1,y,z+1,BW) + velX + velY + velZ;
        velX  = velX
            - GET_COMP(x+1,y,z,W)    - GET_COMP(x+1,y-1,z,NW)
            - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x+1,y,z-1,TW)
            - GET_COMP(x+1,y,z+1,BW);
        velY  = velY
            + GET_COMP(x-1,y-1,z,NE) - GET_COMP(x,y+1,z,S)
            - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x-1,y+1,z,SE)
            - GET_COMP(x,y+1,z-1,TS) - GET_COMP(x,y+1,z+1,BS);
        velZ  = velZ+GET_COMP(x,y-1,z-1,TN) + GET_COMP(x-1,y,z-1,TE) - GET_COMP(x,y,z+1,B) - GET_COMP(x,y-1,z+1,BN) - GET_COMP(x,y+1,z+1,BS) - GET_COMP(x+1,y,z+1,BW) - GET_COMP(x-1,y,z+1,BE);

        // density = rho;
        // velocityX = velX;
        // velocityY = velY;
        // velocityZ = velZ;

        const double dir_indep_trm = one_third*rho - 0.5*( velX*velX + velY*velY + velZ*velZ );

        SET_COMP(C)=omega_trm * GET_COMP(x,y,z,C) + omega_w0*( dir_indep_trm );

        SET_COMP(NW)=omega_trm * GET_COMP(x+1,y-1,z,NW) +
            omega_w2*( dir_indep_trm - ( velX-velY ) + 1.5*SQR( velX-velY ) );
        SET_COMP(SE)=omega_trm * GET_COMP(x-1,y+1,z,SE) +
            omega_w2*( dir_indep_trm + ( velX-velY ) + 1.5*SQR( velX-velY ) );
        SET_COMP(NE)=omega_trm * GET_COMP(x-1,y-1,z,NE) +
            omega_w2*( dir_indep_trm + ( velX+velY ) + 1.5*SQR( velX+velY ) );
        SET_COMP(SW)=omega_trm * GET_COMP(x+1,y+1,z,SW) +
            omega_w2*( dir_indep_trm - ( velX+velY ) + 1.5*SQR( velX+velY ) );

        SET_COMP(TW)=omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        SET_COMP(BE)=omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        SET_COMP(TE)=omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + 1.5*SQR( velX+velZ ) );
        SET_COMP(BW)=omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + 1.5*SQR( velX+velZ ) );

        SET_COMP(TS)=omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        SET_COMP(BN)=omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        SET_COMP(TN)=omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + 1.5*SQR( velY+velZ ) );
        SET_COMP(BS)=omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + 1.5*SQR( velY+velZ ) );

        SET_COMP(N)=omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + 1.5*SQR(velY));
        SET_COMP(S)=omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + 1.5*SQR(velY));
        SET_COMP(E)=omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + 1.5*SQR(velX));
        SET_COMP(W)=omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + 1.5*SQR(velX));
        SET_COMP(T)=omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + 1.5*SQR(velZ));
        SET_COMP(B)=omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + 1.5*SQR(velZ));
    }
}

#undef GET_COMP
#undef SET_COMP

class benchmark
{
public:
    void evaluate()
    {
        for (int dim = 32; dim <= 160; dim += 4) {
            run(dim);
        }
    }

    void run(int dim)
    {
        int repeats = 100;
	if (dim <= 96) {
            repeats *= 10;
        }
        repeats = 1;

        long long useconds = exec(dim, repeats);

        double updates = 1.0 * gridSize(dim) * repeats;
        double seconds = useconds * 10e-6;
        double glups = 10e-9 * updates / seconds;

        std::cout << name() << " " << dim << " " << glups << " GLUPS\n";
    }

protected:
    virtual long long exec(int dim, int repeats) = 0;
    virtual std::string name() = 0;
    virtual size_t gridSize(int dim) = 0;
};

class benchmark_lbm_cuda : public benchmark
{
protected:
    long long exec(int dim, int repeats)
    {
        dim3 dimBlock;
        dim3 dimGrid;
        gen_dims(&dimBlock, &dimGrid, dim);

        return exec(dim, dimBlock, dimGrid, repeats);
    }

    virtual size_t gridSize(int dim)
    {
        dim3 dimBlock;
        dim3 dimGrid;
        gen_dims(&dimBlock, &dimGrid, dim);

        return dimGrid.x * dimBlock.x * dimGrid.y * dimBlock.y * (256 - 4);
    }

    virtual long long exec(int dim, dim3 dimBlock, dim3 dimGrid, int repeats) = 0;

    void gen_dims(dim3 *dimBlock, dim3 *dimGrid, int dim)
    {
        int blockWidth = 1;
        for (; blockWidth <= dim; blockWidth *= 2) {
        }
        blockWidth /= 2;
        blockWidth = std::min(256, blockWidth);
        *dimBlock = dim3(blockWidth, 2, 1);
        *dimGrid = dim3(dim / dimBlock->x, dim / dimBlock->y, 1);
    }
};


template<int DIM>
class benchmark_lbm_cuda_classic_callback
{
public:
    void operator()(int dim, long long *time, dim3 dimBlock, dim3 dimGrid, int repeats)
    {
        int size = (DIM + 2) * (DIM + 2) * (256 + 64) * 20;
        int bytesize = size * sizeof(double);
        std::vector<double> grid(size, 4711);

        double *devGridOld;
        double *devGridNew;
        cudaMalloc(&devGridOld, bytesize);
        cudaMalloc(&devGridNew, bytesize);
        check_cuda_error();

        cudaMemcpy(devGridOld, &grid[0], bytesize, cudaMemcpyHostToDevice);
        cudaMemcpy(devGridNew, &grid[0], bytesize, cudaMemcpyHostToDevice);
        check_cuda_error();

        cudaDeviceSynchronize();
        long long t_start = time_usec();

        for (int t = 0; t < repeats; ++t) {
            update_lbm_flat_array<DIM, DIM, 256><<<dimGrid, dimBlock>>>(dim, dim, 256, devGridOld, devGridNew);
            // update_lbm_classic<DIM, DIM, 256><<<dimGrid, dimBlock>>>(dim, dim, 256, devGridOld, devGridNew);
            std::swap(devGridOld, devGridNew);
        }

        cudaDeviceSynchronize();
        long long t_end = time_usec();
        check_cuda_error();

        cudaMemcpy(&grid[0], devGridNew, bytesize, cudaMemcpyDeviceToHost);
        cudaFree(devGridOld);
        cudaFree(devGridNew);
        check_cuda_error();
        *time = t_end - t_start;
    }
};

class benchmark_lbm_cuda_classic : public benchmark_lbm_cuda
{
protected:
    virtual long long exec(int dim, dim3 dimBlock, dim3 dimGrid, int repeats)
    {
        long long time;
        LibFlatArray::detail::flat_array::bind<benchmark_lbm_cuda_classic_callback>()(dim, &time, dimBlock, dimGrid, repeats);
        return time;
    }

    virtual std::string name()
    {
        return "lbm_cuda_classic";
    }
};

template<typename CELL, typename ACCESSOR1, typename ACCESSOR2>
__global__
void update(ACCESSOR1 accessor1, ACCESSOR2 accessor2)
{
    int indexOld;
    int indexNew;
    ACCESSOR1 accessorOld(accessor1.getData(), &indexOld);
    ACCESSOR2 accessorNew(accessor2.getData(), &indexNew);

    CELL::updateLine(accessorOld, &indexOld, accessorNew, &indexNew, 2, 256 - 2);
}

template<typename CELL, typename ACCESSOR1>
class SoAUpdateFunctorHelper2
{
public:
    SoAUpdateFunctorHelper2(ACCESSOR1 accessor1, int *index1, const dim3& dimBlock, const dim3& dimGrid) :
        accessor1(accessor1),
        index1(index1),
        dimBlock(dimBlock),
        dimGrid(dimGrid)
    {}

    template<typename ACCESSOR2>
    void operator()(ACCESSOR2 accessor2) const
    {
        // fixme: update this shit!
        // update<CELL><<<dimBlock, dimGrid>>>(accessor1, accessor2);
        update_lbm_flat_array<ACCESSOR1::DIM_X, ACCESSOR1::DIM_Y, ACCESSOR1::DIM_Z><<<dimBlock, dimGrid>>>(
            1, 1, 256, (double*)accessor1.get_data(), (double*)accessor2.get_data());
    }

    int index2;

private:
    ACCESSOR1 accessor1;
    int *index1;
    const dim3& dimBlock;
    const dim3& dimGrid;
};

template<typename CELL, typename GRID2>
class SoAUpdateFunctorHelper1
{
public:

    SoAUpdateFunctorHelper1(GRID2 *grid2, const dim3& dimBlock, const dim3& dimGrid) :
        grid2(grid2),
        dimBlock(dimBlock),
        dimGrid(dimGrid)
    {}

    template<typename ACCESSOR1>
    void operator()(ACCESSOR1 accessor1)
    {
        SoAUpdateFunctorHelper2<CELL, ACCESSOR1> helper(accessor1, &index1, dimBlock, dimGrid);
        grid2->callback(helper, &helper.index2);
    }

    int index1;

private:
    GRID2 *grid2;
    const dim3& dimBlock;
    const dim3& dimGrid;
};


template<typename CELL>
class SoAUpdateFunctorPrototype
{
public:
    template<typename GRID1, typename GRID2>
    void operator()(GRID1 *gridOld, GRID2 *gridNew, const dim3& dimBlock, const dim3& dimGrid)
    {
        SoAUpdateFunctorHelper1<CELL, GRID2> helper(gridNew, dimBlock, dimGrid);
        gridOld->callback(helper, &helper.index1);
    }
};

class benchmark_lbm_cuda_flat_array : public benchmark_lbm_cuda
{
    virtual long long exec(int dim, dim3 dimBlock, dim3 dimGrid, int repeats)
    {
        LibFlatArray::soa_grid<CellLBM> gridA(dim, dim, dim);
        LibFlatArray::soa_grid<CellLBM> gridB(dim, dim, dim);
        // fixme: init grid?

        char *dataA = gridA.get_data();
        char *dataB = gridB.get_data();

        char *buf;
        cudaMalloc(reinterpret_cast<void**>(&buf), gridA.byte_size());
        gridA.set_data(buf);
        cudaMalloc(reinterpret_cast<void**>(&buf), gridB.byte_size());
        gridB.set_data(buf);


        cudaDeviceSynchronize();
        long long t_start = time_usec();

        LibFlatArray::soa_grid<CellLBM> *gridOld = &gridA;
        LibFlatArray::soa_grid<CellLBM> *gridNew = &gridB;

        // fixme: do the evolution. 1 functor per call?
        for (int t = 0; t < repeats; ++t) {
            SoAUpdateFunctorPrototype<CellLBM>()(gridOld, gridNew, dimBlock, dimGrid);
            std::swap(gridOld, gridNew);
        }

        cudaDeviceSynchronize();
        long long t_end = time_usec();
        check_cuda_error();

        cudaFree(gridA.get_data());
        cudaFree(gridB.get_data());

        gridA.set_data(dataA);
        gridB.set_data(dataB);

        return t_end - t_start;
    }

    virtual std::string name()
    {
        return "lbm_cuda_flat_array";
    }
};

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " CUDA_DEVICE\n";
        return 1;
    }

    std::stringstream s;
    s << argv[1];
    int cudaDevice;
    s >> cudaDevice;
    cudaSetDevice(cudaDevice);

    // benchmark_lbm_cuda_classic().evaluate();
    benchmark_lbm_cuda_flat_array().evaluate();

    return 0;
}
