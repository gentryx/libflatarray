#ifndef LIBFLATARRAY_EXAMPLES_LBM_CELL_H
#define LIBFLATARRAY_EXAMPLES_LBM_CELL_H

#include <libflatarray/flat_array.hpp>

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

#define GET_COMP(X, Y, Z, DIR)                          \
    accessorOld[LibFlatArray::coord<X, Y, Z>()].DIR()

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
        for (; global_z < endZ; ++global_z) {
#define SQR(X) ((X)*(X))
            const int x = 0;
            const int y = 0;
            const int z = 0;

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

            *indexOld += planeSizeOld;
            *indexNew += planeSizeNew;
        }
    }

#undef GET_COMP
#undef SET_COMP
#undef SQR
};

LIBFLATARRAY_REGISTER_SOA(CellLBM, ((double)(C))((double)(N))((double)(E))((double)(W))((double)(S))((double)(T))((double)(B))((double)(NW))((double)(SW))((double)(NE))((double)(SE))((double)(TW))((double)(BW))((double)(TE))((double)(BE))((double)(TN))((double)(BN))((double)(TS))((double)(BS)))

#endif
