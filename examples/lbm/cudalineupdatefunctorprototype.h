#ifndef LIBFLATARRAY_EXAMPLES_LBM_CUDALINEUPDATEFUNCTORPROTOTYPE_H
#define LIBFLATARRAY_EXAMPLES_LBM_CUDALINEUPDATEFUNCTORPROTOTYPE_H

#include "cell.h"

template<typename CELL, typename ACCESSOR1, typename ACCESSOR2>
__global__
void update(ACCESSOR1 accessor1, ACCESSOR2 accessor2)
{
    ACCESSOR1 accessorOld(accessor1.get_data());
    ACCESSOR2 accessorNew(accessor2.get_data());

    CELL::updateLine(
        accessorOld, &accessorOld.index,
        accessorNew, &accessorNew.index, 2, 256 - 2);
}

template<typename CELL>
class CudaLineUpdateFunctorPrototypeImplementation
{
public:
    CudaLineUpdateFunctorPrototypeImplementation(dim3 dim_block, dim3 dim_grid) :
        dim_block(dim_block),
        dim_grid(dim_grid)
    {}

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(ACCESSOR1 accessor1, ACCESSOR2 accessor2) const
    {
        update<CELL, ACCESSOR1, ACCESSOR2><<<dim_grid, dim_block>>>(accessor1, accessor2);
    }

private:
    dim3 dim_block;
    dim3 dim_grid;
};

template<typename CELL>
class CudaLineUpdateFunctorPrototype
{
public:
    CudaLineUpdateFunctorPrototype(dim3 dim_block, dim3 dim_grid) :
        dim_block(dim_block),
        dim_grid(dim_grid)
    {}

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(ACCESSOR1 accessor1, ACCESSOR2 accessor2) const;

private:
    dim3 dim_block;
    dim3 dim_grid;
};

#define IMPLEMENTATION(CELL, X1, Y1, Z1, X2, Y2, Z2)                    \
    template<>                                                          \
    template<>                                                          \
    void CudaLineUpdateFunctorPrototype<CellLBM>::operator()(           \
        LibFlatArray::soa_accessor<CELL, X1, Y1, Z1, 0> accessor1,      \
        LibFlatArray::soa_accessor<CELL, X2, Y2, Z2, 0> accessor2) const \
    {                                                                   \
        CudaLineUpdateFunctorPrototypeImplementation<CELL> i(dim_block, dim_grid); \
        i(accessor1, accessor2);                                        \
    }


#endif
