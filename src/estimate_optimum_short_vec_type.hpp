/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_ESTIMATE_OPTIMUM_SHORT_VEC_TYPE_HPP
#define FLAT_ARRAY_ESTIMATE_OPTIMUM_SHORT_VEC_TYPE_HPP

#include <libflatarray/detail/streaming_short_vec_switch.hpp>

namespace LibFlatArray {

/**
 * This class serves as a type switch to select an appropriate
 * short_vec type based on the machine architecture and working set
 * size. This is just a heuristic. Users are advised that an
 * analytical performance model can yield much better results.
 *
 * We're primarily concerned with two choices: temporal vs.
 * non-temporal stores and the arity of the vector type. Smaller
 * working sets should use short_vec if they fit well into the cache,
 * larger sets should use streaming_short_vec to benefit from
 * streaming stores.
 *
 * The arity of the vector type should not be smaller than the arity
 * of the supported assembly instructions (e.g. >=8 for AVX512 and
 * doubles).If the arity is larger then we effectively perform
 * loop-unrolling. This may be beneficial for architectures that
 * struggle with out-of-order execution as if lenghtens the loop body
 * and gives them more independent instructions to work on (e.g. Intel
 * Core 2). Modern Intel architectures however may suffer from
 * unrolling as this might make the loop body exceed the size of the
 * loop buffer which holds previously decoded microinstructions.
 *
 * Arguments should be:
 *
 * - CARGO: the main machine data type used inside the kernel, e.g.
 *     float or double. Most kernels will operate on various data
 *     types, but the vector arity should usually be chosen based on
 *     that type which is used most as it has the strongest impact on
 *     register scheduling.
 *
 * - ACCESSOR: an soa_accessor produced by LibFlatArray that provides
 *     the number of elements in the working set. We assume the size
 *     of the working set to be the product of the size of CARGO and
 *     the number of elements in the set.
 *
 * - LAST_LEVEL_CACHE_SIZE_ESTIMATE: if available, the user can give
 *     an estimate of the CPU's cache. Our hard-coded value will
 *     overestimate that size for most architectures, but that's
 *     generally fine. The consequence of overestimating is that for
 *     some medium-sized sets the code will use temporal stores
 *     instead of non-temporal stores, reulting in a performance hit
 *     of less than 30% (true for most codes and current
 *     architectures). Underestimating the cache size will result in
 *     the use of steaming stores even if the working set would fit
 *     just fine into the caches, easily resulting in a performance
 *     hit of 1500% (e.g. 0.4 GLUPS instead of 6 GLUPS for a 3D Jacobi
 *     on an Intel i7-6700HQ). Bottom line: never underestimate the
 *     cache size!
 */
template<typename CARGO, typename ACCESSOR, int LAST_LEVEL_CACHE_SIZE_ESTIMATE = (1 << 25)>
class estimate_optimum_short_vec_type
{
public:
    // Revert to scalar values when running on a CUDA device. The
    // vector unit is much wider, but from a programming PoV it's
    // scalar:
#ifdef __CUDA_ARCH__
    static const int ARITY = 1;
#else
    // for IBM Blue Gene/Q's QPX, which is mutually exclusive to
    // Intel/AMD's AVX/SSE or ARM's NEON ISAs:
#  ifdef __VECTOR4DOUBLE__
    static const int BIT_WIDTH = 256;
#  endif

    // Dito for ARM NEON:
#  ifdef __ARM_NEON__
    static const int BIT_WIDTH = 128;
#  endif

    // Only the case of the IBM PC is complicated. No thanks to you,
    // history!
#  ifdef __AVX512F__
    static const int BIT_WIDTH = 512;
#  else
#    ifdef __AVX__
    static const int BIT_WIDTH = 256;
#    else
#      ifdef __SSE__
    static const int BIT_WIDTH = 128;
#      else
    static const int BIT_WIDTH = sizeof(CARGO);
#      endif
#    endif
#  endif

    // rule of thumb: 2x loop unrolling for CPUs:
    static const int ARITY = 2 * BIT_WIDTH / sizeof(CARGO) / 8;
#endif

    static const int STREAMING_FLAG =
        ACCESSOR::DIM_PROD * sizeof(typename ACCESSOR::element_type) / LAST_LEVEL_CACHE_SIZE_ESTIMATE;

    typedef typename detail::flat_array::streaming_short_vec_switch<CARGO, ARITY, STREAMING_FLAG>::VALUE VALUE;
};

}

#endif
