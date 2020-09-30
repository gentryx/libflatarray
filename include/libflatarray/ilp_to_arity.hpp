/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_ILP_TO_ARITY_HPP
#define FLAT_ARRAY_ILP_TO_ARITY_HPP

#include <libflatarray/short_vec.hpp>

namespace LibFlatArray {

/**
 * This class allows users to select the arity of a short_vec type by
 * specifying the desired degree of instruction level parallelism
 * (i.e. loop unrolling factor). For instance, setting ILP to 4 for
 * double on an AVX-cabable CPU would yield short_vec<duble, 16>, but
 * for a SSE-only CPU it would return a short_vec<double, 8>.
 */
template<typename CARGO, std::size_t ILP>
class ilp_to_arity
{
public:
    // Revert to scalar values when running on a CUDA device. The
    // vector unit is much wider, but from a programming PoV it's
    // scalar:
#ifdef __CUDA_ARCH__
    static const std::size_t ARITY = 1;
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
#  if !defined(__CUDA_ARCH__) && !defined(__ARM_NEON__) && !defined(__MIC__)
#    ifdef LFA_AVX512_HELPER
    static const int BIT_WIDTH = 512;
#    else
#      ifdef __AVX__
    static const int BIT_WIDTH = 256;
#      else
#        ifdef __SSE__
    static const int BIT_WIDTH = 128;
#        else
    static const int BIT_WIDTH = sizeof(CARGO) * 8;
#        endif
#      endif
#    endif
#  endif
    static const std::size_t ARITY = ILP * BIT_WIDTH / sizeof(CARGO) / 8;
#endif

};

}

#endif
