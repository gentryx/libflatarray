/**
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_HELPERS_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_HELPERS_HPP

#include <libflatarray/config.h>

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <cassert>

// uintptr_t is only available through C++11
#ifdef LIBFLATARRAY_WITH_CPP14
#include <cstdint>
#define _SHORTVEC_UINTPTR_T std::uintptr_t
#else
#define _SHORTVEC_UINTPTR_T unsigned long long
#endif

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

/**
 * This macro asserts that the pointer is correctly aligned.
 *
 * @param ptr pointer to check
 * @param alignment alignement
 */
#define SHORTVEC_ASSERT_ALIGNED(ptr, alignment)                         \
    do {                                                                \
        assert((reinterpret_cast<_SHORTVEC_UINTPTR_T>(ptr) % (alignment)) == 0); \
    } while (0)

/**
 * For some implementations there is the problem, that the compiler does not
 * see, that some variables should be used uninitialized.
 * Therefore here are compiler specific macros to disable and enable this warning.
 */
#if defined(__GNUC__) && !defined(__clang__)
#define SHORTVEC_DISABLE_WARNING_UNINITIALIZED             \
    _Pragma("GCC diagnostic push")                         \
    _Pragma("GCC diagnostic ignored \"-Wuninitialized\"")
#define SHORTVEC_ENABLE_WARNING_UNINITIALIZED   \
    _Pragma("GCC diagnostic pop")
#endif

#ifdef __clang__
#define SHORTVEC_DISABLE_WARNING_UNINITIALIZED              \
    _Pragma("clang diagnostic push")                        \
    _Pragma("clang diagnostic ignored \"-Wuninitialized\"")
#define SHORTVEC_ENABLE_WARNING_UNINITIALIZED   \
    _Pragma("clang diagnostic pop")
#endif

/**
 * If compiler is not gcc and not clang, just remove these macros.
 */
#ifndef SHORTVEC_DISABLE_WARNING_UNINITIALIZED
#define SHORTVEC_DISABLE_WARNING_UNINITIALIZED
#endif
#ifndef SHORTVEC_ENABLE_WARNING_UNINITIALIZED
#define SHORTVEC_ENABLE_WARNING_UNINITIALIZED
#endif


#ifdef __SSE4_1__

/**
 * Insertps instruction which allows to insert an memory location
 * into a xmm register.
 * Instruction: insertps xmm, xmm/m32, imm8
 *
 * @param a xmm register
 * @param base base pointer
 * @param offset offset
 * @param idx index, has to be a constant number like 0x10, no variable
 */
#define SHORTVEC_INSERT_PS(a, base, offset, idx)                        \
    do {                                                                \
        asm volatile ("insertps %1, (%q2, %q3, 4), %0\n"                \
                      : "+x" (a) : "N" (idx), "r" (base), "r" (offset) : "memory"); \
    } while (0)

#endif

#ifdef __AVX__

/**
 * Same as above just for AVX.
 * Instruction: vinsertps xmm, xmm, xmm/m32, imm8
 *
 * @param a xmm register
 * @param base base pointer
 * @param offset offset
 * @param idx index, has to be a constant number like 0x10, no variable
 */
#define SHORTVEC_INSERT_PS_AVX(a, base, offset, idx)                    \
    do {                                                                \
        asm volatile ("vinsertps %1, (%q2, %q3, 4), %0, %0\n"           \
                      : "+x" (a) : "N" (idx), "r" (base), "r" (offset) : "memory"); \
    } while (0)

#endif

namespace LibFlatArray {

namespace ShortVecHelpers {

#ifdef __SSE4_1__

/**
 * _mm_extract_ps returns an integer, but we need a float.
 * This union can be used to get a float back.
 */
union ExtractResult {
    int i;
    float f;
};

#endif

}

}

#endif
