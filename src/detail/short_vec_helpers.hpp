#ifndef _SHORT_VEC_HELPERS_H_
#define _SHORT_VEC_HELPERS_H_

#include <libflatarray/config.h>
#include <cassert>

// uintptr_t is only available through C++11
#ifdef LIBFLATARRAY_WITH_CPP14
# include <cstdint>
# define UINTPTR_T uintptr_r
#else
# define UINTPTR_T unsigned long long
#endif

#ifdef __SSE4_1__
# include <smmintrin.h>
#endif

/**
 * This macro asserts that the pointer is correctly aligned.
 *
 * @param ptr pointer to check
 * @param alignment alignement
 */
#define SHORTVEC_ASSERT_ALIGNED(ptr, alignment)                         \
    do {                                                                \
        assert((reinterpret_cast<UINTPTR_T>(ptr) % (alignment)) == 0);  \
    } while (0)
#undef UINTPTR_T

/**
 * For some implementations there is the problem, that the compiler does not
 * see, that some variables should be used uninitialized.
 * Therefore here are compiler specific macros to disable and enable this warning.
 */
#if defined(__GNUC__) && !defined(__clang__)
# define SHORTVEC_DISABLE_WARNING_UNINITIALIZED             \
    _Pragma("GCC diagnostic push")							\
    _Pragma("GCC diagnostic ignored \"-Wuninitialized\"")
# define SHORTVEC_ENABLE_WARNING_UNINITIALIZED  \
    _Pragma("GCC diagnostic pop")
#endif

#ifdef __clang__
# define SHORTVEC_DISABLE_WARNING_UNINITIALIZED             \
    _Pragma("clang diagnostic push")						\
    _Pragma("clang diagnostic ignored \"-Wuninitialized\"")
# define SHORTVEC_ENABLE_WARNING_UNINITIALIZED  \
    _Pragma("clang diagnostic pop")
#endif

/**
 * If compiler is not gcc and not clang, just remove these macros.
 */
#ifndef SHORTVEC_DISABLE_WARNING_UNINITIALIZED
# define SHORTVEC_DISABLE_WARNING_UNINITIALIZED
#endif
#ifndef SHORTVEC_ENABLE_WARNING_UNINITIALIZED
# define SHORTVEC_ENABLE_WARNING_UNINITIALIZED
#endif


namespace LibFlatArray {

namespace ShortVecHelpers {

#ifdef __SSE4_1__

/**
 * _mm_insert_ps needs a __m128 as second parameter, which is kind of annoying
 * since we need just a pointer, which is actually supported by the hardware
 * -> simply define a new function: _mm_insert_ps2...
 */
inline
void _mm_insert_ps2(__m128& a, const float *base, unsigned offset, int idx)
{
    // instruction: insertps xmm, xmm/m32, imm8
    asm volatile ("insertps %1, (%q2, %q3, 4), %0\n"
                  : "+x" (a) : "K" (idx), "r" (base), "r" (offset) : "memory");
}

/**
 * _mm_extract_ps returns an integer, but we need a float.
 * This union can be used to get a float back.
 */
union ExtractResult {
    int i;
    float f;
};

#endif

#ifdef __AVX__

/**
 * For AVX we can use the vinsertps instruction.
 */
inline
void _mm_insert_ps2_avx(__m128& a, const float *base, unsigned offset, int idx)
{
    // vinsertps xmm, xmm, xmm/m32, imm8
    asm volatile (
        "vinsertps %1, (%q2, %q3, 4), %0, %0\n"
        : "+x" (a) : "K" (idx), "r" (base), "r" (offset) : "memory");
}
#endif

}

}

#endif /* _SHORT_VEC_HELPERS_H_ */
