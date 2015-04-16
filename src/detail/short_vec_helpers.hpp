#ifndef _SHORT_VEC_HELPERS_H_
#define _SHORT_VEC_HELPERS_H_

namespace LibFlatArray {

namespace ShortVecHelpers {

#ifdef __SSE4_1__

#include <smmintrin.h>

/**
 * _mm_insert_ps needs a __m128 as second parameter, which is kind of annoying
 * since we need just a pointer, which is actually supported by the hardware
 * -> simply define a new function: _mm_insert_ps2...
 */
inline
void _mm_insert_ps2(__m128& a, const float *base, unsigned offset, int idx)
{
    // instruction: insertps xmm, xmm/m32, imm8
    asm volatile (
        "insertps %0, (%q1, %q2, 4), %3\n"
        : : "K" (idx), "r" (base), "r" (offset), "x" (a));
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
        "vinsertps %0, (%q1, %q2, 4), %3, %3\n"
        : : "K" (idx), "r" (base), "r" (offset), "x" (a));
}
#endif

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

}

}

#endif /* _SHORT_VEC_HELPERS_H_ */
