#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_DOUBLE_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_DOUBLE_16_HPP

#ifdef __AVX512F__

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>
#include <libflatarray/detail/short_vec_helpers.hpp>
#include <libflatarray/config.h>

#ifdef LIBFLATARRAY_WITH_CPP14
#include <initializer_list>
#endif

#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<double, 16>
{
public:
    static const int ARITY = 16;

    typedef short_vec_strategy::avx512 strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 16>& vec);

    inline
    short_vec(const double data = 0) :
        val1(_mm512_set1_pd(data)),
        val2(_mm512_set1_pd(data))
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(const __m512d& val1, const __m512d& val2) :
        val1(val1),
        val2(val2)
    {}

#ifdef LIBFLATARRAY_WITH_CPP14
    inline
    short_vec(const std::initializer_list<double>& il)
    {
        const double *ptr = static_cast<const double *>(&(*il.begin()));
        load(ptr);
    }
#endif

    inline
    void operator-=(const short_vec<double, 16>& other)
    {
        val1 = _mm512_sub_pd(val1, other.val1);
        val2 = _mm512_sub_pd(val2, other.val2);
    }

    inline
    short_vec<double, 16> operator-(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm512_sub_pd(val1, other.val1),
            _mm512_sub_pd(val2, other.val2));
    }

    inline
    void operator+=(const short_vec<double, 16>& other)
    {
        val1 = _mm512_add_pd(val1, other.val1);
        val2 = _mm512_add_pd(val2, other.val2);
    }

    inline
    short_vec<double, 16> operator+(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm512_add_pd(val1, other.val1),
            _mm512_add_pd(val2, other.val2));
    }

    inline
    void operator*=(const short_vec<double, 16>& other)
    {
        val1 = _mm512_mul_pd(val1, other.val1);
        val2 = _mm512_mul_pd(val2, other.val2);
    }

    inline
    short_vec<double, 16> operator*(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm512_mul_pd(val1, other.val1),
            _mm512_mul_pd(val2, other.val2));
    }

    inline
    void operator/=(const short_vec<double, 16>& other)
    {
        val1 = _mm512_div_pd(val1, other.val1);
        val2 = _mm512_div_pd(val2, other.val2);
    }

    inline
    short_vec<double, 16> operator/(const short_vec<double, 16>& other) const
    {
        return short_vec<double, 16>(
            _mm512_div_pd(val1, other.val1),
            _mm512_div_pd(val2, other.val2));
    }

    inline
    short_vec<double, 16> sqrt() const
    {
        return short_vec<double, 16>(
            _mm512_sqrt_pd(val1),
            _mm512_sqrt_pd(val2));
    }

    inline
    void load(const double *data)
    {
        val1 = _mm512_loadu_pd(data + 0);
        val2 = _mm512_loadu_pd(data + 8);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        val1 = _mm512_load_pd(data + 0);
        val2 = _mm512_load_pd(data + 8);
    }

    inline
    void store(double *data) const
    {
        _mm512_storeu_pd(data + 0, val1);
        _mm512_storeu_pd(data + 8, val2);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_store_pd(data + 0, val1);
        _mm512_store_pd(data + 8, val2);
    }

    inline
    void store_nt(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_stream_pd(data + 0, val1);
        _mm512_stream_pd(data + 8, val2);
    }

    inline
    void gather(const double *ptr, const unsigned *offsets)
    {
        __m256i indices;
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets));
        val1    = _mm512_i32gather_pd(indices, ptr, 8);
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets + 8));
        val2    = _mm512_i32gather_pd(indices, ptr, 8);
    }

    inline
    void scatter(double *ptr, const unsigned *offsets) const
    {
        __m256i indices;
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets));
        _mm512_i32scatter_pd(ptr, indices, val1, 8);
        indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets + 8));
        _mm512_i32scatter_pd(ptr, indices, val2, 8);
    }

private:
    __m512d val1;
    __m512d val2;
};

inline
void operator<<(double *data, const short_vec<double, 16>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<double, 16> sqrt(const short_vec<double, 16>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 16>& vec)
{
    const double *data1 = reinterpret_cast<const double *>(&vec.val1);
    const double *data2 = reinterpret_cast<const double *>(&vec.val2);

    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << ", " << data1[4] << ", " << data1[5] << ", " << data1[6] << ", " << data1[7]
         << ", " << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
         << ", " << data2[4] << ", " << data2[5] << ", " << data2[6] << ", " << data2[7]
         << "]";
    return __os;
}

}

#endif
#endif

#endif
