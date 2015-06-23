#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_MIC_DOUBLE_32_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_MIC_DOUBLE_32_HPP

#ifdef __MIC__

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
class short_vec<double, 32>
{
public:
    static const int ARITY = 32;

    typedef short_vec_strategy::mic strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<double, 32>& vec);

    inline
    short_vec(const double data = 0) :
        val1(_mm512_set1_pd(data)),
        val2(_mm512_set1_pd(data)),
        val3(_mm512_set1_pd(data)),
        val4(_mm512_set1_pd(data))
    {}

    inline
    short_vec(const double *data)
    {
        load(data);
    }

    inline
    short_vec(const __m512d& val1, const __m512d& val2, const __m512d& val3, const __m512d& val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
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
    void operator-=(const short_vec<double, 32>& other)
    {
        val1 = _mm512_sub_pd(val1, other.val1);
        val2 = _mm512_sub_pd(val2, other.val2);
        val3 = _mm512_sub_pd(val3, other.val3);
        val4 = _mm512_sub_pd(val4, other.val4);
    }

    inline
    short_vec<double, 32> operator-(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm512_sub_pd(val1, other.val1),
            _mm512_sub_pd(val2, other.val2),
            _mm512_sub_pd(val3, other.val3),
            _mm512_sub_pd(val4, other.val4));
    }

    inline
    void operator+=(const short_vec<double, 32>& other)
    {
        val1 = _mm512_add_pd(val1, other.val1);
        val2 = _mm512_add_pd(val2, other.val2);
        val3 = _mm512_add_pd(val3, other.val3);
        val4 = _mm512_add_pd(val4, other.val4);
    }

    inline
    short_vec<double, 32> operator+(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm512_add_pd(val1, other.val1),
            _mm512_add_pd(val2, other.val2),
            _mm512_add_pd(val3, other.val3),
            _mm512_add_pd(val4, other.val4));
    }

    inline
    void operator*=(const short_vec<double, 32>& other)
    {
        val1 = _mm512_mul_pd(val1, other.val1);
        val2 = _mm512_mul_pd(val2, other.val2);
        val3 = _mm512_mul_pd(val3, other.val3);
        val4 = _mm512_mul_pd(val4, other.val4);
    }

    inline
    short_vec<double, 32> operator*(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm512_mul_pd(val1, other.val1),
            _mm512_mul_pd(val2, other.val2),
            _mm512_mul_pd(val3, other.val3),
            _mm512_mul_pd(val4, other.val4));
    }

    inline
    void operator/=(const short_vec<double, 32>& other)
    {
        val1 = _mm512_div_pd(val1, other.val1);
        val2 = _mm512_div_pd(val2, other.val2);
        val3 = _mm512_div_pd(val3, other.val3);
        val4 = _mm512_div_pd(val4, other.val4);
    }

    inline
    short_vec<double, 32> operator/(const short_vec<double, 32>& other) const
    {
        return short_vec<double, 32>(
            _mm512_div_pd(val1, other.val1),
            _mm512_div_pd(val2, other.val2),
            _mm512_div_pd(val3, other.val3),
            _mm512_div_pd(val4, other.val4));
    }

    inline
    short_vec<double, 32> sqrt() const
    {
        return short_vec<double, 32>(
            _mm512_sqrt_pd(val1),
            _mm512_sqrt_pd(val2),
            _mm512_sqrt_pd(val3),
            _mm512_sqrt_pd(val4));
    }

    inline
    void load(const double *data)
    {
        val1 = _mm512_loadunpacklo_pd(val1, data +  0);
        val1 = _mm512_loadunpackhi_pd(val1, data +  8);
        val2 = _mm512_loadunpacklo_pd(val2, data +  8);
        val2 = _mm512_loadunpackhi_pd(val2, data + 16);
        val3 = _mm512_loadunpacklo_pd(val3, data + 16);
        val3 = _mm512_loadunpackhi_pd(val3, data + 24);
        val4 = _mm512_loadunpacklo_pd(val4, data + 24);
        val4 = _mm512_loadunpackhi_pd(val4, data + 32);
    }

    inline
    void load_aligned(const double *data)
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        val1 = _mm512_load_pd(data +  0);
        val2 = _mm512_load_pd(data +  8);
        val3 = _mm512_load_pd(data + 16);
        val4 = _mm512_load_pd(data + 24);
    }

    inline
    void store(double *data) const
    {
        _mm512_packstorelo_pd(data +  0, val1);
        _mm512_packstorehi_pd(data +  8, val1);
        _mm512_packstorelo_pd(data +  8, val2);
        _mm512_packstorehi_pd(data + 16, val2);
        _mm512_packstorelo_pd(data + 16, val3);
        _mm512_packstorehi_pd(data + 24, val3);
        _mm512_packstorelo_pd(data + 24, val4);
        _mm512_packstorehi_pd(data + 32, val4);
    }

    inline
    void store_aligned(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_store_pd(data +  0, val1);
        _mm512_store_pd(data +  8, val2);
        _mm512_store_pd(data + 16, val3);
        _mm512_store_pd(data + 24, val4);
    }

    inline
    void store_nt(double *data) const
    {
        SHORTVEC_ASSERT_ALIGNED(data, 64);
        _mm512_storenr_pd(data +  0, val1);
        _mm512_storenr_pd(data +  8, val2);
        _mm512_storenr_pd(data + 16, val3);
        _mm512_storenr_pd(data + 24, val4);
    }

    inline
    void gather(const double *ptr, const unsigned *offsets)
    {
        SHORTVEC_ASSERT_ALIGNED(offsets, 64);
        __m512i indices;
        indices = _mm512_load_epi32(offsets + 0);
        val1    = _mm512_i32logather_pd(indices, ptr, 8);
        indices = _mm512_permute4f128_epi32(indices, _MM_PERM_BADC);
        val2    = _mm512_i32logather_pd(indices, ptr, 8);
        indices = _mm512_load_epi32(offsets + 16);
        val3    = _mm512_i32logather_pd(indices, ptr, 8);
        indices = _mm512_permute4f128_epi32(indices, _MM_PERM_BADC);
        val4    = _mm512_i32logather_pd(indices, ptr, 8);
    }

    inline
    void scatter(double *ptr, const unsigned *offsets) const
    {
        SHORTVEC_ASSERT_ALIGNED(offsets, 64);
        __m512i indices;
        indices = _mm512_load_epi32(offsets + 0);
        _mm512_i32loscatter_pd(ptr, indices, val1, 8);
        indices = _mm512_permute4f128_epi32(indices, _MM_PERM_BADC);
        _mm512_i32loscatter_pd(ptr, indices, val2, 8);
        indices = _mm512_load_epi32(offsets + 16);
        _mm512_i32loscatter_pd(ptr, indices, val3, 8);
        indices = _mm512_permute4f128_epi32(indices, _MM_PERM_BADC);
        _mm512_i32loscatter_pd(ptr, indices, val4, 8);
    }

private:
    __m512d val1;
    __m512d val2;
    __m512d val3;
    __m512d val4;
};

inline
void operator<<(double *data, const short_vec<double, 32>& vec)
{
    vec.store(data);
}

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<double, 32> sqrt(const short_vec<double, 32>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<double, 32>& vec)
{
    const double *data1 = reinterpret_cast<const double *>(&vec.val1);
    const double *data2 = reinterpret_cast<const double *>(&vec.val2);
    const double *data3 = reinterpret_cast<const double *>(&vec.val3);
    const double *data4 = reinterpret_cast<const double *>(&vec.val4);

    __os << "["  << data1[0] << ", " << data1[1] << ", " << data1[2] << ", " << data1[3]
         << ", " << data1[4] << ", " << data1[5] << ", " << data1[6] << ", " << data1[7]
         << ", " << data2[0] << ", " << data2[1] << ", " << data2[2] << ", " << data2[3]
         << ", " << data2[4] << ", " << data2[5] << ", " << data2[6] << ", " << data2[7]
         << ", " << data3[0] << ", " << data3[1] << ", " << data3[2] << ", " << data3[3]
         << ", " << data3[4] << ", " << data3[5] << ", " << data3[6] << ", " << data3[7]
         << ", " << data4[0] << ", " << data4[1] << ", " << data4[2] << ", " << data4[3]
         << ", " << data4[4] << ", " << data4[5] << ", " << data4[6] << ", " << data4[7]
         << "]";
    return __os;
}

}

#endif
#endif

#endif
