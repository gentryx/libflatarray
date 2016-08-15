/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_STREAMING_SHORT_VEC_HPP
#define FLAT_ARRAY_STREAMING_SHORT_VEC_HPP

#include <libflatarray/short_vec.hpp>

namespace LibFlatArray {

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

/**
 * Wraps functionality of short_vec, but replaces all stores by
 * streaming (i.e. non-temporal) stores. Downside: all store addresses
 * must be aligned.
 */
template<typename CARGO, int ARITY>
class streaming_short_vec : public short_vec<CARGO, ARITY>
{
public:

    inline
    streaming_short_vec(const CARGO val = 0) : short_vec<CARGO, ARITY>(val)
    {}

    template<typename INIT_TYPE>
    inline
    streaming_short_vec(const INIT_TYPE val) : short_vec<CARGO, ARITY>(val)
    {}

    inline
    void store(CARGO *data)
    {
        short_vec<CARGO, ARITY>::store_nt(data);
    }

    inline
    void store_aligned(CARGO *data)
    {
        short_vec<CARGO, ARITY>::store_nt(data);
    }
};

#ifdef __ICC
#pragma warning pop
#endif

inline
void operator<<(double *data, const streaming_short_vec<double, 8>& vec)
{
    vec.store_nt(data);
}

}

#endif
