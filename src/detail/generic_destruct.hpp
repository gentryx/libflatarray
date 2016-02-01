/**
 * Copyright 2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_GENERIC_DESTRUCT_HPP
#define FLAT_ARRAY_DETAIL_GENERIC_DESTRUCT_HPP

// this fixes compilation for non-cuda builds
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace LibFlatArray {

namespace detail {

namespace flat_array {

template<typename TYPENAME>
__host__ __device__
inline void generic_destruct(TYPENAME *member)
{
    member->~TYPENAME();
}

// primitive types don't have d-tors:
__host__ __device__
inline void generic_destruct(char *member)
{}

__host__ __device__
inline void generic_destruct(float *member)
{}

__host__ __device__
inline void generic_destruct(double *member)
{}

__host__ __device__
inline void generic_destruct(int *member)
{}

__host__ __device__
inline void generic_destruct(unsigned *member)
{}

__host__ __device__
inline void generic_destruct(long *member)
{}

}

}

}

#endif
