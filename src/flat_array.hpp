/**
 * Copyright 2012-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_FLAT_ARRAY_HPP
#define FLAT_ARRAY_FLAT_ARRAY_HPP

#include <libflatarray/detail/macros.hpp>
#include <libflatarray/detail/offset.hpp>
#include <libflatarray/aligned_allocator.hpp>

#ifdef __CUDACC__
#include <libflatarray/cuda_allocator.hpp>
#endif

#include <libflatarray/coord.hpp>
#include <libflatarray/estimate_optimum_short_vec_type.hpp>
#include <libflatarray/macros.hpp>
#include <libflatarray/member_ptr_to_offset.hpp>
#include <libflatarray/number_of_members.hpp>
#include <libflatarray/short_vec.hpp>
#include <libflatarray/streaming_short_vec.hpp>
#include <libflatarray/soa_accessor.hpp>
#include <libflatarray/soa_array.hpp>
#include <libflatarray/soa_grid.hpp>

#endif
