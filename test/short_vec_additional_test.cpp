/**
 * Copyright 2014-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

// include here to have another object file and check if linking with
// origninal test still works.
#include <libflatarray/short_vec.hpp>

// globally disable some warnings with MSVC, that are issued not for a
// specific header, but rather for the interaction of system headers
// and LibFlatArray source:
#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
