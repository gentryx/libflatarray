/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_API_TRAITS_HPP
#define FLAT_ARRAY_API_TRAITS_HPP

#include <libflatarray/macros.hpp>
#include <libflatarray/soa_accessor.hpp>
#include <stdexcept>

namespace LibFlatArray {

/**
 * This class provides means of describing properties of user classes
 * and discovering them from library code. The crucial benefit is that
 * we can define default values if a property is undefined. By this we
 * can achive forward compatibility of user code.
 *
 * This class is equivalent to LibGeoDecomp's APITraits class.
 */
class api_traits
{
public:
    class false_type
    {};

    class true_type
    {};

    /**
     * select_sizes:
     *
     * use the following trait to specify for which grid sizes the
     * soa_functors need to be instantiated. chosing more sizes
     * increases compile time, but also improves running time and
     * reduces memory consumption. conversely, with less sizes to be
     * instantiated compilation is quicker (and can be done with a
     * lower memory footprint), but the code might not be as fast, and
     * might allocate excessive amounts of memory.
     */

    /**
     * Means that the code will be used with rectangular 2D grids
     * only, the z-dimension will be tied at 1.
     */
    class has_default_2d_sizes
    {
    public:
        LIBFLATARRAY_CUSTOM_SIZES(
            (32)(64)(128)(192)(256)(512)(544)(1056),
            (32)(64)(128)(192)(256)(512)(544)(1056),
            (1))
    };

    /**
     * Means that the code will be used with rectangular 3D grids.
     * This wastes memory if your z-dimension is always 1 (i.e. a 2D
     * code). Compilation also takes significantly longer. Can improve
     * running time by 20% to 30%:
     *
     * http://dl.acm.org/citation.cfm?id=2476888
     * http://hpc.pnl.gov/conf/wolfhpc/2012/talks/schafer.pdf
     */
    class has_default_3d_sizes
    {
    public:
        LIBFLATARRAY_CUSTOM_SIZES(
            (32)(64)(128)(192)(256)(512)(544)(1056),
            (32)(64)(128)(192)(256)(512)(544)(1056),
            (32)(64)(128)(192)(256)(512)(544)(1056))
    };

    /**
     * This lets the user choose relevant grid sizes himself. Please
     * also see LIBFLATARRAY_CUSTOM_SIZES() and
     * LIBFLATARRAY_CUSTOM_SIZES_UNIFORM().
     */
    class has_custom_sizes
    {
        typedef void has_sizes;
    };

    template<typename CELL, typename HAS_SIZES = void>
    class select_sizes
    {
    public:
        template<typename FUNCTOR>
        void operator()(
            char *data,
            FUNCTOR& functor,
            const std::size_t dim_x = 1,
            const std::size_t dim_y = 1,
            const std::size_t dim_z = 1)
        {
            has_default_3d_sizes api;
            api.template select_size<CELL>(
                data,
                functor,
                dim_x,
                dim_y,
                dim_z);
        }

        template<typename FUNCTOR>
        void operator()(
            const char *data,
            FUNCTOR& functor,
            const std::size_t dim_x = 1,
            const std::size_t dim_y = 1,
            const std::size_t dim_z = 1)
        {
            has_default_3d_sizes api;
            api.template select_size<CELL>(
                data,
                functor,
                dim_x,
                dim_y,
                dim_z);
        }
    };

    template<typename CELL>
    class select_sizes<CELL, typename CELL::API::has_sizes>
    {
    public:
        template<typename FUNCTOR>
        void operator()(
            char *data,
            FUNCTOR& functor,
            const std::size_t dim_x = 1,
            const std::size_t dim_y = 1,
            const std::size_t dim_z = 1)
        {
            typename CELL::API api;
            api.template select_size<CELL>(
                data,
                functor,
                dim_x,
                dim_y,
                dim_z);
        }

        template<typename FUNCTOR>
        void operator()(
            const char *data,
            FUNCTOR& functor,
            const std::size_t dim_x = 1,
            const std::size_t dim_y = 1,
            const std::size_t dim_z = 1)
        {
            typename CELL::API api;
            api.template select_size<CELL>(
                data,
                functor,
                dim_x,
                dim_y,
                dim_z);
        }
    };

    /**
     * select_asymmetric_dual_callback:
     *
     * If you need to use a functor together with two grids (as you
     * would in most computer simulations), you can use this trait to
     * control whether both grids are always the same size (i.e. you
     * need no asymmetric callback), or if their sizes may differ
     * (makes sense if you need to do some sorts of reductions or run
     * a multi-grid code).
     *
     * WARNING: if soa_accessor is instantiated for n sizes, then
     * has_asymmetric_dual_callback will increase this to n * n.
     */

    class has_asymmetric_dual_callback
    {
    public:
        typedef void has_asymmetric_dual_callback_specified;
        typedef true_type asymmetric_dual_callback_required;
    };

    class has_no_asymmetric_dual_callback
    {
    public:
        typedef void has_asymmetric_dual_callback_specified;
        typedef false_type asymmetric_dual_callback_required;
    };

    template<typename CELL, typename HAS_ASYMMETRIC_DUAL_CALLBACK_SPECIFIED = void>
    class select_asymmetric_dual_callback
    {
    public:
        typedef false_type value;
    };

    template<typename CELL>
    class select_asymmetric_dual_callback<CELL, typename CELL::API::has_asymmetric_dual_callback_specified>
    {
    public:
        typedef typename CELL::API::asymmetric_dual_callback_required value;
    };
};

}

#endif
