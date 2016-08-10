/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_DUAL_CALLBACK_HELPER_HPP
#define FLAT_ARRAY_DETAIL_DUAL_CALLBACK_HELPER_HPP

namespace LibFlatArray {

namespace detail {

namespace flat_array {

template<typename ACCESSOR1, typename FUNCTOR>
class dual_callback_helper2
{
public:
    dual_callback_helper2(ACCESSOR1& accessor1, FUNCTOR& functor) :
        accessor1(accessor1),
        functor(functor)
    {}

    template<typename ACCESSOR2>
    void operator()(ACCESSOR2& accessor2)
    {
        functor(accessor1, accessor2);
    }

private:
    ACCESSOR1& accessor1;
    FUNCTOR& functor;
};

template<typename GRID2, typename FUNCTOR>
class dual_callback_helper1
{
public:
    dual_callback_helper1(GRID2 *grid2, const FUNCTOR& functor) :
        grid2(grid2),
        functor(functor)
    {}

    template<typename ACCESSOR1>
    void operator()(ACCESSOR1& accessor1) const
    {
        dual_callback_helper2<ACCESSOR1, FUNCTOR> helper(accessor1, functor);
        grid2->callback(helper);
    }

private:
    GRID2 *grid2;
    FUNCTOR& functor;
};

class dual_callback_helper
{
public:
    template<typename GRID1, typename GRID2, typename FUNCTOR>
    void operator()(GRID1 *gridOld, GRID2 *gridNew, FUNCTOR& functor)
    {
        dual_callback_helper1<GRID2, FUNCTOR> helper(gridNew, functor);
        gridOld->callback(helper);
    }
};

template<typename GRID_TYPE, typename FUNCTOR>
class dual_callback_helper_symmetric
{
public:
    dual_callback_helper_symmetric(GRID_TYPE *other_grid, FUNCTOR& functor) :
        other_grid(other_grid),
        functor(functor)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR& accessor1) const
    {
        ACCESSOR accessor2(other_grid->get_data());

        functor(accessor1, accessor2);
    }

private:
    GRID_TYPE *other_grid;
    FUNCTOR& functor;
};

template<typename GRID_TYPE, typename FUNCTOR>
class const_dual_callback_helper_symmetric
{
public:
    const_dual_callback_helper_symmetric(GRID_TYPE *other_grid, const FUNCTOR& functor) :
        other_grid(other_grid),
        functor(functor)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR& accessor1) const
    {
        ACCESSOR accessor2(other_grid->get_data(), 0);

        functor(accessor1, accessor2);
    }

private:
    GRID_TYPE *other_grid;
    const FUNCTOR& functor;
};

}

}

}

#endif
