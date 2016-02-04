/**
 * Copyright 2014-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SOA_GRID_HPP
#define FLAT_ARRAY_SOA_GRID_HPP

#include <libflatarray/aligned_allocator.hpp>
#include <libflatarray/api_traits.hpp>
#include <libflatarray/detail/copy_functor.hpp>
#include <libflatarray/detail/construct_functor.hpp>
#include <libflatarray/detail/destroy_functor.hpp>
#include <libflatarray/detail/dual_callback_helper.hpp>
#include <libflatarray/detail/get_set_instance_functor.hpp>
#include <libflatarray/detail/load_save_functor.hpp>
#include <libflatarray/detail/set_byte_size_functor.hpp>

#include <stdexcept>

namespace LibFlatArray {

/**
 * soa_grid is a 1D - 3D container with "Struct of Arrays"-style
 * memory layout but "Array of Structs"-style user interface. Another
 * feature is its 0-overhead address calculation for accesses to
 * members and neighboring elements. The technique has been published
 * here:
 *
 *   http://www.computer.org/csdl/proceedings/sccompanion/2012/4956/00/4956b139-abs.html
 *
 * The AoS user interface aids productivty by allowing the user to
 * write object-oriented kernels. Thanks to the SoA layout, these
 * kernels can still be efficiently vectorized on both, multi-cores
 * and GPUs.
 *
 * The 0-overhead interface trades compile time for running time (and
 * development time). Compilation time may jump from seconds to tens
 * of minutes. See the LBM example for how to split compilation into
 * smaller, parallelizable chunks.
 */
template<typename CELL_TYPE, typename ALLOCATOR = aligned_allocator<char, 4096> >
class soa_grid
{
public:
    friend class TestAssignment;

    explicit soa_grid(std::size_t dim_x = 0, std::size_t dim_y = 0, std::size_t dim_z = 0) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z),
        data(0)
    {
        resize();
    }

    soa_grid(soa_grid& other) :
        dim_x(other.dim_x),
        dim_y(other.dim_y),
        dim_z(other.dim_z),
        my_byte_size(other.byte_size()),
        data(ALLOCATOR().allocate(other.byte_size()))
    {
        init();
        copy_in(other);
    }

    soa_grid(const soa_grid& other) :
        dim_x(other.dim_x),
        dim_y(other.dim_y),
        dim_z(other.dim_z),
        my_byte_size(other.byte_size()),
        data(ALLOCATOR().allocate(other.byte_size()))
    {
        init();
        copy_in(other);
    }

    ~soa_grid()
    {
        destroy();
        ALLOCATOR().deallocate(data, byte_size());
    }

    soa_grid& operator=(const soa_grid& other)
    {
        resize(other.dim_x, other.dim_y, other.dim_z);
        callback(detail::flat_array::construct_functor<CELL_TYPE>(dim_x, dim_y, dim_z));
        return *this;
    }

    void swap(soa_grid& other)
    {
        using std::swap;
        swap(dim_x, other.dim_x);
        swap(dim_x, other.dim_x);
        swap(dim_x, other.dim_x);
        swap(my_byte_size, other.my_byte_size);
        swap(data, other.data);
    }

    void resize(std::size_t new_dim_x, std::size_t new_dim_y, std::size_t new_dim_z)
    {
        if ((dim_x == new_dim_x) &&
            (dim_y == new_dim_y) &&
            (dim_z == new_dim_z)) {
            return;
        }
        dim_x = new_dim_x;
        dim_y = new_dim_y;
        dim_z = new_dim_z;
        resize();
    }

    template<typename FUNCTOR>
    void callback(FUNCTOR functor)
    {
        api_traits::select_sizes<CELL_TYPE>()(data, functor, dim_x, dim_y, dim_z);
    }

    template<typename FUNCTOR>
    void callback(FUNCTOR functor) const
    {
        api_traits::select_sizes<CELL_TYPE>()(data, functor, dim_x, dim_y, dim_z);
    }

    template<typename FUNCTOR>
    void callback(soa_grid<CELL_TYPE> *other_grid, const FUNCTOR& functor)
    {
        typedef typename api_traits::select_asymmetric_dual_callback<CELL_TYPE>::value value;
        dual_callback(other_grid, functor, value());
    }

    template<typename FUNCTOR>
    void callback(soa_grid<CELL_TYPE> *other_grid, const FUNCTOR& functor) const
    {
        typedef typename api_traits::select_asymmetric_dual_callback<CELL_TYPE>::value value;
        dual_callback(other_grid, functor, value());
    }

    void set(std::size_t x, std::size_t y, std::size_t z, const CELL_TYPE& cell)
    {
        callback(detail::flat_array::set_instance_functor<CELL_TYPE>(&cell, x, y, z, 1));
    }

    void set(std::size_t x, std::size_t y, std::size_t z, const CELL_TYPE *cells, std::size_t count)
    {
        callback(detail::flat_array::set_instance_functor<CELL_TYPE>(cells, x, y, z, count));
    }

    CELL_TYPE get(std::size_t x, std::size_t y, std::size_t z) const
    {
        CELL_TYPE cell;
        callback(detail::flat_array::get_instance_functor<CELL_TYPE>(&cell, x, y, z, 1));

        return cell;
    }

    void get(std::size_t x, std::size_t y, std::size_t z, CELL_TYPE *cells, std::size_t count) const
    {
        callback(detail::flat_array::get_instance_functor<CELL_TYPE>(cells, x, y, z, count));
    }

    void load(std::size_t x, std::size_t y, std::size_t z, const char *data, std::size_t count)
    {
        callback(detail::flat_array::load_functor<CELL_TYPE>(x, y, z, data, count));
    }

    void save(std::size_t x, std::size_t y, std::size_t z, char *data, std::size_t count) const
    {
        callback(detail::flat_array::save_functor<CELL_TYPE>(x, y, z, data, count));
    }

    std::size_t byte_size() const
    {
        return my_byte_size;
    }

    char *get_data()
    {
        return data;
    }

    void set_data(char *new_data)
    {
        data = new_data;
    }

    std::size_t get_dim_x()
    {
        return dim_x;
    }

    std::size_t get_dim_y()
    {
        return dim_y;
    }

    std::size_t get_dim_z()
    {
        return dim_z;
    }

private:
    std::size_t dim_x;
    std::size_t dim_y;
    std::size_t dim_z;
    std::size_t my_byte_size;
    // We can't use std::vector here since the code needs to work with CUDA, too.
    char *data;

    /**
     * Adapt size of allocated memory to dim_[x-z]
     */
    void resize()
    {
        destroy();
        ALLOCATOR().deallocate(data, byte_size());

        // we need callback() to round up our grid size
        callback(detail::flat_array::set_byte_size_functor<CELL_TYPE>(&my_byte_size));
        data = ALLOCATOR().allocate(byte_size());
        init();
    }

    template<typename FUNCTOR>
    void dual_callback(soa_grid<CELL_TYPE> *other_grid, const FUNCTOR& functor, api_traits::true_type)
    {
        detail::flat_array::dual_callback_helper()(this, other_grid, functor);
    }

    template<typename FUNCTOR>
    void dual_callback(soa_grid<CELL_TYPE> *other_grid, const FUNCTOR& functor, api_traits::true_type) const
    {
        detail::flat_array::dual_callback_helper()(this, other_grid, functor);
    }

    template<typename FUNCTOR>
    void dual_callback(soa_grid<CELL_TYPE> *other_grid, FUNCTOR& functor, api_traits::false_type) const
    {
        assert_same_grid_sizes(other_grid);
        detail::flat_array::dual_callback_helper_symmetric<soa_grid<CELL_TYPE>, FUNCTOR> helper(
            other_grid, functor);

        api_traits::select_sizes<CELL_TYPE>()(
            data,
            helper,
            dim_x,
            dim_y,
            dim_z);
    }

    template<typename FUNCTOR>
    void dual_callback(soa_grid<CELL_TYPE> *other_grid, const FUNCTOR& functor, api_traits::false_type) const
    {
        assert_same_grid_sizes(other_grid);
        detail::flat_array::const_dual_callback_helper_symmetric<soa_grid<CELL_TYPE>, FUNCTOR> helper(
            other_grid, functor);

        api_traits::select_sizes<CELL_TYPE>()(
            data,
            helper,
            dim_x,
            dim_y,
            dim_z);
    }

    void init()
    {
        callback(detail::flat_array::construct_functor<CELL_TYPE>(dim_x, dim_y, dim_z));
    }

    void destroy()
    {
        if (data == 0) {
            return;
        }

        callback(detail::flat_array::destroy_functor<CELL_TYPE>(dim_x, dim_y, dim_z));
    }

    void copy_in(const soa_grid& other)
    {
        other.callback(this, detail::flat_array::copy_functor<CELL_TYPE>(dim_x, dim_y, dim_z));
    }

    void assert_same_grid_sizes(const soa_grid<CELL_TYPE> *other_grid) const
    {
        if ((dim_x != other_grid->dim_x) || (dim_y != other_grid->dim_y) || (dim_z != other_grid->dim_z)) {
            throw std::invalid_argument("grid dimensions of both grids must match");
        }
    }
};

}

namespace std
{
    template<typename CELL_TYPE>
    void swap(LibFlatArray::soa_grid<CELL_TYPE>& a, LibFlatArray::soa_grid<CELL_TYPE>& b)
    {
        a.swap(b);
    }
}

#endif
