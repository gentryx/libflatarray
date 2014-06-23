/**
 * Copyright 2014 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_SOA_GRID_HPP
#define FLAT_ARRAY_SOA_GRID_HPP

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

    explicit soa_grid(size_t dim_x = 0, size_t dim_y = 0, size_t dim_z = 0) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z),
        data(0)
    {
        resize();
    }

    soa_grid(const soa_grid& other) :
        dim_x(other.dim_x),
        dim_y(other.dim_y),
        dim_z(other.dim_z),
        my_byte_size(other.my_byte_size)
    {
        data = ALLOCATOR().allocate(byte_size());
        std::copy(other.data, other.data + byte_size(), data);
    }

    ~soa_grid()
    {
        ALLOCATOR().deallocate(data, byte_size());
    }

    soa_grid& operator=(const soa_grid& other)
    {
        ALLOCATOR().deallocate(data, byte_size());

        dim_x = other.dim_x;
        dim_y = other.dim_y;
        dim_z = other.dim_z;
        my_byte_size = other.my_byte_size;

        data = ALLOCATOR().allocate(byte_size());
        std::copy(other.data, other.data + byte_size(), data);

        return *this;
    }

    void swap(soa_grid& other)
    {
        std::swap(dim_x, other.dim_x);
        std::swap(dim_x, other.dim_x);
        std::swap(dim_x, other.dim_x);
        std::swap(my_byte_size, other.my_byte_size);
        std::swap(data, other.data);
    }

    void resize(size_t new_dim_x, size_t new_dim_y, size_t new_dim_z)
    {
        dim_x = new_dim_x;
        dim_y = new_dim_y;
        dim_z = new_dim_z;
        resize();
    }

    template<typename FUNCTOR>
    void callback(const FUNCTOR& functor, int index = 0) const
    {
        bind_parameters0(functor, index);
    }

    template<typename FUNCTOR>
    void callback(soa_grid<CELL_TYPE> *otherGrid, const FUNCTOR& functor) const
    {
        detail::flat_array::dual_callback_helper()(this, otherGrid, functor);
    }

    void set(size_t x, size_t y, size_t z, const CELL_TYPE& cell)
    {
        callback(detail::flat_array::set_instance_functor<CELL_TYPE>(&cell, x, y, z, 1), 0);
    }

    void set(size_t x, size_t y, size_t z, const CELL_TYPE *cells, size_t count)
    {
        callback(detail::flat_array::set_instance_functor<CELL_TYPE>(cells, x, y, z, count), 0);
    }

    CELL_TYPE get(size_t x, size_t y, size_t z) const
    {
        CELL_TYPE cell;
        callback(detail::flat_array::get_instance_functor<CELL_TYPE>(&cell, x, y, z, 1), 0);

        return cell;
    }

    void get(size_t x, size_t y, size_t z, CELL_TYPE *cells, size_t count) const
    {
        callback(detail::flat_array::get_instance_functor<CELL_TYPE>(cells, x, y, z, count), 0);
    }

    void load(size_t x, size_t y, size_t z, const char *data, size_t count)
    {
        callback(detail::flat_array::load_functor<CELL_TYPE>(x, y, z, data, count), 0);
    }

    void save(size_t x, size_t y, size_t z, char *data, size_t count) const
    {
        callback(detail::flat_array::save_functor<CELL_TYPE>(x, y, z, data, count), 0);
    }

    size_t byte_size() const
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

private:
    size_t dim_x;
    size_t dim_y;
    size_t dim_z;
    size_t my_byte_size;
    // We can't use std::vector here since the code needs to work with CUDA, too.
    char *data;

    /**
     * Adapt size of allocated memory to dim_[x-z]
     */
    void resize()
    {
        ALLOCATOR().deallocate(data, byte_size());

        // we need callback() to round up our grid size
        callback(detail::flat_array::set_byte_size_functor<CELL_TYPE>(&my_byte_size), 0);
        data = ALLOCATOR().allocate(byte_size());
    }

    template<int DIM_X, int DIM_Y, typename FUNCTOR>
    void bind_parameters2(const FUNCTOR& functor, int index) const
    {
        size_t size = dim_z;

#define CASE(SIZE)                                                      \
        if (size <= SIZE) {                                             \
            soa_accessor<CELL_TYPE, DIM_X, DIM_Y, SIZE, 0>  accessor(   \
                data, index);                                           \
            functor(accessor,                                           \
                    /* fixme */                                         \
                    &accessor.index);                                   \
            return;                                                     \
        }

        // CASE(  1);
        CASE( 32);
        // CASE( 64);
        // CASE( 96);
        // CASE(128);
        // CASE(160);
        // CASE(192);
        // CASE(224);
        CASE(256);
        // CASE(288);
        // CASE(384);
        // CASE(416);
        // CASE(512);
        // CASE(544);
        CASE(1026);
        throw std::out_of_range("grid dimension Z too large");

#undef CASE
    }

    template<int DIM_X, typename FUNCTOR>
    void bind_parameters1(const FUNCTOR& functor, int index) const
    {
        size_t size = dim_y;

#define CASE(SIZE)                                                      \
        if (size <= SIZE) {                                             \
            bind_parameters2<DIM_X, SIZE>(functor, index);              \
            return;                                                     \
        }

        // CASE(  1);
        CASE( 32);
        // CASE( 64);
        // CASE( 96);
        CASE(128);
        // CASE(160);
        CASE(192);
        // CASE(224);
        CASE(256);
        // CASE(288);
        CASE(512);
        CASE(544);
        CASE(1026);
        throw std::out_of_range("grid dimension Y too large");

#undef CASE
    }

    template<typename FUNCTOR>
    void bind_parameters0(const FUNCTOR& functor, int index) const
    {
        size_t size = dim_x;
        // fixme: this would be superfluous if we'd call bind_parameters1
        if (dim_y > size) {
            size = dim_y;
        }

#define CASE(SIZE)                                                      \
        if (size <= SIZE) {                                             \
            /* fixme: */                                                \
            bind_parameters2<SIZE, SIZE>(functor, index);               \
            /* bind_parameters1<SIZE>(functor, index); */               \
            return;                                                     \
        }

        CASE( 32);
        // CASE( 64);
        // CASE( 96);
        CASE(128);
        // CASE(160);
        CASE(192);
        // CASE(224);
        CASE(256);
        // CASE(288);
        // CASE(384);
        // CASE(416);
        CASE(512);
        CASE(544);
        CASE(1056);
        throw std::out_of_range("grid dimension X too large");

#undef CASE
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
