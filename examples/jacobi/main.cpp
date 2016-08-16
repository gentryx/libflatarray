/**
 * Copyright 2012-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <libflatarray/flat_array.hpp>
#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>

extern "C" {
    void update_c99(double *data_new, const double *data_old, int dim_x, int dim_y, int dim_z);
}

class benchmark_c_style_3d_jacobi : public LibFlatArray::cpu_benchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "C99";
    }

    std::string unit()
    {
        return "GLUPS";
    }

    double performance(std::vector<int> dim)
    {
        std::size_t num_cells = std::size_t(dim[0]) * dim[1] * dim[2];
        int max_steps = dim[3];

        std::vector<double> grid_old(num_cells, 0);
        std::vector<double> grid_new(num_cells, 0);
        init(&grid_old, dim[0], dim[1], dim[2]);

        double sum1 = sum(grid_old, dim[0], dim[1], dim[2]);
        update_c99(grid_new.data(), grid_old.data(), dim[0], dim[1], dim[2]);
        double sum2 = sum(grid_new, dim[0], dim[1], dim[2]);

        double delta = std::fabs(sum1 - sum2);
        if (delta > 0.1) {
            throw std::logic_error("consistency check failed");
        }

        double time_start = time();

        for (int t = 0; t < max_steps; ++t) {
            update_c99(grid_new.data(), grid_old.data(), dim[0], dim[1], dim[2]);
            using std::swap;
            swap(grid_old, grid_new);
        }

        double time_end = time();
        double time_total = time_end - time_start;

        double lattice_updates = num_cells * max_steps;
        double glups = lattice_updates * 1e-9 / time_total;
        return glups;
    }

private:
    void init(std::vector<double> *grid, std::size_t dim_x, std::size_t dim_y, std::size_t dim_z)
    {
        for (std::size_t z = 2; z < (dim_z - 2); ++z) {
            for (std::size_t y = 2; y < (dim_y - 2); ++y) {
                for (std::size_t x = 2; x < (dim_x - 2); ++x) {
                    std::size_t index = z * dim_x * dim_y + y * dim_x + x;
                    (*grid)[index] = 1;
                }
            }
        }
    }

    double sum(const std::vector<double> grid, std::size_t dim_x, std::size_t dim_y, std::size_t dim_z)
    {
        double sum = 0;

        for (std::size_t z = 1; z < (dim_z - 1); ++z) {
            for (std::size_t y = 1; y < (dim_y - 1); ++y) {
                for (std::size_t x = 1; x < (dim_x - 1); ++x) {
                    std::size_t index = z * dim_x * dim_y + y * dim_x + x;
                    sum += grid[index];
                }
            }
        }

        return sum;
    }
};

class Cell
{
public:
    /**
     * This friend declaration is required to give utility classes,
     * e.g. operator<<(CELL_TYPE, LibFlatArray::soa_accessor), access
     * to our private members.
     */
    LIBFLATARRAY_ACCESS

    friend class benchmark_libflatarray_3d_jacobi;

    inline
    explicit Cell(const double temp = 0) :
        temp(temp)
    {}

private:
    double temp;
};

LIBFLATARRAY_REGISTER_SOA(Cell, ((double)(temp)))

class update_functor
{
public:
    update_functor(std::size_t dim_x, std::size_t dim_y, std::size_t dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<typename SHORT_VEC, typename SOA_ACCESSOR_1, typename SOA_ACCESSOR_2>
    void update(
        std::size_t& x,
        std::size_t end_x,
        std::size_t y,
        std::size_t z,
        SOA_ACCESSOR_1& accessor_old,
        SOA_ACCESSOR_2& accessor_new) const
    {
        accessor_old.index = SOA_ACCESSOR_1::gen_index(x, y, z);
        accessor_new.index = SOA_ACCESSOR_2::gen_index(x, y, z);

        SHORT_VEC buf;
        SHORT_VEC factor = 1.0 / 6.0;

        for (; x < (end_x - SHORT_VEC::ARITY + 1); x += SHORT_VEC::ARITY) {
            using LibFlatArray::coord;
            buf =  &accessor_old[coord< 0,  0, -1>()].temp();
            buf += &accessor_old[coord< 0, -1,  0>()].temp();
            buf += &accessor_old[coord<-1,  0,  0>()].temp();
            buf += &accessor_old[coord< 1,  0,  0>()].temp();
            buf += &accessor_old[coord< 0,  1,  0>()].temp();
            buf += &accessor_old[coord< 0,  0,  1>()].temp();
            buf *= factor;

            &accessor_new.temp() << buf;

            accessor_new += SHORT_VEC::ARITY;
            accessor_old += SHORT_VEC::ARITY;
        }

    }

    template<typename SOA_ACCESSOR_1, typename SOA_ACCESSOR_2>
    void operator()(SOA_ACCESSOR_1 accessor_old, SOA_ACCESSOR_2 accessor_new) const
    {
        typedef typename LibFlatArray::estimate_optimum_short_vec_type<double, SOA_ACCESSOR_1>::VALUE my_short_vec;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) firstprivate(accessor_old, accessor_new)
#endif
        for (std::size_t z = 1; z < (dim_z - 1); ++z) {
            for (std::size_t y = 1; y < (dim_y - 1); ++y) {
                std::size_t x = 1;
                std::size_t end_x = dim_x - 1;

                LIBFLATARRAY_LOOP_PEELER_TEMPLATE(my_short_vec, std::size_t, x, end_x, update, y, z, accessor_old, accessor_new);
            }
        }
    }

private:
    std::size_t dim_x;
    std::size_t dim_y;
    std::size_t dim_z;
};

class benchmark_libflatarray_3d_jacobi : public LibFlatArray::cpu_benchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "LFA";
    }

    std::string unit()
    {
        return "GLUPS";
    }

    double performance(std::vector<int> dim)
    {
        std::size_t num_cells = std::size_t(dim[0]) * dim[1] * dim[2];
        int max_steps = dim[3];

        LibFlatArray::soa_grid<Cell> grid_old(dim[0], dim[1], dim[2]);
        LibFlatArray::soa_grid<Cell> grid_new(dim[0], dim[1], dim[2]);

        init(&grid_old);

        double sum1 = sum(grid_old);
        update_functor functor(dim[0], dim[1], dim[2]);
        grid_old.callback(&grid_new, functor);
        double sum2 = sum(grid_new);

        double delta = std::fabs(sum1 - sum2);
        if (delta > 0.1) {
            throw std::logic_error("consistency check failed");
        }
        double time_start = time();

        for (int t = 0; t < max_steps; ++t) {
            grid_old.callback(&grid_new, functor);
            using std::swap;
            swap(grid_old, grid_new);
        }

        double time_end = time();
        double time_total = time_end - time_start;

        double lattice_updates = num_cells * max_steps;
        double glups = lattice_updates * 1e-9 / time_total;
        return glups;
    }

private:
    void init(LibFlatArray::soa_grid<Cell> *grid)
    {
        for (std::size_t z = 2; z < (grid->get_dim_z() - 2); ++z) {
            for (std::size_t y = 2; y < (grid->get_dim_y() - 2); ++y) {
                for (std::size_t x = 2; x < (grid->get_dim_x() - 2); ++x) {
                    grid->set(x, y, z, Cell(1));
                }
            }
        }
    }

    double sum(const LibFlatArray::soa_grid<Cell>& grid)
    {
        double sum = 0;

        for (std::size_t z = 0; z < grid.get_dim_z(); ++z) {
            for (std::size_t y = 0; y < grid.get_dim_y(); ++y) {
                for (std::size_t x = 0; x < grid.get_dim_x(); ++x) {
                    sum += grid.get(x, y, z).temp;
                }
            }
        }

        return sum;
    }
};

void print_usage(const std::string& program_name)
{
    std::cerr << "USAGE: " << program_name << " [DIM_X] [DIM_Y] [DIM_Z] [STEPS]" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc > 5) {
        print_usage(argv[0]);
        return 1;
    }

    std::stringstream buf;
    for (int i = 1; i < argc; ++i) {
        buf << argv[i] << " ";
    }

    int dim_x = 512;
    int dim_y = 512;
    int dim_z = 512;
    int steps = 10;

    if (argc >= 2) {
        buf >> dim_x;
    }
    if (argc >= 3) {
        buf >> dim_y;
    }
    if (argc >= 4) {
        buf >> dim_z;
    }
    if (argc >= 5) {
        buf >> steps;
    }

    if ((dim_x * dim_y * dim_z * steps) == 0) {
        print_usage(argv[0]);
        return 2;
    }

    std::vector<int> dim;
    dim.push_back(dim_x);
    dim.push_back(dim_y);
    dim.push_back(dim_z);
    dim.push_back(steps);

    LibFlatArray::evaluate eval("", "none");
    eval.print_header();

    eval(benchmark_c_style_3d_jacobi(),      dim, true);
    eval(benchmark_libflatarray_3d_jacobi(), dim, true);

    return 0;
}
