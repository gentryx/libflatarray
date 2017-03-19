/**
 * Copyright 2016-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifdef _MSC_BUILD
#  define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <libflatarray/flat_array.hpp>
#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>

extern "C" {
    void filter_c99(double *data_new, const double *data_old, int dim_x, int dim_y, int dim_z);
}

class benchmark_c_style_gauss : public LibFlatArray::cpu_benchmark
{
public:
    std::string family()
    {
        return "Gauss";
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
        filter_c99(grid_new.data(), grid_old.data(), dim[0], dim[1], dim[2]);
        double sum2 = sum(grid_new, dim[0], dim[1], dim[2]);

        double delta = std::fabs(sum1 - sum2);
        if (delta > 0.1) {
            throw std::logic_error("consistency check failed");
        }

        double time_start = time();

        for (int t = 0; t < max_steps; ++t) {
            filter_c99(grid_new.data(), grid_old.data(), dim[0], dim[1], dim[2]);
            using std::swap;
            swap(grid_old, grid_new);
        }

        double time_end = time();
        double time_total = time_end - time_start;

        double active_cells = (dim[0] - 0) * (dim[1] - 4) * (dim[2] - 4);
        double lattice_updates = active_cells * max_steps;
        double glups = lattice_updates * 1e-9 / time_total;
        return glups;
    }

private:
    void init(std::vector<double> *grid, std::size_t dim_x, std::size_t dim_y, std::size_t dim_z)
    {
        for (std::size_t z = 4; z < (dim_z - 4); ++z) {
            for (std::size_t y = 4; y < (dim_y - 4); ++y) {
                for (std::size_t x = 0; x < (dim_x - 0); ++x) {
                    std::size_t index = z * dim_x * dim_y + y * dim_x + x;
                    (*grid)[index] = 1;
                }
            }
        }
    }

    double sum(const std::vector<double> grid, std::size_t dim_x, std::size_t dim_y, std::size_t dim_z)
    {
        double sum = 0;

        for (std::size_t z = 0; z < (dim_z - 0); ++z) {
            for (std::size_t y = 0; y < (dim_y - 0); ++y) {
                for (std::size_t x = 0; x < (dim_x - 0); ++x) {
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

    friend class benchmark_libflatarray_gauss;

    inline
    explicit Cell(const double temp = 0) :
        temp(temp)
    {}

private:
    double temp;
};

LIBFLATARRAY_REGISTER_SOA(Cell, ((double)(temp)))

class filter_functor
{
public:
    filter_functor(std::size_t dim_x, std::size_t dim_y, std::size_t dim_z) :
        dim_x(dim_x),
        dim_y(dim_y),
        dim_z(dim_z)
    {}

    template<typename SHORT_VEC, typename SOA_ACCESSOR_1, typename SOA_ACCESSOR_2>
    void filter(
        std::size_t& x,
        std::size_t end_x,
        std::size_t y,
        std::size_t z,
        SOA_ACCESSOR_1& accessor_old,
        SOA_ACCESSOR_2& accessor_new,
        const SHORT_VEC weight_00,
        const SHORT_VEC weight_01,
        const SHORT_VEC weight_02,
        const SHORT_VEC weight_11,
        const SHORT_VEC weight_12,
        const SHORT_VEC weight_22) const
    {
        accessor_old.index() = SOA_ACCESSOR_1::gen_index(x, y, z);
        accessor_new.index() = SOA_ACCESSOR_2::gen_index(x, y, z);

        SHORT_VEC buf;

        for (; x < end_x; x += SHORT_VEC::ARITY) {
            using LibFlatArray::coord;
            buf =  SHORT_VEC(&accessor_old[coord< 0, -2, -2>()].temp()) * weight_22;
            buf += SHORT_VEC(&accessor_old[coord< 0, -1, -2>()].temp()) * weight_12;
            buf += SHORT_VEC(&accessor_old[coord< 0,  0, -2>()].temp()) * weight_02;
            buf += SHORT_VEC(&accessor_old[coord< 0,  1, -2>()].temp()) * weight_12;
            buf += SHORT_VEC(&accessor_old[coord< 0,  2, -2>()].temp()) * weight_22;

            buf += SHORT_VEC(&accessor_old[coord< 0, -2, -1>()].temp()) * weight_12;
            buf += SHORT_VEC(&accessor_old[coord< 0, -1, -1>()].temp()) * weight_11;
            buf += SHORT_VEC(&accessor_old[coord< 0,  0, -1>()].temp()) * weight_01;
            buf += SHORT_VEC(&accessor_old[coord< 0,  1, -1>()].temp()) * weight_11;
            buf += SHORT_VEC(&accessor_old[coord< 0,  2, -1>()].temp()) * weight_12;

            buf += SHORT_VEC(&accessor_old[coord< 0, -2,  0>()].temp()) * weight_02;
            buf += SHORT_VEC(&accessor_old[coord< 0, -1,  0>()].temp()) * weight_01;
            buf += SHORT_VEC(&accessor_old[coord< 0,  0,  0>()].temp()) * weight_00;
            buf += SHORT_VEC(&accessor_old[coord< 0,  1,  0>()].temp()) * weight_01;
            buf += SHORT_VEC(&accessor_old[coord< 0,  2,  0>()].temp()) * weight_02;

            buf += SHORT_VEC(&accessor_old[coord< 0, -2,  1>()].temp()) * weight_12;
            buf += SHORT_VEC(&accessor_old[coord< 0, -1,  1>()].temp()) * weight_11;
            buf += SHORT_VEC(&accessor_old[coord< 0,  0,  1>()].temp()) * weight_01;
            buf += SHORT_VEC(&accessor_old[coord< 0,  1,  1>()].temp()) * weight_11;
            buf += SHORT_VEC(&accessor_old[coord< 0,  2,  1>()].temp()) * weight_12;

            buf += SHORT_VEC(&accessor_old[coord< 0, -2,  2>()].temp()) * weight_22;
            buf += SHORT_VEC(&accessor_old[coord< 0, -1,  2>()].temp()) * weight_12;
            buf += SHORT_VEC(&accessor_old[coord< 0,  0,  2>()].temp()) * weight_02;
            buf += SHORT_VEC(&accessor_old[coord< 0,  1,  2>()].temp()) * weight_12;
            buf += SHORT_VEC(&accessor_old[coord< 0,  2,  2>()].temp()) * weight_22;

            &accessor_new.temp() << buf;

            accessor_new += SHORT_VEC::ARITY;
            accessor_old += SHORT_VEC::ARITY;
        }

    }

    template<typename SOA_ACCESSOR_1, typename SOA_ACCESSOR_2>
    void operator()(SOA_ACCESSOR_1 accessor_old, SOA_ACCESSOR_2 accessor_new) const
    {
        typedef typename LibFlatArray::estimate_optimum_short_vec_type<double, SOA_ACCESSOR_1>::VALUE my_short_vec;

        double weights[5][5];
        double sum = 0;

        for (int y = 0; y < 5; ++y) {
            for (int x = 0; x < 5; ++x) {
                double x_component = x - 2;
                double y_component = y - 2;
                weights[y][x] = exp(-0.5 * (x_component * x_component +
                                            y_component * y_component)) / 2 / M_PI;
                sum += weights[y][x];
            }
        }
        for (int y = 0; y < 5; ++y) {
            for (int x = 0; x < 5; ++x) {
                weights[y][x] /= sum;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) firstprivate(accessor_old, accessor_new)
#endif
        for (int z = 2; z < int(dim_z - 2); ++z) {
            for (std::size_t y = 2; y < (dim_y - 2); ++y) {
                std::size_t x = 0;
                std::size_t end_x = dim_x - 0;

                LIBFLATARRAY_LOOP_PEELER_TEMPLATE(
                    my_short_vec, std::size_t, x, end_x, filter, y, std::size_t(z), accessor_old, accessor_new,
                    weights[2][2], weights[2][1], weights[2][0], weights[1][1], weights[1][0], weights[0][0]);
            }
        }
    }

private:
    std::size_t dim_x;
    std::size_t dim_y;
    std::size_t dim_z;
};

class benchmark_libflatarray_gauss : public LibFlatArray::cpu_benchmark
{
public:
    std::string family()
    {
        return "Gauss";
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
        filter_functor functor(dim[0], dim[1], dim[2]);
        grid_old.callback(&grid_new, functor);
        double sum2 = sum(grid_new);

        double delta = std::fabs(sum1 - sum2);
        if (delta > 0.1) {
            double delta = sum2 - sum1;
            std::cout << "sum1: " << sum1 << "\n"
                      << "sum2: " << sum2 << "\n"
                      << "dlta: " << delta << "\n";
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
        for (std::size_t z = 4; z < (grid->dim_z() - 4); ++z) {
            for (std::size_t y = 4; y < (grid->dim_y() - 4); ++y) {
                for (std::size_t x = 0; x < (grid->dim_x() - 0); ++x) {
                    grid->set(x, y, z, Cell(1));
                }
            }
        }
    }

    double sum(const LibFlatArray::soa_grid<Cell>& grid)
    {
        double sum = 0;

        for (std::size_t z = 0; z < grid.dim_z(); ++z) {
            for (std::size_t y = 0; y < grid.dim_y(); ++y) {
                for (std::size_t x = 0; x < grid.dim_x(); ++x) {
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

    eval(benchmark_c_style_gauss(),      dim, true);
    eval(benchmark_libflatarray_gauss(), dim, true);

    return 0;
}
