#ifndef TEST_H
#define TEST_H

#include <cmath>
#include <sstream>

// Runner and ADD_TEST are some convenience functions to simplify
// definition of new tests. ADD_TEST will add scaffolding that causes
// the following block to be executed once the program starts.
// Advantage: tests have no longer to be manually added to main().
template<typename TEST>
class Runner
{
public:
    Runner()
    {
        TEST()();
    }
};

#define ADD_TEST(TEST_NAME)                     \
    class TEST_NAME                             \
    {                                           \
    public:                                     \
        void operator()();                      \
                                                \
    private:                                    \
        static Runner<TEST_NAME> runner;        \
    };                                          \
                                                \
    Runner<TEST_NAME> TEST_NAME::runner;        \
                                                \
    void TEST_NAME::operator()()                \


#define TEST_REAL_ACCURACY(A, B, RELATIVE_ERROR_LIMIT)                  \
    {                                                                   \
        double a = (A);                                                 \
        double b = (B);                                                 \
        double delta = std::abs(a - b);                                 \
        double relativeError = delta / std::abs(a);                     \
        if (relativeError > RELATIVE_ERROR_LIMIT) {                     \
            std::stringstream buf;                                      \
            buf << "in file "                                           \
                << __FILE__ << ":"                                      \
                << __LINE__ << ": "                                     \
                << "difference exceeds tolerance.\n"                    \
                << "   A: " << a << "\n"                                \
                << "   B: " << b << "\n"                                \
                << "   delta: " << delta << "\n"                        \
                << "   relativeError: " << relativeError << "\n";       \
            throw std::logic_error(buf.str());                          \
        }                                                               \
    }

// lazy (read: bad, inexact) test for equality. we can't use stict
// equality (operator==()), as vector units may yield
// non-IEEE-compliannt results. Single-precision accuracy (i.e. ~20
// bits for the mantissa or 6 digits) shall be suffice for functional
// testing.
#define TEST_REAL(A, B)                                                 \
    TEST_REAL_ACCURACY(A, B, 0.000001)

#endif
