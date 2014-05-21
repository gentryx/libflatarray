#ifndef TEST_H
#define TEST_H

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
        // std::cout << "Runner<" << typeid(TEST).name() << ">()\n";
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


// really lazy (read: bad, inexact) test for equality. we can't use
// stict equality (operator==()), as vector units may yield
// non-IEEE-compliannt results.
#define TEST_REAL(A, B)                                 \
    {                                                   \
        double a = (A);                                 \
        double b = (B);                                 \
        double delta = a - b;                           \
        delta = delta < 0? -delta : delta;              \
        if (delta > 0.0001) {                           \
            throw std::logic_error(                     \
                "difference exceeds tolerance");        \
        }                                               \
    }

#endif
