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


#endif
