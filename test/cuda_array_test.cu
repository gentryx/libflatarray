#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/cudaarray.h>
#include <libgeodecomp/misc/cudautil.h>

#include <cuda.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CUDAArrayTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        std::vector<double> hostVec1(30, -1);
        std::vector<double> hostVec2(30, -2);
        std::vector<double> hostVec3(30, -3);
        std::vector<double> hostVec4(30, -4);

        for (int i = 0; i < 30; ++i) {
            hostVec1[i] = i + 0.5;
            TS_ASSERT_EQUALS(-2, hostVec2[i]);
            TS_ASSERT_EQUALS(-3, hostVec3[i]);
            TS_ASSERT_EQUALS(-4, hostVec4[i]);
        }

        CUDAArray<double> deviceArray1(&hostVec1[0], 30);
        CUDAArray<double> deviceArray2(hostVec1);
        CUDAArray<double> deviceArray3(deviceArray1);
        CUDAArray<double> deviceArray4;
        deviceArray4 = CUDAArray<double>(30);
        deviceArray4.load(&hostVec1[0]);

        deviceArray2.save(&hostVec2[0]);
        deviceArray3.save(&hostVec3[0]);
        deviceArray4.save(&hostVec4[0]);

        for (int i = 0; i < 30; ++i) {
            double expected = i + 0.5;
            TS_ASSERT_EQUALS(expected, hostVec2[i]);
            TS_ASSERT_EQUALS(expected, hostVec3[i]);
            TS_ASSERT_EQUALS(expected, hostVec4[i]);
        }

        TS_ASSERT_DIFFERS(deviceArray1.data(), deviceArray2.data());
        TS_ASSERT_DIFFERS(deviceArray1.data(), deviceArray3.data());
        TS_ASSERT_DIFFERS(deviceArray1.data(), deviceArray3.data());

        TS_ASSERT_DIFFERS(deviceArray2.data(), deviceArray3.data());
        TS_ASSERT_DIFFERS(deviceArray2.data(), deviceArray4.data());

        TS_ASSERT_DIFFERS(deviceArray3.data(), deviceArray4.data());

        cudaDeviceSynchronize();
        CUDAUtil::checkForError();
#endif
    }

    void testInitialization()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        int value = 4711;
        CUDAArray<int> deviceArray(3, value);

        std::vector<int> hostVec(3);
        deviceArray.save(&hostVec[0]);

        TS_ASSERT_EQUALS(hostVec[0], 4711);
        TS_ASSERT_EQUALS(hostVec[1], 4711);
        TS_ASSERT_EQUALS(hostVec[2], 4711);
#endif
    }

    void testResizeAfterAssignment()
    {
        CUDAArray<double> deviceArray1(20, 12.34);
        CUDAArray<double> deviceArray2(30, 666);
        CUDAArray<double> deviceArray3(25, 31);

        std::vector<double> hostVec(30);
        deviceArray2.save(hostVec.data());

        for (int i = 0; i < 30; ++i) {
            TS_ASSERT_EQUALS(hostVec[i], 666);
        }

        deviceArray1 = deviceArray2;
        deviceArray2 = deviceArray3;

        TS_ASSERT_EQUALS(deviceArray1.size(),     30);
        TS_ASSERT_EQUALS(deviceArray1.capacity(), 30);

        TS_ASSERT_EQUALS(deviceArray2.size(),     25);
        TS_ASSERT_EQUALS(deviceArray2.capacity(), 30);

        hostVec = std::vector<double>(30, -1);
        deviceArray1.save(hostVec.data());

        for (int i = 0; i < 30; ++i) {
            TS_ASSERT_EQUALS(hostVec[i], 666);
        }

        deviceArray2.save(hostVec.data());

        for (int i = 0; i < 25; ++i) {
            TS_ASSERT_EQUALS(hostVec[i], 31);
        }

        TS_ASSERT_DIFFERS(deviceArray1.data(), deviceArray2.data());
        TS_ASSERT_DIFFERS(deviceArray1.data(), deviceArray3.data());
        TS_ASSERT_DIFFERS(deviceArray2.data(), deviceArray3.data());
    }

    void testResize()
    {
        CUDAArray<double> deviceArray(200, 1.3);
        TS_ASSERT_EQUALS(200, deviceArray.size());
        TS_ASSERT_EQUALS(200, deviceArray.capacity());

        deviceArray.resize(150);
        TS_ASSERT_EQUALS(150, deviceArray.size());
        TS_ASSERT_EQUALS(200, deviceArray.capacity());

        {
            std::vector<double> hostVec(250, 10);
            deviceArray.save(hostVec.data());

            for (int i = 0; i < 150; ++i) {
                TS_ASSERT_EQUALS(hostVec[i], 1.3);
            }
            for (int i = 150; i < 250; ++i) {
                TS_ASSERT_EQUALS(hostVec[i], 10);
            }
        }

        deviceArray.resize(250, 27);
        TS_ASSERT_EQUALS(250, deviceArray.size());
        TS_ASSERT_EQUALS(250, deviceArray.capacity());

        {
            std::vector<double> hostVec(250, -1);
            deviceArray.save(hostVec.data());

            for (int i = 0; i < 150; ++i) {
                TS_ASSERT_EQUALS(hostVec[i], 1.3);
            }
            for (int i = 150; i < 250; ++i) {
                TS_ASSERT_EQUALS(hostVec[i], 27);
            }
        }

        // ensure content is kept intact if shrunk and enlarged
        // afterwards sans default initialization:
        deviceArray.resize(10);
        deviceArray.resize(210);
        TS_ASSERT_EQUALS(210, deviceArray.size());
        TS_ASSERT_EQUALS(250, deviceArray.capacity());

        {
            std::vector<double> hostVec(250, -1);
            deviceArray.save(hostVec.data());

            for (int i = 0; i < 150; ++i) {
                TS_ASSERT_EQUALS(hostVec[i], 1.3);
            }
            for (int i = 150; i < 210; ++i) {
                TS_ASSERT_EQUALS(hostVec[i], 27);
            }
        }
    }

    void testResize2()
    {
        CUDAArray<double> array(10, 5);
        array.resize(20, 6);
        array.resize(15, 7);
        double v = 8.0;
        array.resize(20, v);

        std::vector<double> vec(20);
        array.save(vec.data());

        for (int i = 0; i < 10; ++i) {
            TS_ASSERT_EQUALS(vec[i], 5);
        }
        for (int i = 10; i < 15; ++i) {
            TS_ASSERT_EQUALS(vec[i], 6);
        }
        for (int i = 15; i < 20; ++i) {
            TS_ASSERT_EQUALS(vec[i], 8);
        }
    }

    void testReserve()
    {
        CUDAArray<double> deviceArray(31, 1.3);
        TS_ASSERT_EQUALS(31, deviceArray.size());
        TS_ASSERT_EQUALS(31, deviceArray.capacity());

        deviceArray.reserve(55);
        TS_ASSERT_EQUALS(31, deviceArray.size());
        TS_ASSERT_EQUALS(55, deviceArray.capacity());

        std::vector<double> hostVec(31, -1);
        deviceArray.save(hostVec.data());

        for (int i = 0; i < 31; ++i) {
            TS_ASSERT_EQUALS(hostVec[i], 1.3)
        }
    }

};

}
