cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

find_package(Boost REQUIRED date_time filesystem system)
find_package(CUDA)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(jacobi example_jacobi.cpp)
target_link_libraries(jacobi ${Boost_LIBRARIES})

add_executable(interface_test example_interface_test.cpp)
target_link_libraries(interface_test ${Boost_LIBRARIES})

SET(CUDA_NVCC_FLAGS "-arch=compute_20")
cuda_add_executable(lbm example_lbm.cu)
target_link_libraries(lbm ${Boost_LIBRARIES})
