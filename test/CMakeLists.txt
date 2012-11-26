cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

find_package(Boost REQUIRED date_time filesystem system)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(jacobi example_jacobi.cpp)
target_link_libraries(jacobi ${Boost_LIBRARIES})
