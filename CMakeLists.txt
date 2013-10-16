cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

find_package(Boost REQUIRED date_time filesystem system)
find_package(CUDA)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(test)
add_subdirectory(examples)
