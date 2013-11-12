add_executable(soa_array_test soa_array_test.cpp)
add_executable(soa_grid_test soa_grid_test.cpp)
target_link_libraries(soa_grid_test ${Boost_LIBRARIES})
