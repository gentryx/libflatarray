add_executable(aligned_allocator_test aligned_allocator_test.cpp)
add_executable(soa_array_test soa_array_test.cpp)
add_executable(soa_grid_test soa_grid_test.cpp)
cuda_add_executable(cuda_allocator_test cuda_allocator_test.cu)

target_link_libraries(aligned_allocator_test ${Boost_LIBRARIES})
target_link_libraries(soa_array_test ${Boost_LIBRARIES})
target_link_libraries(soa_grid_test ${Boost_LIBRARIES})
target_link_libraries(cuda_allocator_test ${Boost_LIBRARIES})
