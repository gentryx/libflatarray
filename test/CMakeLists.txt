add_executable(aligned_allocator_test aligned_allocator_test.cpp)
add_executable(soa_array_test soa_array_test.cpp)
add_executable(soa_grid_test soa_grid_test.cpp)

add_custom_target(run_aligned_allocator_test echo "running aligned_allocator_test" && ./aligned_allocator_test)
add_custom_target(run_soa_array_test echo "running soa_array_test" && ./soa_array_test)
add_custom_target(run_soa_grid_test echo "running soa_grid_test" && ./soa_grid_test)

add_dependencies(test run_aligned_allocator_test)
add_dependencies(test run_soa_array_test)
add_dependencies(test run_soa_grid_test)

add_dependencies(run_aligned_allocator_test aligned_allocator_test)
add_dependencies(run_soa_array_test soa_array_test)
add_dependencies(run_soa_grid_test soa_grid_test)

if(WITH_CUDA)
  cuda_add_executable(cuda_allocator_test cuda_allocator_test.cu)
endif()

target_link_libraries(aligned_allocator_test ${Boost_LIBRARIES})
target_link_libraries(soa_array_test ${Boost_LIBRARIES})
target_link_libraries(soa_grid_test ${Boost_LIBRARIES})

if(WITH_CUDA)
  target_link_libraries(cuda_allocator_test ${Boost_LIBRARIES})
endif()
