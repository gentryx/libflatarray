if(WITH_CUDA)
  cuda_add_executable(lbm2 main.cu flatarray_implementation_0.cu flatarray_implementation_1.cu flatarray_implementation_2.cu flatarray_implementation_3.cu flatarray_implementation_4.cu flatarray_implementation_5.cu flatarray_implementation_6.cu flatarray_implementation_7.cu flatarray_implementation_8.cu)
  target_link_libraries(lbm2 ${Boost_LIBRARIES})
endif()
