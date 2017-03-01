add_executable(gauss main.cpp filter_c99.c)
target_link_libraries(gauss ${libflatarray_LIBS})

if(OPENMP_FOUND)
  if(CMAKE_VERSION VERSION_GREATER 2.8.11)
    target_compile_options(gauss PRIVATE ${OpenMP_CXX_FLAGS})
  endif()
  set_target_properties(gauss PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
endif()
