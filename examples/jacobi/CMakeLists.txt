find_package(OpenMP)

add_executable(jacobi main.cpp update_c99.c)

if(OPENMP_FOUND)
  if(CMAKE_VERSION VERSION_GREATER 2.8.11)
    target_compile_options(jacobi PRIVATE "-fopenmp")
  endif()
  set_target_properties(jacobi PROPERTIES LINK_FLAGS "-fopenmp")
endif()
