find_package(OpenMP)

add_executable(jacobi main.cpp update_c99.c)

if(OPENMP_FOUND)
  target_compile_options(jacobi PRIVATE "-fopenmp")
  set_target_properties(jacobi PROPERTIES LINK_FLAGS "-fopenmp")
endif()
