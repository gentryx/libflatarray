find_package(OpenMP)

if(MSVC)
  add_subdirectory(jacobi)
  add_subdirectory(gauss)
endif()
add_subdirectory(lbm)
add_subdirectory(performance_tests)
add_subdirectory(smoothed_particle_hydrodynamics)
