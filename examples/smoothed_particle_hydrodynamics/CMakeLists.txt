find_package(Silo)

if(Silo_FOUND)
  add_executable(sph main.c)
  target_link_libraries(sph ${Silo_LIBRARY})
endif()
