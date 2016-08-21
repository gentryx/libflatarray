find_package(Silo)

message(INFO "andi: ${Silo_FOUND}, ${Silo_LIBRARY}")
add_executable(sph main.c)
target_link_libraries(sph ${Silo_LIBRARY})
