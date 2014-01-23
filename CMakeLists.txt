cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

# 1. GENERIC PACKAGE INFORMATION
set(PACKAGE_NAME libflatarray)
set(PACKAGE_VERSION "0.1.0")
set(PACKAGE_VENDOR "Chair for Computer Science 3, FAU Erlangen, Germany")
set(PACKAGE_HOMEPAGE "http://www.libgeodecomp.org/libflatarray.html")
set(PACKAGE_EMAIL "users@libgeodecomp.org")
project(${PACKAGE_NAME})

# 2. CONFIGURE INSTALLER
set(CPACK_PACKAGE_NAME ${PACKAGE_NAME})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION ${PACKAGE_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${PACKAGE_NAME})

# will be shown e.g. in windows' control center package info
set(CPACK_PACKAGE_VENDOR ${PACKAGE_VENDOR})
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${PACKAGE_NAME})

include(CPack)

file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")
foreach(i ${HEADERS})
  install(FILES ${i} DESTINATION include/${PACKAGE_NAME})
endforeach()

# 3. BUILD DESCRIPTION
find_package(Boost REQUIRED date_time filesystem system)
find_package(CUDA)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(test)
add_subdirectory(examples)
