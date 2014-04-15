cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

# 1. GENERIC PACKAGE INFORMATION
set(PACKAGE_NAME libflatarray)
set(PACKAGE_VERSION "0.1.1")
set(PACKAGE_VENDOR "Chair for Computer Science 3, FAU Erlangen, Germany")
set(PACKAGE_HOMEPAGE "http://www.libgeodecomp.org/libflatarray.html")
set(PACKAGE_EMAIL "users@libgeodecomp.org")
project(${PACKAGE_NAME})

find_package(Boost REQUIRED date_time filesystem system)
find_package(CUDA)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" SUPPORTS_MARCH_NATIVE)
if(SUPPORTS_MARCH_NATIVE AND NOT APPLE)
  set(DEFAULT_FLAGS "${DEFAULT_FLAGS} -march=native")
endif()

# pretty print build options
function(print_options)
  message("-- The following options have been configured:")
  message(${OPTIONS_LIST})
  message("")
endfunction(print_options)

# dumps selected build options to a config header
function(dump_config outfile)
  set(CONTENT "#ifndef LIBFLATARRAY_CONFIG_H\n\n")
  set(CONTENT "${CONTENT}${CONFIG_HEADER}\n")
  set(CONTENT "${CONTENT}#endif\n")
  file(WRITE "${outfile}.new" "${CONTENT}")

  execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${CMAKE_CURRENT_SOURCE_DIR}/${outfile}" "${CMAKE_CURRENT_SOURCE_DIR}/${outfile}.new" RESULT_VARIABLE res)
  if(res GREATER 0)
    file(WRITE "${outfile}" "${CONTENT}")
  endif()

  file(REMOVE "${outfile}.new")
endfunction(dump_config)

# generic function to add user-configurable options. add_to_header may be used to propagate the option to a header file.
function(add_config_option name help_message default add_to_header)
  if(NOT DEFINED ${name})
    set(${name} "${default}")
    set(${name} "${default}" CACHE STRING "${help_message}" FORCE)
  endif()

  set(OPTIONS_LIST "${OPTIONS_LIST}\n\n * ${name}=\"${${name}}\",\n   default=\"${default}\"\n   ${help_message}" PARENT_SCOPE)

  if(add_to_header)
    if(${name})
      set(CONFIG_HEADER "${CONFIG_HEADER}#define LIBFLATARRAY_${name} ${${name}}\n" PARENT_SCOPE)
    endif()
  endif()
endfunction(add_config_option)

add_config_option(ADDITIONAL_COMPILE_FLAGS "Add these flags when compiling." "${DEFAULT_FLAGS}" false)
add_config_option(WITH_CUDA "Enable modules which harness Nvidia CUDA GPUs" ${CUDA_FOUND} true)

if(WITH_CUDA AND NOT CUDA_FOUND)
  message(FATAL_ERROR "WITH_CUDA selected, but could not find the NVIDIA CUDA toolkit.")
endif()

dump_config("src/config.h")
print_options()

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${ADDITIONAL_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADDITIONAL_COMPILE_FLAGS}")

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

file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/*.hpp")
foreach(i ${HEADERS})
  install(FILES ${i} DESTINATION include/${PACKAGE_NAME}/detail)
endforeach()

file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/testbed/*.hpp")
foreach(i ${HEADERS})
  install(FILES ${i} DESTINATION include/${PACKAGE_NAME}/testbed)
endforeach()

# 3. BUILD DESCRIPTION
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(WITH_CUDA)
  set(CUDA_NVCC_FLAGS "--gpu-architecture=sm_20")
endif()

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_custom_target(test echo "Tests passed.")

add_subdirectory(test)
add_subdirectory(examples)
