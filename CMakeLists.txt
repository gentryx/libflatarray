cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

# 1. GENERIC PACKAGE INFORMATION
if(PACKAGE_NAME)
  set(GLOBAL_PACKAGE_NAME ${PACKAGE_NAME})
endif()

set(PACKAGE_NAME libflatarray)
set(PACKAGE_VERSION "0.2.0")
set(PACKAGE_VENDOR "Chair for Computer Science 3, FAU Erlangen, Germany")
set(PACKAGE_HOMEPAGE "http://www.libgeodecomp.org/libflatarray.html")
set(PACKAGE_EMAIL "users@libgeodecomp.org")
project(${PACKAGE_NAME})

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" SUPPORTS_MARCH_NATIVE)
if(SUPPORTS_MARCH_NATIVE AND NOT APPLE)
  set(DEFAULT_FLAGS "${DEFAULT_FLAGS} -march=native")
endif()

if(NOT WITH_SIMPLE_BOOST)
  find_package(Boost REQUIRED date_time filesystem system)
endif()
find_package(CUDA)

# pretty print build options
function(lfa_print_options)
  message("-- The following options have been configured:")
  message(${OPTIONS_LIST})
  message("")
endfunction(lfa_print_options)

# dumps selected build options to a config header
function(lfa_dump_config outfile)
  set(CONTENT "#ifndef LIBFLATARRAY_CONFIG_H\n\n")
  set(CONTENT "${CONTENT}${CONFIG_HEADER}\n")
  set(CONTENT "${CONTENT}#endif\n")
  set(PATHNAME "${CMAKE_BINARY_DIR}/${PACKAGE_NAME}/${outfile}")
  file(WRITE "${PATHNAME}.new" "${CONTENT}")

  execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${PATHNAME}" "${PATHNAME}.new" RESULT_VARIABLE res)
  if(res GREATER 0)
    file(WRITE "${PATHNAME}" "${CONTENT}")
  endif()

  file(REMOVE "${PATHNAME}.new")
endfunction(lfa_dump_config)

# generic function to add user-configurable options. add_to_header may be used to propagate the option to a header file.
function(lfa_add_config_option name help_message default add_to_header)
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
endfunction(lfa_add_config_option)

lfa_add_config_option(ADDITIONAL_COMPILE_FLAGS "Add these flags when compiling." "${DEFAULT_FLAGS}" false)
lfa_add_config_option(UNITEXEC "May be used to specify a wrapper which then calls a unit test executable. Handy if for instance the unit tests shall be run inside of Intel's Software Development Emulator (SDE)." "" false)
lfa_add_config_option(WITH_CUDA "Enable modules which harness Nvidia CUDA GPUs" ${CUDA_FOUND} true)
lfa_add_config_option(WITH_SIMPLE_BOOST "fixme: temporary workaround for weird boost installs, resolutin will be to bundle header-only boost libs" false false)
lfa_add_config_option(LIB_INSTALL_DIR "Directory where libraries will be installed to, may be overridden for 64-bit installations" "lib" false)
lfa_add_config_option(WITH_CPP14 "Enable C++ 14 features of LibFlatArray" false true)

if(WITH_CUDA AND NOT CUDA_FOUND)
  message(FATAL_ERROR "WITH_CUDA selected, but could not find the NVIDIA CUDA toolkit.")
endif()

if(WITH_CPP14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

lfa_dump_config("config.h")
lfa_print_options()

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

set(ConfigPackageLocation ${LIB_INSTALL_DIR}/cmake/${PACKAGE_NAME})

if(CMAKE_VERSION VERSION_GREATER 2.8.10)
  include(CMakePackageConfigHelpers)

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake"
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion)

  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake"
    DESTINATION "${ConfigPackageLocation}")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake"
  "
get_filename_component(LIBFLATARRAY_CMAKE_DIR \${CMAKE_CURRENT_LIST_FILE} PATH)
set(libflatarray_INCLUDE_DIR \"\${LIBFLATARRAY_CMAKE_DIR}/../../../include\")
")

set(ConfigPackageLocation ${LIB_INSTALL_DIR}/cmake/${PACKAGE_NAME})
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake"
  DESTINATION "${ConfigPackageLocation}")

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

install(
  FILES "${CMAKE_BINARY_DIR}/${PACKAGE_NAME}/config.h"
    DESTINATION include/${PACKAGE_NAME})

# 3. BUILD DESCRIPTION
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(WITH_CUDA)
  set(CUDA_NVCC_FLAGS "--gpu-architecture=sm_20")
endif()

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR})

add_custom_target(check echo "Tests passed.")

add_subdirectory(test)
add_subdirectory(examples)

if(GLOBAL_PACKAGE_NAME)
  set(PACKAGE_NAME ${GLOBAL_PACKAGE_NAME})
endif()

