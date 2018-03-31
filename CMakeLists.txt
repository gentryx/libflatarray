cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
include(CheckCXXCompilerFlag)

if (CMAKE_VERSION VERSION_GREATER 2.7)
  cmake_policy(SET CMP0012 NEW)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

# 1. GENERIC PACKAGE INFORMATION
if(PACKAGE_NAME)
  set(GLOBAL_PACKAGE_NAME ${PACKAGE_NAME})
endif()

set(PACKAGE_NAME libflatarray)
set(PACKAGE_VERSION "0.3.0")
set(PACKAGE_VENDOR "Chair for Computer Science 3, FAU Erlangen, Germany")
set(PACKAGE_HOMEPAGE "http://www.libgeodecomp.org/libflatarray.html")
set(PACKAGE_EMAIL "users@libgeodecomp.org")
project(${PACKAGE_NAME})

set(DEFAULT_FLAGS "${DEFAULT_FLAGS} -Wall")

check_cxx_compiler_flag("-Wno-sign-conversion" SUPPORTS_WARN_NO_WSIGN)
if(SUPPORTS_WARN_NO_WSIGN)
  set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -Wno-sign-conversion")
endif()

check_cxx_compiler_flag("/wd4365" SUPPORTS_WARN_DISABLE_4365)
if(SUPPORTS_WARN_DISABLE_4365)
  set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /wd4365")
endif()

check_cxx_compiler_flag("-Wnon-virtual-dtor" SUPPORTS_WARN_NON_VIRTUAL_DTOR)
if(SUPPORTS_WARN_NON_WSIGN)
  set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -Wnon-virtual-dtor")
endif()

check_cxx_compiler_flag("-march=native" SUPPORTS_MARCH_NATIVE)
if(SUPPORTS_MARCH_NATIVE AND NOT APPLE)
  set(DEFAULT_FLAGS "${DEFAULT_FLAGS} -march=native")
endif()

find_package(CUDA)
# Force the selection of the NVCC host compiler to circumvent the
# studid FindCUDA CMake module from resolving symlinks (*wink* ccache
# *wink*).
if(CUDA_FOUND)
  get_filename_component(exe_name "${CUDA_HOST_COMPILER}" NAME)
  if(exe_name STREQUAL "ccache")
    set(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Override host side compiler for NVCC" FORCE)
  endif()
endif()

find_package(Silo)

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

if(MSVC)
  set(SUPPORTS_CPP14 true)
else()
  check_cxx_compiler_flag("-std=c++14" SUPPORTS_CPP14)
  if(NOT SUPPORTS_CPP14)
    set(SUPPORTS_CPP14 false)
  endif()

  check_cxx_compiler_flag("-std=c++11" SUPPORTS_CPP11)
  if(NOT SUPPORTS_CPP11)
    set(SUPPORTS_CPP11 false)
  endif()

  check_cxx_compiler_flag("-std=c++0x" SUPPORTS_CPP0X)
  if(NOT SUPPORTS_CPP0X)
    set(SUPPORTS_CPP0X false)
  endif()
endif()

lfa_add_config_option(LFA_ADDITIONAL_COMPILE_FLAGS "Add these flags when compiling." "${DEFAULT_FLAGS}" false)
lfa_add_config_option(UNITEXEC "May be used to specify a wrapper which then calls a unit test executable. Handy if for instance the unit tests shall be run inside of Intel's Software Development Emulator (SDE)." "" false)
lfa_add_config_option(WITH_CUDA "Enable modules which harness NVIDIA CUDA GPUs" ${CUDA_FOUND} true)
lfa_add_config_option(LIB_INSTALL_DIR "Directory where libraries will be installed to, may be overridden for 64-bit installations" "lib" false)
lfa_add_config_option(WITH_CPP14 "Enable C++ 14 features of LibFlatArray" ${SUPPORTS_CPP14} true)
lfa_add_config_option(WITH_FORCED_CPP11 "Some C++11-compatible compilers support sufficient features of C++14 to allow LibFlatArray to enable C++14 mode. Use this setting in conjunction with WITH_CPP14" false true)
lfa_add_config_option(WITH_INCREASED_PRECISION "Some architectures (*cough*ARM*cough*) yield poor accuracy when using fast estimates. This option enforces mitigates that flaw at the expense of a few extra cycles." false true)
lfa_add_config_option(WITH_SILO "Enable code examples that require LLNL's Silo library." ${Silo_FOUND} true)

if(WITH_CUDA AND NOT CUDA_FOUND)
  message(FATAL_ERROR "WITH_CUDA selected, but could not find the NVIDIA CUDA toolkit.")
endif()

if(NOT MSVC)
  if(WITH_CPP14)
    # At the moment, CUDA doesn't support C++14, so we limt ourselves to
    # C++11 and hope for the best.
    IF(WITH_FORCED_CPP11)
      set(libflatarray_FLAGS "-std=c++11")
    else()
      set(libflatarray_FLAGS "-std=c++14")
    endif()

  else()
    if(SUPPORTS_CPP11)
      set(libflatarray_FLAGS "-std=c++11")
    else()
      if(SUPPORTS_CPP0X)
        set(libflatarray_FLAGS "-std=c++0x")
      else()
        message(FATAL_ERROR "Apparently your C++ compiler supports neither C++11 nor a preliminary C++0x, but LibFlatArray requres extended initializer lists.")
      endif()
    endif()
  endif()
endif()

if(WIN32)
  set(libflatarray_LIBS "ws2_32")
else()
  set(libflatarray_LIBS "")
endif()

set(libflatarray_FLAGS "${libflatarray_FLAGS} ${LFA_ADDITIONAL_COMPILE_FLAGS}")

get_directory_property(lfa_has_parent PARENT_DIRECTORY)
if(lfa_has_parent)
  set(libflatarray_FLAGS "${libflatarray_FLAGS}" PARENT_SCOPE)
  set(libflatarray_LIBS "${libflatarray_LIBS}" PARENT_SCOPE)
endif()

# We need to back up previously set compiler flags so we can restore
# them at the end of this file. Purpose: avoid pollution of
# user-defined compiler flags. This is important since some compilers
# will complain if certain options are given multiple times (e.g. nvcc
# and -std=c++11).
set(LFA_C_FLAGS_BACKUP    "${CMAKE_C_FLAGS}")
set(LFA_CXX_FLAGS_BACKUP  "${CMAKE_CXX_FLAGS}")
set(LFA_NVCC_FLAGS_BACKUP "${CUDA_NVCC_FLAGS}")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -std=c99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${libflatarray_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${libflatarray_FLAGS}")

lfa_dump_config("config.h")
lfa_print_options()

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
set(libflatarray_INCLUDE_DIRS \"\${CMAKE_CURRENT_LIST_DIR}\" \"${CMAKE_CURRENT_LIST_DIR}/include\")
set(libflatarray_FLAGS \"${libflatarray_FLAGS}\")
set(libflatarray_LIBS \"${libflatarray_LIBS}\")
")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}/${PACKAGE_NAME}Config.cmake"
  "
set(libflatarray_INCLUDE_DIRS \"\${CMAKE_CURRENT_LIST_DIR}/../../../include\")
set(libflatarray_FLAGS \"${libflatarray_FLAGS}\")
set(libflatarray_LIBS \"${libflatarray_LIBS}\")
")

set(ConfigPackageLocation ${LIB_INSTALL_DIR}/cmake/${PACKAGE_NAME})
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}/${PACKAGE_NAME}Config.cmake"
  DESTINATION "${ConfigPackageLocation}")

file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/libflatarray/*.hpp")
foreach(i ${HEADERS})
  install(FILES ${i} DESTINATION include/${PACKAGE_NAME})
endforeach()

file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/libflatarray/detail/*.hpp")
foreach(i ${HEADERS})
  install(FILES ${i} DESTINATION include/${PACKAGE_NAME}/detail)
endforeach()

file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/libflatarray/testbed/*.hpp")
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
  if (${CUDA_VERSION} VERSION_LESS 8.0)
    set(CUDA_NVCC_FLAGS "--gpu-architecture=compute_20")
  else()
    set(CUDA_NVCC_FLAGS "--gpu-architecture=compute_35")
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR})

add_custom_target(check echo "Tests passed.")
add_custom_target(tests echo "All tests have been built.")

add_subdirectory(test)
add_subdirectory(examples)

if(GLOBAL_PACKAGE_NAME)
  set(PACKAGE_NAME ${GLOBAL_PACKAGE_NAME})
endif()

# restore compiler flags to original values:
set(CMAKE_C_FLAGS   "${LFA_C_FLAGS_BACKUP}")
set(CMAKE_CXX_FLAGS "${LFA_CXX_FLAGS_BACKUP}")
set(CUDA_NVCC_FLAGS "${LFA_NVCC_FLAGS_BACKUP}")
