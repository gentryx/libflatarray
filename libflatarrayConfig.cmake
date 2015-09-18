# this dummy config is provided to ease integration when LibFlatArray
# is bundled into other CMake projects
get_filename_component(LIBFLATARRAY_CMAKE_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
set(libflatarray_INCLUDE_DIR "${LIBFLATARRAY_CMAKE_DIR}/src")
