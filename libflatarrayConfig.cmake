get_filename_component(LIBFLATARRAY_CMAKE_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
include_directories("${LIBFLATARRAY_CMAKE_DIR}/src")
message(info "LIBFLATARRAY at ${LIBFLATARRAY_CMAKE_DIR}/src")