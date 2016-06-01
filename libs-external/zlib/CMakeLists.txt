project(zlib)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

file(GLOB zlib_SRC_FILES "*.c")

include_directories("${PROJECT_SOURCE_DIR}")

if(SINGLE_LIBRARY)
add_library(zlib INTERFACE)
set(zlib_SRC_FILES ${zlib_SRC_FILES} PARENT_SCOPE)
else()
add_library(zlib OBJECT ${zlib_SRC_FILES})
endif()
