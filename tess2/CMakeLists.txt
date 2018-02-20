project(libtess2)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

file(GLOB libtess2_SRC_FILES "source/*.c")

include_directories("${PROJECT_SOURCE_DIR}/include")

if(SINGLE_LIBRARY)
add_library(libtess2 INTERFACE)
set(libtess2_SRC_FILES ${libtess2_SRC_FILES} PARENT_SCOPE)
else()
add_library(libtess2 OBJECT ${libtess2_SRC_FILES})
endif()
