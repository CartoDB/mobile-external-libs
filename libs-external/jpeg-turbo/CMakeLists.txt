project(jpeg)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

file(GLOB jpeg_SRC_FILES "*.c")

include_directories("${PROJECT_SOURCE_DIR}")

if(SINGLE_LIBRARY)
add_library(jpeg INTERFACE)
set(jpeg_SRC_FILES ${jpeg_SRC_FILES} PARENT_SCOPE)
else()
add_library(jpeg OBJECT ${jpeg_SRC_FILES})
endif()
