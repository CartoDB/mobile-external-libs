project(png)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

file(GLOB png_SRC_FILES "*.c")

include_directories("${PROJECT_SOURCE_DIR}")

if(SINGLE_LIBRARY)
add_library(png INTERFACE)
set(png_SRC_FILES ${png_SRC_FILES} PARENT_SCOPE)
else()
add_library(png OBJECT ${png_SRC_FILES})
endif()
