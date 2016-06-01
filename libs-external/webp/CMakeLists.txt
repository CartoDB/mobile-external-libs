project(webp)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

file(GLOB webp_SRC_FILES "src/*/*.c")

include_directories("${PROJECT_SOURCE_DIR}/include")

if(SINGLE_LIBRARY)
add_library(webp INTERFACE)
set(webp_SRC_FILES ${webp_SRC_FILES} PARENT_SCOPE)
else()
add_library(webp OBJECT ${webp_SRC_FILES})
endif()
