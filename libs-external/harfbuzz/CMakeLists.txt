project(harfbuzz)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

file(GLOB harfbuzz_SRC_FILES "*.cc")

if(SINGLE_LIBRARY)
add_library(harfbuzz INTERFACE)
set(harfbuzz_SRC_FILES ${harfbuzz_SRC_FILES} PARENT_SCOPE)
else()
add_library(harfbuzz OBJECT ${harfbuzz_SRC_FILES})
endif()
