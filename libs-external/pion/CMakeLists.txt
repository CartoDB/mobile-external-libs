project(pion)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

file(GLOB pion_SRC_FILES "src/*.cpp")

include_directories("${PROJECT_SOURCE_DIR}/include")

if(SINGLE_LIBRARY)
add_library(pion INTERFACE)
set(pion_SRC_FILES ${pion_SRC_FILES} PARENT_SCOPE)
else()
add_library(pion OBJECT ${pion_SRC_FILES})
endif()
