project(date)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

file(GLOB date_SRC_FILES "src/*.cpp")

include_directories("${PROJECT_SOURCE_DIR}/include")

if(SINGLE_LIBRARY)
add_library(date INTERFACE)
set(date_SRC_FILES ${date_SRC_FILES} PARENT_SCOPE)
else()
add_library(date OBJECT ${date_SRC_FILES})
endif()

