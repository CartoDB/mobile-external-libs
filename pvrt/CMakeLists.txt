project(pvrt)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

file(GLOB pvrt_SRC_FILES "*.cpp")

include_directories("${PROJECT_SOURCE_DIR}/include")

if(SINGLE_LIBRARY)
add_library(pvrt INTERFACE)
set(pvrt_SRC_FILES ${pvrt_SRC_FILES} PARENT_SCOPE)
else()
add_library(pvrt OBJECT ${pvrt_SRC_FILES})
endif()
