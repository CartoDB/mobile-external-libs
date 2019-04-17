project(miniz)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

if(ANDROID OR IOS)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-strict-aliasing")
endif()

file(GLOB miniz_SRC_FILES "miniz.c")

if(SINGLE_LIBRARY)
add_library(miniz INTERFACE)
set(miniz_SRC_FILES ${miniz_SRC_FILES} PARENT_SCOPE)
else()
add_library(miniz OBJECT ${miniz_SRC_FILES})
endif()
