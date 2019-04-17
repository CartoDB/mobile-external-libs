project(msdfgen)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

if(ANDROID OR IOS)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-strict-aliasing")
endif()

file(GLOB msdfgen_SRC_FILES "core/*.cpp")

include_directories("${PROJECT_SOURCE_DIR}")

if(SINGLE_LIBRARY)
add_library(msdfgen INTERFACE)
set(msdfgen_SRC_FILES ${msdfgen_SRC_FILES} PARENT_SCOPE)
else()
add_library(msdfgen OBJECT ${msdfgen_SRC_FILES})
endif()
