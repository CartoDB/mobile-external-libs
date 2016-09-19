project(valhalla)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBOOST_SPIRIT_THREADSAFE -DBOOST_NO_CXX11_SCOPED_ENUMS")

file(GLOB valhalla_SRC_FILES "source/*/*.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/odin/service.cc")

include_directories("${PROJECT_SOURCE_DIR}/include/valhalla" "${PROJECT_SOURCE_DIR}/../protobuf-2.5.0/src")

if(SINGLE_LIBRARY)
add_library(valhalla INTERFACE)
set(valhalla_SRC_FILES ${valhalla_SRC_FILES} PARENT_SCOPE)
else()
add_library(valhalla OBJECT ${valhalla_SRC_FILES})
endif()
