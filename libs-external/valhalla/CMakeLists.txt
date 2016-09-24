project(valhalla)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBOOST_SPIRIT_THREADSAFE -DBOOST_NO_CXX11_SCOPED_ENUMS")

file(GLOB valhalla_SRC_FILES "source/*/*.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/odin/service.cc" "${PROJECT_SOURCE_DIR}/source/meili/service.cc" "${PROJECT_SOURCE_DIR}/source/meili/valhalla_map_match_service.cc" "${PROJECT_SOURCE_DIR}/source/thor/service.cc" "${PROJECT_SOURCE_DIR}/source/thor/attributes_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/isochrone_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/route_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/optimized_route_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/matrix_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/formlocalpath.cc" "${PROJECT_SOURCE_DIR}/source/thor/timedistancematrix.cc")

include_directories("${PROJECT_SOURCE_DIR}/include/valhalla" "${PROJECT_SOURCE_DIR}/../protobuf-2.5.0/src")

if(SINGLE_LIBRARY)
add_library(valhalla INTERFACE)
set(valhalla_SRC_FILES ${valhalla_SRC_FILES} PARENT_SCOPE)
else()
add_library(valhalla OBJECT ${valhalla_SRC_FILES})
endif()
