project(valhalla)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

add_definitions("-DBOOST_SPIRIT_THREADSAFE -DBOOST_NO_CXX11_SCOPED_ENUMS -DRAPIDJSON_HAS_CXX11_RVALUE_REFS=1 -DRAPIDJSON_HAS_STDSTRING=1 -DRAPIDJSON_ENDIAN=RAPIDJSON_LITTLEENDIAN")

file(GLOB valhalla_SRC_FILES "source/*/*.cc" "source/*.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/meili/valhalla_run_map_match.cc" "${PROJECT_SOURCE_DIR}/source/meili/traffic_segment_matcher.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/thor/formlocalpath.cc" "${PROJECT_SOURCE_DIR}/source/thor/expandfromnode.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/tyr/actor.cc")
if(APPLE)
list(APPEND valhalla_SRC_FILES "source/date/ios.mm")
endif(APPLE)

include_directories(BEFORE "${PROJECT_SOURCE_DIR}/include/valhalla" "${PROJECT_SOURCE_DIR}/../protobuf/src" "${PROJECT_SOURCE_DIR}/../rapidjson/include")

if(SINGLE_LIBRARY)
add_library(valhalla INTERFACE)
set(valhalla_SRC_FILES ${valhalla_SRC_FILES} PARENT_SCOPE)
else()
add_library(valhalla OBJECT ${valhalla_SRC_FILES})
endif()
