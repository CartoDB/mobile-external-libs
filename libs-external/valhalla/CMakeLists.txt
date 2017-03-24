project(valhalla)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

add_definitions("-DBOOST_SPIRIT_THREADSAFE -DBOOST_NO_CXX11_SCOPED_ENUMS -DRAPIDJSON_HAS_CXX11_RVALUE_REFS=1 -DRAPIDJSON_HAS_STDSTRING=1 -DRAPIDJSON_ENDIAN=RAPIDJSON_LITTLEENDIAN")

file(GLOB valhalla_SRC_FILES "source/*/*.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/baldr/graphfsreader.cc" "${PROJECT_SOURCE_DIR}/source/baldr/graphtilefsstorage.cc" "${PROJECT_SOURCE_DIR}/source/baldr/shared_tiles.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/odin/service.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/meili/service.cc" "${PROJECT_SOURCE_DIR}/source/meili/valhalla_meili_worker.cc" "${PROJECT_SOURCE_DIR}/source/meili/valhalla_run_map_match.cc" "${PROJECT_SOURCE_DIR}/source/meili/valhalla_map_match_service.cc" "${PROJECT_SOURCE_DIR}/source/meili/traffic_segment_matcher.cc" "${PROJECT_SOURCE_DIR}/source/meili/map_matcher_factory.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/thor/service.cc" "${PROJECT_SOURCE_DIR}/source/thor/attributes_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/isochrone_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/route_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/optimized_route_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/trace_route_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/matrix_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/formlocalpath.cc" "${PROJECT_SOURCE_DIR}/source/thor/timedistancematrix.cc" "${PROJECT_SOURCE_DIR}/source/thor/trace_attributes_action.cc" "${PROJECT_SOURCE_DIR}/source/thor/expandfromnode.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/loki/service.cc" "${PROJECT_SOURCE_DIR}/source/loki/attributes_action.cc" "${PROJECT_SOURCE_DIR}/source/loki/isochrone_action.cc" "${PROJECT_SOURCE_DIR}/source/loki/locate_action.cc" "${PROJECT_SOURCE_DIR}/source/loki/route_action.cc" "${PROJECT_SOURCE_DIR}/source/loki/trace_route_action.cc" "${PROJECT_SOURCE_DIR}/source/loki/matrix_action.cc")

file(GLOB boost_datetime_SRC_FILES "${PROJECT_SOURCE_DIR}/../boost/libs/date_time/src/gregorian/*.cpp")
list(APPEND valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/../boost/libs/system/src/error_code.cpp" "${boost_datetime_SRC_FILES}")

include_directories(BEFORE "${PROJECT_SOURCE_DIR}/include/valhalla" "${PROJECT_SOURCE_DIR}/../protobuf/src" "${PROJECT_SOURCE_DIR}/../rapidjson/include")

if(SINGLE_LIBRARY)
add_library(valhalla INTERFACE)
set(valhalla_SRC_FILES ${valhalla_SRC_FILES} PARENT_SCOPE)
else()
add_library(valhalla OBJECT ${valhalla_SRC_FILES})
endif()
