project(valhalla)

file(GLOB valhalla_SRC_FILES "source/*/*.cc" "source/*.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/meili/valhalla_run_map_match.cc" "${PROJECT_SOURCE_DIR}/source/meili/traffic_segment_matcher.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/thor/formlocalpath.cc" "${PROJECT_SOURCE_DIR}/source/thor/expandfromnode.cc")
list(REMOVE_ITEM valhalla_SRC_FILES "${PROJECT_SOURCE_DIR}/source/tyr/actor.cc")

set(valhalla_FLAGS "-DBOOST_SPIRIT_THREADSAFE" "-DBOOST_NO_CXX11_SCOPED_ENUMS" "-DRAPIDJSON_HAS_CXX11_RVALUE_REFS=1" "-DRAPIDJSON_HAS_STDSTRING=1" "-DRAPIDJSON_ENDIAN=RAPIDJSON_LITTLEENDIAN")
if(WIN32)
  set(valhalla_FLAGS "${valhalla_FLAGS}" "/wd4018" "/wd4101" "/wd4267" "/wd4305" "/wd4804" "/wd4805")
endif()

set(valhalla_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/include/valhalla" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(valhalla_SRC_FILES ${valhalla_SRC_FILES} PARENT_SCOPE)
  set(valhalla_FLAGS ${valhalla_FLAGS} PARENT_SCOPE)
  add_library(valhalla INTERFACE)
else()
  add_compile_options(${valhalla_FLAGS})
  include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/include/valhalla" "${PROJECT_SOURCE_DIR}/../protobuf/src" "${PROJECT_SOURCE_DIR}/../rapidjson/include" "${PROJECT_SOURCE_DIR}/../date/include" "${PROJECT_SOURCE_DIR}/../sqlite" "${PROJECT_SOURCE_DIR}/../sqlite3pp")
  add_library(valhalla OBJECT ${valhalla_SRC_FILES})
endif()
