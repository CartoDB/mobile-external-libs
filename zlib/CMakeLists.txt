project(zlib)

file(GLOB zlib_SRC_FILES "zlib/*.c")

set(zlib_FLAGS "")
if(WIN32)
  set(zlib_FLAGS ${zlib_FLAGS} "/wd4267")
endif()

set(zlib_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/config" "${PROJECT_SOURCE_DIR}/zlib" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(zlib_SRC_FILES ${zlib_SRC_FILES} PARENT_SCOPE)
  set(zlib_FLAGS ${zlib_FLAGS} PARENT_SCOPE)
  add_library(zlib INTERFACE)
else()
  add_compile_options(${zlib_FLAGS})
  include_directories("${PROJECT_SOURCE_DIR}/config" "${PROJECT_SOURCE_DIR}/zlib")
  add_library(zlib OBJECT ${zlib_SRC_FILES})
endif()
