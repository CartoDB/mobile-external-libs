project(tess2)

file(GLOB tess2_SRC_FILES "source/*.c")

set(tess2_FLAGS "")
if(WIN32)
  set(tess2_FLAGS "${tess2_FLAGS}" "/wd4267")
endif()

set(tess2_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(tess2_SRC_FILES ${tess2_SRC_FILES} PARENT_SCOPE)
  set(tess2_FLAGS ${tess2_FLAGS} PARENT_SCOPE)
  add_library(tess2 INTERFACE)
else()
  add_compile_options(${tess2_FLAGS})
  include_directories("${PROJECT_SOURCE_DIR}/include")
  add_library(tess2 OBJECT ${tess2_SRC_FILES})
endif()
