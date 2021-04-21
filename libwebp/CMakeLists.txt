project(webp)

file(GLOB webp_SRC_FILES "webp/src/*/*.c")

set(webp_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/libwebp" "${PROJECT_SOURCE_DIR}/libwebp/src" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(webp_SRC_FILES ${webp_SRC_FILES} PARENT_SCOPE)
  add_library(webp INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}/libwebp")
  add_library(webp OBJECT ${webp_SRC_FILES})
endif()
