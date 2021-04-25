project(libwebp)

file(GLOB libwebp_SRC_FILES "libwebp/src/*/*.c")

set(libwebp_FLAGS "-DWEBP_EXTERN=extern" "-DWEBP_DISABLE_STATS")

set(libwebp_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/libwebp" "${PROJECT_SOURCE_DIR}/libwebp/src" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(libwebp_SRC_FILES ${libwebp_SRC_FILES} PARENT_SCOPE)
  set(libwebp_FLAGS ${libwebp_FLAGS} PARENT_SCOPE)
  add_library(libwebp INTERFACE)
else()
  add_compile_options(${libwebp_FLAGS})
  include_directories("${PROJECT_SOURCE_DIR}/libwebp")
  add_library(libwebp OBJECT ${libwebp_SRC_FILES})
endif()
