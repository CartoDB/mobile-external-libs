project(brotli)

file(GLOB brotli_SRC_FILES "brotli/c/common/*.c" "brotli/c/dec/*.c")

set(brotli_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/brotli/c/include" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(brotli_SRC_FILES ${brotli_SRC_FILES} PARENT_SCOPE)
  add_library(brotli INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}/brotli/c/include")
  add_library(brotli OBJECT ${brotli_SRC_FILES})
endif()
