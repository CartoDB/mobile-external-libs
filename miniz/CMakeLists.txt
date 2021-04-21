project(miniz)

file(GLOB miniz_SRC_FILES "miniz/miniz.c" "miniz/miniz_zip.c" "miniz/miniz_tinfl.c")

set(miniz_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/config" "${PROJECT_SOURCE_DIR}/miniz" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(miniz_SRC_FILES ${miniz_SRC_FILES} PARENT_SCOPE)
  add_library(miniz INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}/config" "${PROJECT_SOURCE_DIR}/miniz")
  add_library(miniz OBJECT ${miniz_SRC_FILES})
endif()
