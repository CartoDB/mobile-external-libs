project(bidi)

file(GLOB bidi_SRC_FILES "bidi.c" "ucdn.c")

set(bidi_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(bidi_SRC_FILES ${bidi_SRC_FILES} PARENT_SCOPE)
  add_library(bidi INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}")
  add_library(bidi OBJECT ${bidi_SRC_FILES})
endif()

