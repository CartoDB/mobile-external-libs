project(rg_etc1)

file(GLOB rg_etc1_SRC_FILES "*.cpp")

set(rg_etc1_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(rg_etc1_SRC_FILES ${rg_etc1_SRC_FILES} PARENT_SCOPE)
  add_library(rg_etc1 INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}")
  add_library(rg_etc1 OBJECT ${rg_etc1_SRC_FILES})
endif()
