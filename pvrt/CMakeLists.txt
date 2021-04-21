project(pvrt)

file(GLOB pvrt_SRC_FILES "*.cpp")

set(pvrt_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(pvrt_SRC_FILES ${pvrt_SRC_FILES} PARENT_SCOPE)
  add_library(pvrt INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}")
  add_library(pvrt OBJECT ${pvrt_SRC_FILES})
endif()
