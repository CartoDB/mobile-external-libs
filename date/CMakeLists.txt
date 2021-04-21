project(date)

file(GLOB date_SRC_FILES "src/*.cpp")

set(date_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(date_SRC_FILES ${date_SRC_FILES} PARENT_SCOPE)
  add_library(date INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}/include")
  add_library(date OBJECT ${date_SRC_FILES})
endif()

