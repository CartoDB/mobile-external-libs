project(pion)

file(GLOB pion_SRC_FILES "src/*.cpp")

set(pion_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(pion_SRC_FILES ${pion_SRC_FILES} PARENT_SCOPE)
  add_library(pion INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}/include")
  add_library(pion OBJECT ${pion_SRC_FILES})
endif()
