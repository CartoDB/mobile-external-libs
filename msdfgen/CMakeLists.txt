project(msdfgen)

file(GLOB msdfgen_SRC_FILES "core/*.cpp")

set(msdfgen_FLAGS "")
if(WIN32)
  set(msdfgen_FLAGS ${msdfgen_FLAGS} "/wd4267")
endif()

set(msdfgen_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(msdfgen_SRC_FILES ${msdfgen_SRC_FILES} PARENT_SCOPE)
  set(msdfgen_FLAGS ${msdfgen_FLAGS} PARENT_SCOPE)
  add_library(msdfgen INTERFACE)
else()
  add_compile_options(${msdfgen_FLAGS})
  include_directories("${PROJECT_SOURCE_DIR}")
  add_library(msdfgen OBJECT ${msdfgen_SRC_FILES})
endif()
