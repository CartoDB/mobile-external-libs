project(androidcpufeatures)

file(GLOB androidcpufeatures_SRC_FILES "cpufeatures/*.c")

set(androidcpufeatures_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpufeatures" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(androidcpufeatures_SRC_FILES ${androidcpufeatures_SRC_FILES} PARENT_SCOPE)
  add_library(androidcpufeatures INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}/cpufeatures")
  add_library(androidcpufeatures OBJECT ${androidcpufeatures_SRC_FILES})
endif()
