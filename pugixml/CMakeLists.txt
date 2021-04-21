project(pugixml)

file(GLOB pugixml_SRC_FILES "pugixml/src/*.cpp")

set(pugixml_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/pugixml/src" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(pugixml_SRC_FILES ${pugixml_SRC_FILES} PARENT_SCOPE)
  add_library(pugixml INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}/pugixml/src")
  add_library(pugixml OBJECT ${pugixml_SRC_FILES})
endif()
