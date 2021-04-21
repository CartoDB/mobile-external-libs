project(sqlite3pp)

file(GLOB sqlite3pp_SRC_FILES "*.cpp")

set(sqlite3pp_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(sqlite3pp_SRC_FILES ${sqlite3pp_SRC_FILES} PARENT_SCOPE)
  add_library(sqlite3pp INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/../sqlite")
  add_library(sqlite3pp OBJECT ${sqlite3pp_SRC_FILES})
endif()
