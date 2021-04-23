project(sqlite)

file(GLOB sqlite_SRC_FILES "*.c")

set(sqlite_FLAGS "")
if(WIN32)
  set(sqlite_FLAGS ${sqlite_FLAGS} "-DSQLITE_OS_WINRT" "-DSQLITE_OMIT_WAL")
endif()

set(sqlite_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(sqlite_SRC_FILES ${sqlite_SRC_FILES} PARENT_SCOPE)
  set(sqlite_FLAGS ${sqlite_FLAGS} PARENT_SCOPE)
  add_library(sqlite INTERFACE)
else()
  add_compile_options(${sqlite_FLAGS})
  include_directories("${PROJECT_SOURCE_DIR}")
  add_library(sqlite OBJECT ${sqlite_SRC_FILES})
endif()
