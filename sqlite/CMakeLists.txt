project(sqlite)

file(GLOB sqlite_SRC_FILES "*.c")

set(sqlite_FLAGS "-DSQLITE_TEMP_STORE=2" "-DSQLITE_THREADSAFE=2" "-DSQLITE_OMIT_DEPRECATED" "-DSQLITE_OMIT_WAL" "-DSQLITE_OMIT_COMPILEOPTION_DIAGS" "-DSQLITE_OMIT_EXPLAIN" "-DSQLITE_OMIT_LOAD_EXTENSION" "-DSQLITE_OMIT_SHARED_CACHE" "-DSQLITE_OMIT_TRACE")
if(WIN32)
  set(sqlite_FLAGS ${sqlite_FLAGS} "-DSQLITE_OS_WINRT")
elseif(ANDROID OR IOS)
  set(sqlite_FLAGS ${sqlite_FLAGS} "-Wno-implicit-int-float-conversion")
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
