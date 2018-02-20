project(sqlite)

set(libsqlite_include_dirs
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set(libsqlite_include_dirs ${libsqlite_include_dirs} PARENT_SCOPE)

set(SOURCES ./sqlite3.h ./sqlite3.c)

file(GLOB sqlite_SRC_FILES ${SOURCES})

if(SINGLE_LIBRARY)
add_library(sqlite INTERFACE)
set(sqlite_SRC_FILES ${sqlite_SRC_FILES} PARENT_SCOPE)
else()
add_library(sqlite OBJECT ${SOURCES})
endif()
