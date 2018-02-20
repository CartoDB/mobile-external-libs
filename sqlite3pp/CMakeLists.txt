project(sqlite3pp)

set(libsqlite3pp_include_dirs
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set(libsqlite3pp_include_dirs ${libsqlite3pp_include_dirs} PARENT_SCOPE)

include_directories(
        ${libsqlite_include_dirs}
	${libsqlite3pp_include_dirs}
)

set(SOURCES ./sqlite3pp.cpp ./sqlite3ppext.cpp)

file(GLOB sqlite3pp_SRC_FILES ${SOURCES})

if(SINGLE_LIBRARY)
add_library(sqlite3pp INTERFACE)
set(sqlite3pp_SRC_FILES ${sqlite3pp_SRC_FILES} PARENT_SCOPE)
else()
add_library(sqlite3pp OBJECT ${sqlite3pp_SRC_FILES})
endif()
