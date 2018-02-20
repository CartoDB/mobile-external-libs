project(harfbuzz)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

file(GLOB harfbuzz_SRC_FILES "src/hb*.cc")
list(REMOVE_ITEM harfbuzz_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/hb-uniscribe.cc" "${CMAKE_CURRENT_SOURCE_DIR}/src/hb-coretext.cc" "${CMAKE_CURRENT_SOURCE_DIR}/src/hb-directwrite.cc" "${CMAKE_CURRENT_SOURCE_DIR}/src/hb-icu.cc" "${CMAKE_CURRENT_SOURCE_DIR}/src/hb-glib.cc" "${CMAKE_CURRENT_SOURCE_DIR}/src/hb-gobject-enums.cc" "${CMAKE_CURRENT_SOURCE_DIR}/src/hb-gobject-structs.cc" "${CMAKE_CURRENT_SOURCE_DIR}/src/hb-graphite2.cc")

if(SINGLE_LIBRARY)
add_library(harfbuzz INTERFACE)
set(harfbuzz_SRC_FILES ${harfbuzz_SRC_FILES} PARENT_SCOPE)
else()
add_library(harfbuzz OBJECT ${harfbuzz_SRC_FILES})
endif()
