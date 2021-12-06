project(harfbuzz)

file(GLOB harfbuzz_SRC_FILES "harfbuzz/src/hb*.cc")

set(harfbuzz_FLAGS "-DHAVE_FREETYPE" "-DHB_NO_MT" "-DHB_DISABLE_DEPRECATED" "-DHB_NDEBUG" "-DHB_NO_ATEXIT" "-DHB_NO_BUFFER_MESSAGE" "-DHB_NO_BUFFER_SERIALIZE" "-DHB_NO_BITMAP" "-DHB_NO_CFF" "-DHB_NO_COLOR" "-DHB_NO_DRAW" "-DHB_NO_ERRNO" "-DHB_NO_FACE_COLLECT_UNICODES" "-DHB_NO_GETENV" "-DHB_NO_LAYOUT_UNUSED" "-DHB_NO_MATH" "-DHB_NO_MMAP" "-DHB_NO_OPEN" "-DHB_NO_SETLOCALE" "-DHB_NO_AAT" "-DHB_NO_LEGACY")
if(WIN32)
  set(harfbuzz_FLAGS ${harfbuzz_FLAGS} "/wd4065" "/wd4244" "/wd4267" "/wd4334" "/wd4996")
endif()

set(harfbuzz_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/harfbuzz/src" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(harfbuzz_SRC_FILES ${harfbuzz_SRC_FILES} PARENT_SCOPE)
  set(harfbuzz_FLAGS ${harfbuzz_FLAGS} PARENT_SCOPE)
  add_library(harfbuzz INTERFACE)
else()
  add_compile_options(${harfbuzz_FLAGS})
  include_directories("${PROJECT_SOURCE_DIR}/harfbuzz/src" "${PROJECT_SOURCE_DIR}/../freetype/config" "${PROJECT_SOURCE_DIR}/../freetype/freetype/include")
  add_library(harfbuzz OBJECT ${harfbuzz_SRC_FILES})
endif()
