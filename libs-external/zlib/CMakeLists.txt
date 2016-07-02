project(zlib)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

set(zlib_SRC_FILES
  adler32.c
  crc32.c
  deflate.c
  infback.c
  inffast.c
  inflate.c
  inftrees.c
  trees.c
  zutil.c
  compress.c
  uncompr.c
  gzclose.c
  gzlib.c
  gzread.c
  gzwrite.c
)

include_directories("${PROJECT_SOURCE_DIR}")

if(SINGLE_LIBRARY)
add_library(zlib INTERFACE)
set(zlib_SRC_FILES ${zlib_SRC_FILES} PARENT_SCOPE)
else()
add_library(zlib OBJECT ${zlib_SRC_FILES})
endif()
