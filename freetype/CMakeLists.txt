project(freetype)

file(GLOB freetype_SRC_FILES
  freetype/src/autofit/autofit.c
  freetype/src/base/ftbase.c
  freetype/src/base/ftdebug.c
  freetype/src/base/ftbbox.c
  freetype/src/base/ftbdf.c
  freetype/src/base/ftbitmap.c
  freetype/src/base/ftcid.c
  freetype/src/base/ftfstype.c
  freetype/src/base/ftgasp.c
  freetype/src/base/ftglyph.c
  freetype/src/base/ftgxval.c
  freetype/src/base/ftinit.c
  freetype/src/base/ftmm.c
  freetype/src/base/ftotval.c
  freetype/src/base/ftpatent.c
  freetype/src/base/ftpfr.c
  freetype/src/base/ftstroke.c
  freetype/src/base/ftsynth.c
  freetype/src/base/fttype1.c
  freetype/src/base/ftwinfnt.c
  freetype/src/base/ftsystem.c
  freetype/src/bdf/bdf.c
  freetype/src/bzip2/ftbzip2.c
  freetype/src/cache/ftcache.c
  freetype/src/cff/cff.c
  freetype/src/cid/type1cid.c
  freetype/src/gzip/ftgzip.c
  freetype/src/lzw/ftlzw.c
  freetype/src/pcf/pcf.c
  freetype/src/pfr/pfr.c
  freetype/src/psaux/psaux.c
  freetype/src/pshinter/pshinter.c
  freetype/src/psnames/psnames.c
  freetype/src/raster/raster.c
  freetype/src/sdf/sdf.c
  freetype/src/sfnt/sfnt.c
  freetype/src/smooth/smooth.c
  freetype/src/truetype/truetype.c
  freetype/src/type1/type1.c
  freetype/src/type42/type42.c
  freetype/src/winfonts/winfnt.c
)

set(freetype_FLAGS "-DFT2_BUILD_LIBRARY")
if(WIN32)
  set(freetype_FLAGS ${freetype_FLAGS} "/fp:fast" "/wd4244" "/wd4267" "/wd4312" "/wd4996")
elseif(ANDROID OR APPLE)
  set(freetype_FLAGS ${freetype_FLAGS} "-ffast-math" "-Wno-narrowing" "-Wno-shorten-64-to-32")
endif()

set(freetype_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/config" "${PROJECT_SOURCE_DIR}/freetype/include" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(freetype_SRC_FILES ${freetype_SRC_FILES} PARENT_SCOPE)
  set(freetype_FLAGS ${freetype_FLAGS} PARENT_SCOPE)
  add_library(freetype INTERFACE)
else()
  add_compile_options(${freetype_FLAGS})
  include_directories(freetype "${PROJECT_SOURCE_DIR}/config" "${PROJECT_SOURCE_DIR}/freetype/include" "${PROJECT_SOURCE_DIR}/../brotli/brotli/c/include" "${PROJECT_SOURCE_DIR}/../harfbuzz/harfbuzz/src")
  add_library(freetype OBJECT ${freetype_SRC_FILES})
endif()
