project(libpng)

file(GLOB libpng_SRC_FILES
  libpng/png.c libpng/pngerror.c
  libpng/pngget.c libpng/pngset.c libpng/pngmem.c libpng/pngtrans.c libpng/pngpread.c
  libpng/pngread.c libpng/pngrio.c libpng/pngrtran.c libpng/pngrutil.c
  libpng/pngwrite.c libpng/pngwio.c libpng/pngwtran.c libpng/pngwutil.c
)

set(libpng_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/config" "${PROJECT_SOURCE_DIR}/libpng" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(libpng_SRC_FILES ${libpng_SRC_FILES} PARENT_SCOPE)
  add_library(libpng INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}/config" "${PROJECT_SOURCE_DIR}/libpng")
  add_library(libpng OBJECT ${libpng_SRC_FILES})
endif()
