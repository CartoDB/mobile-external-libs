project(png)

file(GLOB png_SRC_FILES
  libpng/png.c libpng/pngerror.c
  libpng/pngget.c libpng/pngset.c libpng/pngmem.c libpng/pngtrans.c libpng/pngpread.c
  libpng/pngread.c libpng/pngrio.c libpng/pngrtran.c libpng/pngrutil.c
  libpng/pngwrite.c libpng/pngwio.c libpng/pngwtran.c libpng/pngwutil.c
)

set(png_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/libpng" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(png_SRC_FILES ${png_SRC_FILES} PARENT_SCOPE)
  add_library(png INTERFACE)
else()
  include_directories("${PROJECT_SOURCE_DIR}/libpng")
  add_library(png OBJECT ${png_SRC_FILES})
endif()
