project(rg_etc1)

set(SOURCES ./rg_etc1.h ./rg_etc1.cpp)

set(rg_etc1_include_dirs
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set(rg_etc1_include_dirs ${libpugixml_include_dirs} PARENT_SCOPE)

file(GLOB rg_etc1_SRC_FILES ${SOURCES})

if(SINGLE_LIBRARY)
add_library(rg_etc1 INTERFACE)
set(rg_etc1_SRC_FILES ${rg_etc1_SRC_FILES} PARENT_SCOPE)
else()
add_library(rg_etc1 OBJECT ${SOURCES})
endif()
