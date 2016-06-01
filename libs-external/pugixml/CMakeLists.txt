project(pugixml)

set(libpugixml_include_dirs
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(libpugixml_include_dirs ${libpugixml_include_dirs} PARENT_SCOPE)

set(SOURCES ./src/pugixml.hpp ./src/pugiconfig.hpp ./src/pugixml.cpp)

file(GLOB pugixml_SRC_FILES ${SOURCES})

if(SINGLE_LIBRARY)
add_library(pugixml INTERFACE)
set(pugixml_SRC_FILES ${pugixml_SRC_FILES} PARENT_SCOPE)
else()
add_library(pugixml OBJECT ${SOURCES})
endif()
