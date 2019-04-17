project(bidi)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

if(ANDROID OR IOS)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-strict-aliasing")
endif()

file(GLOB bidi_SRC_FILES "bidi.c" "ucdn.c")

if(SINGLE_LIBRARY)
add_library(bidi INTERFACE)
set(bidi_SRC_FILES ${bidi_SRC_FILES} PARENT_SCOPE)
else()
add_library(bidi OBJECT ${bidi_SRC_FILES})
endif()

