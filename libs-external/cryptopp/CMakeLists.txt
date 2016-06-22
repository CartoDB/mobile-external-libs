project(cryptopp)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

add_definitions(-DCRYPTOPP_DISABLE_ASM -DCRYPTOPP_DISABLE_SSE2)

file(GLOB cryptopp_SRC_FILES "*.cpp")

include_directories("${PROJECT_SOURCE_DIR}")

if(SINGLE_LIBRARY)
add_library(cryptopp INTERFACE)
set(cryptopp_SRC_FILES ${cryptopp_SRC_FILES} PARENT_SCOPE)
else()
add_library(cryptopp OBJECT ${cryptopp_SRC_FILES})
endif()
