project(jpeg)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNO_GETENV")

file(GLOB jpeg_SRC_FILES
  jaricom.c jcapimin.c jcapistd.c jcarith.c jccoefct.c jccolor.c
  jcdctmgr.c jchuff.c jcinit.c jcmainct.c jcmarker.c jcmaster.c
  jcomapi.c jcparam.c jcprepct.c jcsample.c jctrans.c jdapimin.c
  jdapistd.c jdarith.c jdatadst.c jdatasrc.c jdcoefct.c jdcolor.c
  jddctmgr.c jdhuff.c jdinput.c jdmainct.c jdmarker.c jdmaster.c
  jdmerge.c jdpostct.c jdsample.c jdtrans.c jerror.c jfdctflt.c
  jfdctfst.c jfdctint.c jidctflt.c jidctfst.c jidctint.c jquant1.c
  jquant2.c jutils.c jmemmgr.c
  jmemansi.c
)

include_directories("${PROJECT_SOURCE_DIR}")

if(SINGLE_LIBRARY)
add_library(jpeg INTERFACE)
set(jpeg_SRC_FILES ${jpeg_SRC_FILES} PARENT_SCOPE)
else()
add_library(jpeg OBJECT ${jpeg_SRC_FILES})
endif()
