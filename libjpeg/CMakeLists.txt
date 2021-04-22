project(libjpeg)

file(GLOB libjpeg_SRC_FILES
  libjpeg/jcapimin.c libjpeg/jcapistd.c libjpeg/jccoefct.c libjpeg/jccolor.c libjpeg/jcdctmgr.c libjpeg/jchuff.c
  libjpeg/jcicc.c libjpeg/jcinit.c libjpeg/jcmainct.c libjpeg/jcmarker.c libjpeg/jcmaster.c libjpeg/jcomapi.c libjpeg/jcparam.c
  libjpeg/jcphuff.c libjpeg/jcprepct.c libjpeg/jcsample.c libjpeg/jctrans.c libjpeg/jdapimin.c libjpeg/jdapistd.c libjpeg/jdatadst.c
  libjpeg/jdatasrc.c libjpeg/jdcoefct.c libjpeg/jdcolor.c libjpeg/jddctmgr.c libjpeg/jdhuff.c libjpeg/jdicc.c libjpeg/jdinput.c
  libjpeg/jdmainct.c libjpeg/jdmarker.c libjpeg/jdmaster.c libjpeg/jdmerge.c libjpeg/jdphuff.c libjpeg/jdpostct.c libjpeg/jdsample.c
  libjpeg/jdtrans.c libjpeg/jerror.c libjpeg/jfdctflt.c libjpeg/jfdctfst.c libjpeg/jfdctint.c libjpeg/jidctflt.c libjpeg/jidctfst.c
  libjpeg/jidctint.c libjpeg/jidctred.c libjpeg/jquant1.c libjpeg/jquant2.c libjpeg/jutils.c libjpeg/jmemmgr.c libjpeg/jmemnobs.c
  libjpeg/jaricom.c libjpeg/jcarith.c libjpeg/jdarith.c libjpeg/jsimd_none.c
)

set(libjpeg_FLAGS "-DNO_GETENV")

set(libjpeg_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/config" "${PROJECT_SOURCE_DIR}/libjpeg" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(libjpeg_SRC_FILES ${libjpeg_SRC_FILES} PARENT_SCOPE)
  set(libjpeg_FLAGS ${libjpeg_FLAGS} PARENT_SCOPE)
  add_library(libjpeg INTERFACE)
else()
  add_compile_options(${libjpeg_FLAGS})
  include_directories("${PROJECT_SOURCE_DIR}/config" "${PROJECT_SOURCE_DIR}/libjpeg")
  add_library(libjpeg OBJECT ${libjpeg_SRC_FILES})
endif()
