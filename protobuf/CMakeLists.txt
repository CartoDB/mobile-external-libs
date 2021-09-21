project(protobuf)

file(GLOB protobuf_SRC_FILES
  protobuf/src/google/protobuf/any_lite.cc
  protobuf/src/google/protobuf/arena.cc
  protobuf/src/google/protobuf/arenastring.cc
  protobuf/src/google/protobuf/extension_set.cc
  protobuf/src/google/protobuf/generated_enum_util.cc
  protobuf/src/google/protobuf/generated_message_table_driven_lite.cc
  protobuf/src/google/protobuf/generated_message_util.cc
  protobuf/src/google/protobuf/implicit_weak_message.cc
  protobuf/src/google/protobuf/io/coded_stream.cc
  protobuf/src/google/protobuf/io/io_win32.cc
  protobuf/src/google/protobuf/io/strtod.cc
  protobuf/src/google/protobuf/io/zero_copy_stream.cc
  protobuf/src/google/protobuf/io/zero_copy_stream_impl.cc
  protobuf/src/google/protobuf/io/zero_copy_stream_impl_lite.cc
  protobuf/src/google/protobuf/map.cc
  protobuf/src/google/protobuf/message_lite.cc
  protobuf/src/google/protobuf/parse_context.cc
  protobuf/src/google/protobuf/repeated_field.cc
  protobuf/src/google/protobuf/stubs/bytestream.cc
  protobuf/src/google/protobuf/stubs/common.cc
  protobuf/src/google/protobuf/stubs/int128.cc
  protobuf/src/google/protobuf/stubs/status.cc
  protobuf/src/google/protobuf/stubs/statusor.cc
  protobuf/src/google/protobuf/stubs/stringpiece.cc
  protobuf/src/google/protobuf/stubs/stringprintf.cc
  protobuf/src/google/protobuf/stubs/structurally_valid.cc
  protobuf/src/google/protobuf/stubs/strutil.cc
  protobuf/src/google/protobuf/stubs/time.cc
  protobuf/src/google/protobuf/wire_format_lite.cc
)

set(protobuf_FLAGS "")
if(WIN32)
  set(protobuf_FLAGS ${protobuf_FLAGS} "/wd4018" "/wd4065" "/wd4506" "/wd4267")
elseif(ANDROID OR APPLE)
  set(protobuf_FLAGS ${protobuf_FLAGS} "-DHAVE_PTHREAD" "-Wno-narrowing" "-Wno-conversion")
endif()

set(protobuf_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/protobuf/src" PARENT_SCOPE)

if(SINGLE_LIBRARY)
  set(protobuf_SRC_FILES ${protobuf_SRC_FILES} PARENT_SCOPE)
  set(protobuf_FLAGS ${protobuf_FLAGS} PARENT_SCOPE)
  add_library(protobuf INTERFACE)
else()
  add_compile_options(${protobuf_FLAGS})
  include_directories("${PROJECT_SOURCE_DIR}/protobuf/src")
  add_library(protobuf OBJECT ${protobuf_SRC_FILES})
endif()
