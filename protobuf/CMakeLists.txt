project(protobuf)

set(CMAKE_BINARRY_DIR "${CMAKE_SOURCE_DIR}/build")

set(protobuf_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/src
)

set(protobuf_INCLUDE_DIRS ${protobuf_INCLUDE_DIRS} PARENT_SCOPE)  # adding include dirs to a parent scope

file(GLOB protobuf_SRC_FILES
src/google/protobuf/descriptor.cc
src/google/protobuf/descriptor_database.cc
src/google/protobuf/descriptor.pb.cc
src/google/protobuf/dynamic_message.cc
src/google/protobuf/extension_set.cc
src/google/protobuf/extension_set_heavy.cc
src/google/protobuf/generated_message_reflection.cc
src/google/protobuf/generated_message_util.cc
src/google/protobuf/message.cc
src/google/protobuf/message_lite.cc
src/google/protobuf/reflection_ops.cc
src/google/protobuf/repeated_field.cc
src/google/protobuf/service.cc
src/google/protobuf/text_format.cc
src/google/protobuf/unknown_field_set.cc
src/google/protobuf/wire_format.cc
src/google/protobuf/wire_format_lite.cc

src/google/protobuf/io/coded_stream.cc
src/google/protobuf/io/zero_copy_stream_impl.cc
src/google/protobuf/io/tokenizer.cc
src/google/protobuf/io/zero_copy_stream_impl_lite.cc
src/google/protobuf/io/gzip_stream.cc
src/google/protobuf/io/printer.cc
src/google/protobuf/io/zero_copy_stream.cc

src/google/protobuf/stubs/common.cc
src/google/protobuf/stubs/once.cc
src/google/protobuf/stubs/strutil.cc
src/google/protobuf/stubs/stringprintf.cc
src/google/protobuf/stubs/structurally_valid.cc
src/google/protobuf/stubs/substitute.cc
)

if(SINGLE_LIBRARY)
add_library(protobuf INTERFACE)
set(protobuf_SRC_FILES ${protobuf_SRC_FILES} PARENT_SCOPE)
else()
add_library(protobuf OBJECT ${protobuf_SRC_FILES})
endif()

