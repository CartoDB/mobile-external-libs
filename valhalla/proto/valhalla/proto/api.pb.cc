// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: api.proto

#include "api.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace valhalla {
constexpr Api::Api(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : options_(nullptr)
  , trip_(nullptr)
  , directions_(nullptr)
  , info_(nullptr){}
struct ApiDefaultTypeInternal {
  constexpr ApiDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~ApiDefaultTypeInternal() {}
  union {
    Api _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT ApiDefaultTypeInternal _Api_default_instance_;
}  // namespace valhalla
namespace valhalla {

// ===================================================================

class Api::_Internal {
 public:
  using HasBits = decltype(std::declval<Api>()._has_bits_);
  static const ::valhalla::Options& options(const Api* msg);
  static void set_has_options(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static const ::valhalla::Trip& trip(const Api* msg);
  static void set_has_trip(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static const ::valhalla::Directions& directions(const Api* msg);
  static void set_has_directions(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static const ::valhalla::Info& info(const Api* msg);
  static void set_has_info(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
};

const ::valhalla::Options&
Api::_Internal::options(const Api* msg) {
  return *msg->options_;
}
const ::valhalla::Trip&
Api::_Internal::trip(const Api* msg) {
  return *msg->trip_;
}
const ::valhalla::Directions&
Api::_Internal::directions(const Api* msg) {
  return *msg->directions_;
}
const ::valhalla::Info&
Api::_Internal::info(const Api* msg) {
  return *msg->info_;
}
void Api::clear_options() {
  if (options_ != nullptr) options_->Clear();
  _has_bits_[0] &= ~0x00000001u;
}
void Api::clear_trip() {
  if (trip_ != nullptr) trip_->Clear();
  _has_bits_[0] &= ~0x00000002u;
}
void Api::clear_directions() {
  if (directions_ != nullptr) directions_->Clear();
  _has_bits_[0] &= ~0x00000004u;
}
void Api::clear_info() {
  if (info_ != nullptr) info_->Clear();
  _has_bits_[0] &= ~0x00000008u;
}
Api::Api(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::MessageLite(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:valhalla.Api)
}
Api::Api(const Api& from)
  : ::PROTOBUF_NAMESPACE_ID::MessageLite(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<std::string>(from._internal_metadata_);
  if (from._internal_has_options()) {
    options_ = new ::valhalla::Options(*from.options_);
  } else {
    options_ = nullptr;
  }
  if (from._internal_has_trip()) {
    trip_ = new ::valhalla::Trip(*from.trip_);
  } else {
    trip_ = nullptr;
  }
  if (from._internal_has_directions()) {
    directions_ = new ::valhalla::Directions(*from.directions_);
  } else {
    directions_ = nullptr;
  }
  if (from._internal_has_info()) {
    info_ = new ::valhalla::Info(*from.info_);
  } else {
    info_ = nullptr;
  }
  // @@protoc_insertion_point(copy_constructor:valhalla.Api)
}

inline void Api::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&options_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&info_) -
    reinterpret_cast<char*>(&options_)) + sizeof(info_));
}

Api::~Api() {
  // @@protoc_insertion_point(destructor:valhalla.Api)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<std::string>();
}

inline void Api::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  if (this != internal_default_instance()) delete options_;
  if (this != internal_default_instance()) delete trip_;
  if (this != internal_default_instance()) delete directions_;
  if (this != internal_default_instance()) delete info_;
}

void Api::ArenaDtor(void* object) {
  Api* _this = reinterpret_cast< Api* >(object);
  (void)_this;
}
void Api::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void Api::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Api::Clear() {
// @@protoc_insertion_point(message_clear_start:valhalla.Api)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000000fu) {
    if (cached_has_bits & 0x00000001u) {
      GOOGLE_DCHECK(options_ != nullptr);
      options_->Clear();
    }
    if (cached_has_bits & 0x00000002u) {
      GOOGLE_DCHECK(trip_ != nullptr);
      trip_->Clear();
    }
    if (cached_has_bits & 0x00000004u) {
      GOOGLE_DCHECK(directions_ != nullptr);
      directions_->Clear();
    }
    if (cached_has_bits & 0x00000008u) {
      GOOGLE_DCHECK(info_ != nullptr);
      info_->Clear();
    }
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<std::string>();
}

const char* Api::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional .valhalla.Options options = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr = ctx->ParseMessage(_internal_mutable_options(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional .valhalla.Trip trip = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr = ctx->ParseMessage(_internal_mutable_trip(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional .valhalla.Directions directions = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 26)) {
          ptr = ctx->ParseMessage(_internal_mutable_directions(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional .valhalla.Info info = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 34)) {
          ptr = ctx->ParseMessage(_internal_mutable_info(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<std::string>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* Api::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:valhalla.Api)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional .valhalla.Options options = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        1, _Internal::options(this), target, stream);
  }

  // optional .valhalla.Trip trip = 2;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        2, _Internal::trip(this), target, stream);
  }

  // optional .valhalla.Directions directions = 3;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        3, _Internal::directions(this), target, stream);
  }

  // optional .valhalla.Info info = 4;
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        4, _Internal::info(this), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = stream->WriteRaw(_internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).data(),
        static_cast<int>(_internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).size()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:valhalla.Api)
  return target;
}

size_t Api::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:valhalla.Api)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000000fu) {
    // optional .valhalla.Options options = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *options_);
    }

    // optional .valhalla.Trip trip = 2;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *trip_);
    }

    // optional .valhalla.Directions directions = 3;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *directions_);
    }

    // optional .valhalla.Info info = 4;
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *info_);
    }

  }
  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    total_size += _internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).size();
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Api::CheckTypeAndMergeFrom(
    const ::PROTOBUF_NAMESPACE_ID::MessageLite& from) {
  MergeFrom(*::PROTOBUF_NAMESPACE_ID::internal::DownCast<const Api*>(
      &from));
}

void Api::MergeFrom(const Api& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:valhalla.Api)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x0000000fu) {
    if (cached_has_bits & 0x00000001u) {
      _internal_mutable_options()->::valhalla::Options::MergeFrom(from._internal_options());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_mutable_trip()->::valhalla::Trip::MergeFrom(from._internal_trip());
    }
    if (cached_has_bits & 0x00000004u) {
      _internal_mutable_directions()->::valhalla::Directions::MergeFrom(from._internal_directions());
    }
    if (cached_has_bits & 0x00000008u) {
      _internal_mutable_info()->::valhalla::Info::MergeFrom(from._internal_info());
    }
  }
  _internal_metadata_.MergeFrom<std::string>(from._internal_metadata_);
}

void Api::CopyFrom(const Api& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:valhalla.Api)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Api::IsInitialized() const {
  return true;
}

void Api::InternalSwap(Api* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(Api, info_)
      + sizeof(Api::info_)
      - PROTOBUF_FIELD_OFFSET(Api, options_)>(
          reinterpret_cast<char*>(&options_),
          reinterpret_cast<char*>(&other->options_));
}

std::string Api::GetTypeName() const {
  return "valhalla.Api";
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace valhalla
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::valhalla::Api* Arena::CreateMaybeMessage< ::valhalla::Api >(Arena* arena) {
  return Arena::CreateMessageInternal< ::valhalla::Api >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
