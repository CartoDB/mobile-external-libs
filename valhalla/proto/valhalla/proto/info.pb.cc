// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: info.proto

#include "info.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace valhalla {
constexpr Statistic::Statistic(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : name_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , value_(0){}
struct StatisticDefaultTypeInternal {
  constexpr StatisticDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~StatisticDefaultTypeInternal() {}
  union {
    Statistic _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT StatisticDefaultTypeInternal _Statistic_default_instance_;
constexpr Info::Info(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : statistics_(){}
struct InfoDefaultTypeInternal {
  constexpr InfoDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~InfoDefaultTypeInternal() {}
  union {
    Info _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT InfoDefaultTypeInternal _Info_default_instance_;
}  // namespace valhalla
namespace valhalla {

// ===================================================================

class Statistic::_Internal {
 public:
  using HasBits = decltype(std::declval<Statistic>()._has_bits_);
  static void set_has_name(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_value(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

Statistic::Statistic(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::MessageLite(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:valhalla.Statistic)
}
Statistic::Statistic(const Statistic& from)
  : ::PROTOBUF_NAMESPACE_ID::MessageLite(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<std::string>(from._internal_metadata_);
  name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from._internal_has_name()) {
    name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_name(), 
      GetArena());
  }
  value_ = from.value_;
  // @@protoc_insertion_point(copy_constructor:valhalla.Statistic)
}

void Statistic::SharedCtor() {
name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
value_ = 0;
}

Statistic::~Statistic() {
  // @@protoc_insertion_point(destructor:valhalla.Statistic)
  SharedDtor();
  _internal_metadata_.Delete<std::string>();
}

void Statistic::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void Statistic::ArenaDtor(void* object) {
  Statistic* _this = reinterpret_cast< Statistic* >(object);
  (void)_this;
}
void Statistic::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void Statistic::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Statistic::Clear() {
// @@protoc_insertion_point(message_clear_start:valhalla.Statistic)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    name_.ClearNonDefaultToEmpty();
  }
  value_ = 0;
  _has_bits_.Clear();
  _internal_metadata_.Clear<std::string>();
}

const char* Statistic::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // optional string name = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_name();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional double value = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 17)) {
          _Internal::set_has_value(&has_bits);
          value_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<std::string>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* Statistic::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:valhalla.Statistic)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional string name = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_name(), target);
  }

  // optional double value = 2;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(2, this->_internal_value(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = stream->WriteRaw(_internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).data(),
        static_cast<int>(_internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).size()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:valhalla.Statistic)
  return target;
}

size_t Statistic::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:valhalla.Statistic)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional string name = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_name());
    }

    // optional double value = 2;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 + 8;
    }

  }
  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    total_size += _internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).size();
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Statistic::CheckTypeAndMergeFrom(
    const ::PROTOBUF_NAMESPACE_ID::MessageLite& from) {
  MergeFrom(*::PROTOBUF_NAMESPACE_ID::internal::DownCast<const Statistic*>(
      &from));
}

void Statistic::MergeFrom(const Statistic& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:valhalla.Statistic)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<std::string>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_name(from._internal_name());
    }
    if (cached_has_bits & 0x00000002u) {
      value_ = from.value_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void Statistic::CopyFrom(const Statistic& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:valhalla.Statistic)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Statistic::IsInitialized() const {
  return true;
}

void Statistic::InternalSwap(Statistic* other) {
  using std::swap;
  _internal_metadata_.Swap<std::string>(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  name_.Swap(&other->name_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  swap(value_, other->value_);
}

std::string Statistic::GetTypeName() const {
  return "valhalla.Statistic";
}


// ===================================================================

class Info::_Internal {
 public:
};

Info::Info(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::MessageLite(arena),
  statistics_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:valhalla.Info)
}
Info::Info(const Info& from)
  : ::PROTOBUF_NAMESPACE_ID::MessageLite(),
      statistics_(from.statistics_) {
  _internal_metadata_.MergeFrom<std::string>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:valhalla.Info)
}

void Info::SharedCtor() {
}

Info::~Info() {
  // @@protoc_insertion_point(destructor:valhalla.Info)
  SharedDtor();
  _internal_metadata_.Delete<std::string>();
}

void Info::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void Info::ArenaDtor(void* object) {
  Info* _this = reinterpret_cast< Info* >(object);
  (void)_this;
}
void Info::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void Info::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Info::Clear() {
// @@protoc_insertion_point(message_clear_start:valhalla.Info)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  statistics_.Clear();
  _internal_metadata_.Clear<std::string>();
}

const char* Info::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // repeated .valhalla.Statistic statistics = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_statistics(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<std::string>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* Info::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:valhalla.Info)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .valhalla.Statistic statistics = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->_internal_statistics_size()); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(1, this->_internal_statistics(i), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = stream->WriteRaw(_internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).data(),
        static_cast<int>(_internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).size()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:valhalla.Info)
  return target;
}

size_t Info::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:valhalla.Info)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .valhalla.Statistic statistics = 1;
  total_size += 1UL * this->_internal_statistics_size();
  for (const auto& msg : this->statistics_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    total_size += _internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).size();
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Info::CheckTypeAndMergeFrom(
    const ::PROTOBUF_NAMESPACE_ID::MessageLite& from) {
  MergeFrom(*::PROTOBUF_NAMESPACE_ID::internal::DownCast<const Info*>(
      &from));
}

void Info::MergeFrom(const Info& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:valhalla.Info)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<std::string>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  statistics_.MergeFrom(from.statistics_);
}

void Info::CopyFrom(const Info& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:valhalla.Info)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Info::IsInitialized() const {
  return true;
}

void Info::InternalSwap(Info* other) {
  using std::swap;
  _internal_metadata_.Swap<std::string>(&other->_internal_metadata_);
  statistics_.InternalSwap(&other->statistics_);
}

std::string Info::GetTypeName() const {
  return "valhalla.Info";
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace valhalla
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::valhalla::Statistic* Arena::CreateMaybeMessage< ::valhalla::Statistic >(Arena* arena) {
  return Arena::CreateMessageInternal< ::valhalla::Statistic >(arena);
}
template<> PROTOBUF_NOINLINE ::valhalla::Info* Arena::CreateMaybeMessage< ::valhalla::Info >(Arena* arena) {
  return Arena::CreateMessageInternal< ::valhalla::Info >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>