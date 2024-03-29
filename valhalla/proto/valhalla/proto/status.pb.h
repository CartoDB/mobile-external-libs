// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: status.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_status_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_status_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3019000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3019001 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_status_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_status_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const uint32_t offsets[];
};
namespace valhalla {
class Status;
struct StatusDefaultTypeInternal;
extern StatusDefaultTypeInternal _Status_default_instance_;
}  // namespace valhalla
PROTOBUF_NAMESPACE_OPEN
template<> ::valhalla::Status* Arena::CreateMaybeMessage<::valhalla::Status>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace valhalla {

// ===================================================================

class Status final :
    public ::PROTOBUF_NAMESPACE_ID::MessageLite /* @@protoc_insertion_point(class_definition:valhalla.Status) */ {
 public:
  inline Status() : Status(nullptr) {}
  ~Status() override;
  explicit constexpr Status(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Status(const Status& from);
  Status(Status&& from) noexcept
    : Status() {
    *this = ::std::move(from);
  }

  inline Status& operator=(const Status& from) {
    CopyFrom(from);
    return *this;
  }
  inline Status& operator=(Status&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const std::string& unknown_fields() const {
    return _internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString);
  }
  inline std::string* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<std::string>();
  }

  static const Status& default_instance() {
    return *internal_default_instance();
  }
  static inline const Status* internal_default_instance() {
    return reinterpret_cast<const Status*>(
               &_Status_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(Status& a, Status& b) {
    a.Swap(&b);
  }
  inline void Swap(Status* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Status* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  Status* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<Status>(arena);
  }
  void CheckTypeAndMergeFrom(const ::PROTOBUF_NAMESPACE_ID::MessageLite& from)  final;
  void CopyFrom(const Status& from);
  void MergeFrom(const Status& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(Status* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "valhalla.Status";
  }
  protected:
  explicit Status(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  std::string GetTypeName() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kBboxFieldNumber = 5,
    kVersionFieldNumber = 6,
    kHasTilesFieldNumber = 1,
    kHasAdminsFieldNumber = 2,
    kHasTimezonesFieldNumber = 3,
    kHasLiveTrafficFieldNumber = 4,
  };
  // optional string bbox = 5;
  bool has_bbox() const;
  private:
  bool _internal_has_bbox() const;
  public:
  void clear_bbox();
  const std::string& bbox() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_bbox(ArgT0&& arg0, ArgT... args);
  std::string* mutable_bbox();
  PROTOBUF_NODISCARD std::string* release_bbox();
  void set_allocated_bbox(std::string* bbox);
  private:
  const std::string& _internal_bbox() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_bbox(const std::string& value);
  std::string* _internal_mutable_bbox();
  public:

  // optional string version = 6;
  bool has_version() const;
  private:
  bool _internal_has_version() const;
  public:
  void clear_version();
  const std::string& version() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_version(ArgT0&& arg0, ArgT... args);
  std::string* mutable_version();
  PROTOBUF_NODISCARD std::string* release_version();
  void set_allocated_version(std::string* version);
  private:
  const std::string& _internal_version() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_version(const std::string& value);
  std::string* _internal_mutable_version();
  public:

  // optional bool has_tiles = 1;
  bool has_has_tiles() const;
  private:
  bool _internal_has_has_tiles() const;
  public:
  void clear_has_tiles();
  bool has_tiles() const;
  void set_has_tiles(bool value);
  private:
  bool _internal_has_tiles() const;
  void _internal_set_has_tiles(bool value);
  public:

  // optional bool has_admins = 2;
  bool has_has_admins() const;
  private:
  bool _internal_has_has_admins() const;
  public:
  void clear_has_admins();
  bool has_admins() const;
  void set_has_admins(bool value);
  private:
  bool _internal_has_admins() const;
  void _internal_set_has_admins(bool value);
  public:

  // optional bool has_timezones = 3;
  bool has_has_timezones() const;
  private:
  bool _internal_has_has_timezones() const;
  public:
  void clear_has_timezones();
  bool has_timezones() const;
  void set_has_timezones(bool value);
  private:
  bool _internal_has_timezones() const;
  void _internal_set_has_timezones(bool value);
  public:

  // optional bool has_live_traffic = 4;
  bool has_has_live_traffic() const;
  private:
  bool _internal_has_has_live_traffic() const;
  public:
  void clear_has_live_traffic();
  bool has_live_traffic() const;
  void set_has_live_traffic(bool value);
  private:
  bool _internal_has_live_traffic() const;
  void _internal_set_has_live_traffic(bool value);
  public:

  // @@protoc_insertion_point(class_scope:valhalla.Status)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr bbox_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr version_;
  bool has_tiles_;
  bool has_admins_;
  bool has_timezones_;
  bool has_live_traffic_;
  friend struct ::TableStruct_status_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Status

// optional bool has_tiles = 1;
inline bool Status::_internal_has_has_tiles() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool Status::has_has_tiles() const {
  return _internal_has_has_tiles();
}
inline void Status::clear_has_tiles() {
  has_tiles_ = false;
  _has_bits_[0] &= ~0x00000004u;
}
inline bool Status::_internal_has_tiles() const {
  return has_tiles_;
}
inline bool Status::has_tiles() const {
  // @@protoc_insertion_point(field_get:valhalla.Status.has_tiles)
  return _internal_has_tiles();
}
inline void Status::_internal_set_has_tiles(bool value) {
  _has_bits_[0] |= 0x00000004u;
  has_tiles_ = value;
}
inline void Status::set_has_tiles(bool value) {
  _internal_set_has_tiles(value);
  // @@protoc_insertion_point(field_set:valhalla.Status.has_tiles)
}

// optional bool has_admins = 2;
inline bool Status::_internal_has_has_admins() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool Status::has_has_admins() const {
  return _internal_has_has_admins();
}
inline void Status::clear_has_admins() {
  has_admins_ = false;
  _has_bits_[0] &= ~0x00000008u;
}
inline bool Status::_internal_has_admins() const {
  return has_admins_;
}
inline bool Status::has_admins() const {
  // @@protoc_insertion_point(field_get:valhalla.Status.has_admins)
  return _internal_has_admins();
}
inline void Status::_internal_set_has_admins(bool value) {
  _has_bits_[0] |= 0x00000008u;
  has_admins_ = value;
}
inline void Status::set_has_admins(bool value) {
  _internal_set_has_admins(value);
  // @@protoc_insertion_point(field_set:valhalla.Status.has_admins)
}

// optional bool has_timezones = 3;
inline bool Status::_internal_has_has_timezones() const {
  bool value = (_has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool Status::has_has_timezones() const {
  return _internal_has_has_timezones();
}
inline void Status::clear_has_timezones() {
  has_timezones_ = false;
  _has_bits_[0] &= ~0x00000010u;
}
inline bool Status::_internal_has_timezones() const {
  return has_timezones_;
}
inline bool Status::has_timezones() const {
  // @@protoc_insertion_point(field_get:valhalla.Status.has_timezones)
  return _internal_has_timezones();
}
inline void Status::_internal_set_has_timezones(bool value) {
  _has_bits_[0] |= 0x00000010u;
  has_timezones_ = value;
}
inline void Status::set_has_timezones(bool value) {
  _internal_set_has_timezones(value);
  // @@protoc_insertion_point(field_set:valhalla.Status.has_timezones)
}

// optional bool has_live_traffic = 4;
inline bool Status::_internal_has_has_live_traffic() const {
  bool value = (_has_bits_[0] & 0x00000020u) != 0;
  return value;
}
inline bool Status::has_has_live_traffic() const {
  return _internal_has_has_live_traffic();
}
inline void Status::clear_has_live_traffic() {
  has_live_traffic_ = false;
  _has_bits_[0] &= ~0x00000020u;
}
inline bool Status::_internal_has_live_traffic() const {
  return has_live_traffic_;
}
inline bool Status::has_live_traffic() const {
  // @@protoc_insertion_point(field_get:valhalla.Status.has_live_traffic)
  return _internal_has_live_traffic();
}
inline void Status::_internal_set_has_live_traffic(bool value) {
  _has_bits_[0] |= 0x00000020u;
  has_live_traffic_ = value;
}
inline void Status::set_has_live_traffic(bool value) {
  _internal_set_has_live_traffic(value);
  // @@protoc_insertion_point(field_set:valhalla.Status.has_live_traffic)
}

// optional string bbox = 5;
inline bool Status::_internal_has_bbox() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool Status::has_bbox() const {
  return _internal_has_bbox();
}
inline void Status::clear_bbox() {
  bbox_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& Status::bbox() const {
  // @@protoc_insertion_point(field_get:valhalla.Status.bbox)
  return _internal_bbox();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void Status::set_bbox(ArgT0&& arg0, ArgT... args) {
 _has_bits_[0] |= 0x00000001u;
 bbox_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:valhalla.Status.bbox)
}
inline std::string* Status::mutable_bbox() {
  std::string* _s = _internal_mutable_bbox();
  // @@protoc_insertion_point(field_mutable:valhalla.Status.bbox)
  return _s;
}
inline const std::string& Status::_internal_bbox() const {
  return bbox_.Get();
}
inline void Status::_internal_set_bbox(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  bbox_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* Status::_internal_mutable_bbox() {
  _has_bits_[0] |= 0x00000001u;
  return bbox_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* Status::release_bbox() {
  // @@protoc_insertion_point(field_release:valhalla.Status.bbox)
  if (!_internal_has_bbox()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  auto* p = bbox_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (bbox_.IsDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited())) {
    bbox_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  return p;
}
inline void Status::set_allocated_bbox(std::string* bbox) {
  if (bbox != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  bbox_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), bbox,
      GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (bbox_.IsDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited())) {
    bbox_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:valhalla.Status.bbox)
}

// optional string version = 6;
inline bool Status::_internal_has_version() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool Status::has_version() const {
  return _internal_has_version();
}
inline void Status::clear_version() {
  version_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000002u;
}
inline const std::string& Status::version() const {
  // @@protoc_insertion_point(field_get:valhalla.Status.version)
  return _internal_version();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void Status::set_version(ArgT0&& arg0, ArgT... args) {
 _has_bits_[0] |= 0x00000002u;
 version_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:valhalla.Status.version)
}
inline std::string* Status::mutable_version() {
  std::string* _s = _internal_mutable_version();
  // @@protoc_insertion_point(field_mutable:valhalla.Status.version)
  return _s;
}
inline const std::string& Status::_internal_version() const {
  return version_.Get();
}
inline void Status::_internal_set_version(const std::string& value) {
  _has_bits_[0] |= 0x00000002u;
  version_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* Status::_internal_mutable_version() {
  _has_bits_[0] |= 0x00000002u;
  return version_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* Status::release_version() {
  // @@protoc_insertion_point(field_release:valhalla.Status.version)
  if (!_internal_has_version()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000002u;
  auto* p = version_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (version_.IsDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited())) {
    version_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  return p;
}
inline void Status::set_allocated_version(std::string* version) {
  if (version != nullptr) {
    _has_bits_[0] |= 0x00000002u;
  } else {
    _has_bits_[0] &= ~0x00000002u;
  }
  version_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), version,
      GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (version_.IsDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited())) {
    version_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:valhalla.Status.version)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace valhalla

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_status_2eproto
