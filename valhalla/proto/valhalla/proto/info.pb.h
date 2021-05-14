// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: info.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_info_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_info_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3015000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3015008 < PROTOBUF_MIN_PROTOC_VERSION
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
#define PROTOBUF_INTERNAL_EXPORT_info_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_info_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
namespace valhalla {
class Info;
struct InfoDefaultTypeInternal;
extern InfoDefaultTypeInternal _Info_default_instance_;
class Statistic;
struct StatisticDefaultTypeInternal;
extern StatisticDefaultTypeInternal _Statistic_default_instance_;
}  // namespace valhalla
PROTOBUF_NAMESPACE_OPEN
template<> ::valhalla::Info* Arena::CreateMaybeMessage<::valhalla::Info>(Arena*);
template<> ::valhalla::Statistic* Arena::CreateMaybeMessage<::valhalla::Statistic>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace valhalla {

// ===================================================================

class Statistic PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::MessageLite /* @@protoc_insertion_point(class_definition:valhalla.Statistic) */ {
 public:
  inline Statistic() : Statistic(nullptr) {}
  virtual ~Statistic();
  explicit constexpr Statistic(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Statistic(const Statistic& from);
  Statistic(Statistic&& from) noexcept
    : Statistic() {
    *this = ::std::move(from);
  }

  inline Statistic& operator=(const Statistic& from) {
    CopyFrom(from);
    return *this;
  }
  inline Statistic& operator=(Statistic&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
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

  static const Statistic& default_instance() {
    return *internal_default_instance();
  }
  static inline const Statistic* internal_default_instance() {
    return reinterpret_cast<const Statistic*>(
               &_Statistic_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(Statistic& a, Statistic& b) {
    a.Swap(&b);
  }
  inline void Swap(Statistic* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Statistic* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline Statistic* New() const final {
    return CreateMaybeMessage<Statistic>(nullptr);
  }

  Statistic* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Statistic>(arena);
  }
  void CheckTypeAndMergeFrom(const ::PROTOBUF_NAMESPACE_ID::MessageLite& from)
    final;
  void CopyFrom(const Statistic& from);
  void MergeFrom(const Statistic& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  void DiscardUnknownFields();
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(Statistic* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "valhalla.Statistic";
  }
  protected:
  explicit Statistic(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  std::string GetTypeName() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kNameFieldNumber = 1,
    kValueFieldNumber = 2,
  };
  // optional string name = 1;
  bool has_name() const;
  private:
  bool _internal_has_name() const;
  public:
  void clear_name();
  const std::string& name() const;
  void set_name(const std::string& value);
  void set_name(std::string&& value);
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  std::string* mutable_name();
  std::string* release_name();
  void set_allocated_name(std::string* name);
  private:
  const std::string& _internal_name() const;
  void _internal_set_name(const std::string& value);
  std::string* _internal_mutable_name();
  public:

  // optional double value = 2;
  bool has_value() const;
  private:
  bool _internal_has_value() const;
  public:
  void clear_value();
  double value() const;
  void set_value(double value);
  private:
  double _internal_value() const;
  void _internal_set_value(double value);
  public:

  // @@protoc_insertion_point(class_scope:valhalla.Statistic)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr name_;
  double value_;
  friend struct ::TableStruct_info_2eproto;
};
// -------------------------------------------------------------------

class Info PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::MessageLite /* @@protoc_insertion_point(class_definition:valhalla.Info) */ {
 public:
  inline Info() : Info(nullptr) {}
  virtual ~Info();
  explicit constexpr Info(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Info(const Info& from);
  Info(Info&& from) noexcept
    : Info() {
    *this = ::std::move(from);
  }

  inline Info& operator=(const Info& from) {
    CopyFrom(from);
    return *this;
  }
  inline Info& operator=(Info&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
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

  static const Info& default_instance() {
    return *internal_default_instance();
  }
  static inline const Info* internal_default_instance() {
    return reinterpret_cast<const Info*>(
               &_Info_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(Info& a, Info& b) {
    a.Swap(&b);
  }
  inline void Swap(Info* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Info* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline Info* New() const final {
    return CreateMaybeMessage<Info>(nullptr);
  }

  Info* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Info>(arena);
  }
  void CheckTypeAndMergeFrom(const ::PROTOBUF_NAMESPACE_ID::MessageLite& from)
    final;
  void CopyFrom(const Info& from);
  void MergeFrom(const Info& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  void DiscardUnknownFields();
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(Info* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "valhalla.Info";
  }
  protected:
  explicit Info(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  std::string GetTypeName() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kStatisticsFieldNumber = 1,
  };
  // repeated .valhalla.Statistic statistics = 1;
  int statistics_size() const;
  private:
  int _internal_statistics_size() const;
  public:
  void clear_statistics();
  ::valhalla::Statistic* mutable_statistics(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::valhalla::Statistic >*
      mutable_statistics();
  private:
  const ::valhalla::Statistic& _internal_statistics(int index) const;
  ::valhalla::Statistic* _internal_add_statistics();
  public:
  const ::valhalla::Statistic& statistics(int index) const;
  ::valhalla::Statistic* add_statistics();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::valhalla::Statistic >&
      statistics() const;

  // @@protoc_insertion_point(class_scope:valhalla.Info)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::valhalla::Statistic > statistics_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_info_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Statistic

// optional string name = 1;
inline bool Statistic::_internal_has_name() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool Statistic::has_name() const {
  return _internal_has_name();
}
inline void Statistic::clear_name() {
  name_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& Statistic::name() const {
  // @@protoc_insertion_point(field_get:valhalla.Statistic.name)
  return _internal_name();
}
inline void Statistic::set_name(const std::string& value) {
  _internal_set_name(value);
  // @@protoc_insertion_point(field_set:valhalla.Statistic.name)
}
inline std::string* Statistic::mutable_name() {
  // @@protoc_insertion_point(field_mutable:valhalla.Statistic.name)
  return _internal_mutable_name();
}
inline const std::string& Statistic::_internal_name() const {
  return name_.Get();
}
inline void Statistic::_internal_set_name(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArena());
}
inline void Statistic::set_name(std::string&& value) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::move(value), GetArena());
  // @@protoc_insertion_point(field_set_rvalue:valhalla.Statistic.name)
}
inline void Statistic::set_name(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  _has_bits_[0] |= 0x00000001u;
  name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(value), GetArena());
  // @@protoc_insertion_point(field_set_char:valhalla.Statistic.name)
}
inline void Statistic::set_name(const char* value,
    size_t size) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(
      reinterpret_cast<const char*>(value), size), GetArena());
  // @@protoc_insertion_point(field_set_pointer:valhalla.Statistic.name)
}
inline std::string* Statistic::_internal_mutable_name() {
  _has_bits_[0] |= 0x00000001u;
  return name_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArena());
}
inline std::string* Statistic::release_name() {
  // @@protoc_insertion_point(field_release:valhalla.Statistic.name)
  if (!_internal_has_name()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return name_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}
inline void Statistic::set_allocated_name(std::string* name) {
  if (name != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  name_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), name,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:valhalla.Statistic.name)
}

// optional double value = 2;
inline bool Statistic::_internal_has_value() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool Statistic::has_value() const {
  return _internal_has_value();
}
inline void Statistic::clear_value() {
  value_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline double Statistic::_internal_value() const {
  return value_;
}
inline double Statistic::value() const {
  // @@protoc_insertion_point(field_get:valhalla.Statistic.value)
  return _internal_value();
}
inline void Statistic::_internal_set_value(double value) {
  _has_bits_[0] |= 0x00000002u;
  value_ = value;
}
inline void Statistic::set_value(double value) {
  _internal_set_value(value);
  // @@protoc_insertion_point(field_set:valhalla.Statistic.value)
}

// -------------------------------------------------------------------

// Info

// repeated .valhalla.Statistic statistics = 1;
inline int Info::_internal_statistics_size() const {
  return statistics_.size();
}
inline int Info::statistics_size() const {
  return _internal_statistics_size();
}
inline void Info::clear_statistics() {
  statistics_.Clear();
}
inline ::valhalla::Statistic* Info::mutable_statistics(int index) {
  // @@protoc_insertion_point(field_mutable:valhalla.Info.statistics)
  return statistics_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::valhalla::Statistic >*
Info::mutable_statistics() {
  // @@protoc_insertion_point(field_mutable_list:valhalla.Info.statistics)
  return &statistics_;
}
inline const ::valhalla::Statistic& Info::_internal_statistics(int index) const {
  return statistics_.Get(index);
}
inline const ::valhalla::Statistic& Info::statistics(int index) const {
  // @@protoc_insertion_point(field_get:valhalla.Info.statistics)
  return _internal_statistics(index);
}
inline ::valhalla::Statistic* Info::_internal_add_statistics() {
  return statistics_.Add();
}
inline ::valhalla::Statistic* Info::add_statistics() {
  // @@protoc_insertion_point(field_add:valhalla.Info.statistics)
  return _internal_add_statistics();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::valhalla::Statistic >&
Info::statistics() const {
  // @@protoc_insertion_point(field_list:valhalla.Info.statistics)
  return statistics_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace valhalla

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_info_2eproto