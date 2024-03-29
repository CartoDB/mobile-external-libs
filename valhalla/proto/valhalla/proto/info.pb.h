// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: info.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_info_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_info_2eproto

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
#include <google/protobuf/generated_enum_util.h>
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
  static const uint32_t offsets[];
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

enum StatisticType : int {
  count = 0,
  gauge = 1,
  timing = 2,
  set = 3
};
bool StatisticType_IsValid(int value);
constexpr StatisticType StatisticType_MIN = count;
constexpr StatisticType StatisticType_MAX = set;
constexpr int StatisticType_ARRAYSIZE = StatisticType_MAX + 1;

const std::string& StatisticType_Name(StatisticType value);
template<typename T>
inline const std::string& StatisticType_Name(T enum_t_value) {
  static_assert(::std::is_same<T, StatisticType>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function StatisticType_Name.");
  return StatisticType_Name(static_cast<StatisticType>(enum_t_value));
}
bool StatisticType_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, StatisticType* value);
// ===================================================================

class Statistic final :
    public ::PROTOBUF_NAMESPACE_ID::MessageLite /* @@protoc_insertion_point(class_definition:valhalla.Statistic) */ {
 public:
  inline Statistic() : Statistic(nullptr) {}
  ~Statistic() override;
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
  void UnsafeArenaSwap(Statistic* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  Statistic* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<Statistic>(arena);
  }
  void CheckTypeAndMergeFrom(const ::PROTOBUF_NAMESPACE_ID::MessageLite& from)  final;
  void CopyFrom(const Statistic& from);
  void MergeFrom(const Statistic& from);
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
  void InternalSwap(Statistic* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "valhalla.Statistic";
  }
  protected:
  explicit Statistic(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  std::string GetTypeName() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kKeyFieldNumber = 1,
    kValueFieldNumber = 2,
    kFrequencyFieldNumber = 3,
    kTypeFieldNumber = 4,
  };
  // optional string key = 1;
  bool has_key() const;
  private:
  bool _internal_has_key() const;
  public:
  void clear_key();
  const std::string& key() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_key(ArgT0&& arg0, ArgT... args);
  std::string* mutable_key();
  PROTOBUF_NODISCARD std::string* release_key();
  void set_allocated_key(std::string* key);
  private:
  const std::string& _internal_key() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_key(const std::string& value);
  std::string* _internal_mutable_key();
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

  // optional float frequency = 3;
  bool has_frequency() const;
  private:
  bool _internal_has_frequency() const;
  public:
  void clear_frequency();
  float frequency() const;
  void set_frequency(float value);
  private:
  float _internal_frequency() const;
  void _internal_set_frequency(float value);
  public:

  // optional .valhalla.StatisticType type = 4;
  bool has_type() const;
  private:
  bool _internal_has_type() const;
  public:
  void clear_type();
  ::valhalla::StatisticType type() const;
  void set_type(::valhalla::StatisticType value);
  private:
  ::valhalla::StatisticType _internal_type() const;
  void _internal_set_type(::valhalla::StatisticType value);
  public:

  // @@protoc_insertion_point(class_scope:valhalla.Statistic)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr key_;
  double value_;
  float frequency_;
  int type_;
  friend struct ::TableStruct_info_2eproto;
};
// -------------------------------------------------------------------

class Info final :
    public ::PROTOBUF_NAMESPACE_ID::MessageLite /* @@protoc_insertion_point(class_definition:valhalla.Info) */ {
 public:
  inline Info() : Info(nullptr) {}
  ~Info() override;
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
  void UnsafeArenaSwap(Info* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  Info* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<Info>(arena);
  }
  void CheckTypeAndMergeFrom(const ::PROTOBUF_NAMESPACE_ID::MessageLite& from)  final;
  void CopyFrom(const Info& from);
  void MergeFrom(const Info& from);
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
  void InternalSwap(Info* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "valhalla.Info";
  }
  protected:
  explicit Info(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  std::string GetTypeName() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kStatisticsFieldNumber = 1,
    kErrorFieldNumber = 2,
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

  // optional bool error = 2;
  bool has_error() const;
  private:
  bool _internal_has_error() const;
  public:
  void clear_error();
  bool error() const;
  void set_error(bool value);
  private:
  bool _internal_error() const;
  void _internal_set_error(bool value);
  public:

  // @@protoc_insertion_point(class_scope:valhalla.Info)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::valhalla::Statistic > statistics_;
  bool error_;
  friend struct ::TableStruct_info_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Statistic

// optional string key = 1;
inline bool Statistic::_internal_has_key() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool Statistic::has_key() const {
  return _internal_has_key();
}
inline void Statistic::clear_key() {
  key_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& Statistic::key() const {
  // @@protoc_insertion_point(field_get:valhalla.Statistic.key)
  return _internal_key();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void Statistic::set_key(ArgT0&& arg0, ArgT... args) {
 _has_bits_[0] |= 0x00000001u;
 key_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:valhalla.Statistic.key)
}
inline std::string* Statistic::mutable_key() {
  std::string* _s = _internal_mutable_key();
  // @@protoc_insertion_point(field_mutable:valhalla.Statistic.key)
  return _s;
}
inline const std::string& Statistic::_internal_key() const {
  return key_.Get();
}
inline void Statistic::_internal_set_key(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  key_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* Statistic::_internal_mutable_key() {
  _has_bits_[0] |= 0x00000001u;
  return key_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* Statistic::release_key() {
  // @@protoc_insertion_point(field_release:valhalla.Statistic.key)
  if (!_internal_has_key()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  auto* p = key_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (key_.IsDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited())) {
    key_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  return p;
}
inline void Statistic::set_allocated_key(std::string* key) {
  if (key != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  key_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), key,
      GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (key_.IsDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited())) {
    key_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:valhalla.Statistic.key)
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

// optional float frequency = 3;
inline bool Statistic::_internal_has_frequency() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool Statistic::has_frequency() const {
  return _internal_has_frequency();
}
inline void Statistic::clear_frequency() {
  frequency_ = 0;
  _has_bits_[0] &= ~0x00000004u;
}
inline float Statistic::_internal_frequency() const {
  return frequency_;
}
inline float Statistic::frequency() const {
  // @@protoc_insertion_point(field_get:valhalla.Statistic.frequency)
  return _internal_frequency();
}
inline void Statistic::_internal_set_frequency(float value) {
  _has_bits_[0] |= 0x00000004u;
  frequency_ = value;
}
inline void Statistic::set_frequency(float value) {
  _internal_set_frequency(value);
  // @@protoc_insertion_point(field_set:valhalla.Statistic.frequency)
}

// optional .valhalla.StatisticType type = 4;
inline bool Statistic::_internal_has_type() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool Statistic::has_type() const {
  return _internal_has_type();
}
inline void Statistic::clear_type() {
  type_ = 0;
  _has_bits_[0] &= ~0x00000008u;
}
inline ::valhalla::StatisticType Statistic::_internal_type() const {
  return static_cast< ::valhalla::StatisticType >(type_);
}
inline ::valhalla::StatisticType Statistic::type() const {
  // @@protoc_insertion_point(field_get:valhalla.Statistic.type)
  return _internal_type();
}
inline void Statistic::_internal_set_type(::valhalla::StatisticType value) {
  assert(::valhalla::StatisticType_IsValid(value));
  _has_bits_[0] |= 0x00000008u;
  type_ = value;
}
inline void Statistic::set_type(::valhalla::StatisticType value) {
  _internal_set_type(value);
  // @@protoc_insertion_point(field_set:valhalla.Statistic.type)
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
  ::valhalla::Statistic* _add = _internal_add_statistics();
  // @@protoc_insertion_point(field_add:valhalla.Info.statistics)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::valhalla::Statistic >&
Info::statistics() const {
  // @@protoc_insertion_point(field_list:valhalla.Info.statistics)
  return statistics_;
}

// optional bool error = 2;
inline bool Info::_internal_has_error() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool Info::has_error() const {
  return _internal_has_error();
}
inline void Info::clear_error() {
  error_ = false;
  _has_bits_[0] &= ~0x00000001u;
}
inline bool Info::_internal_error() const {
  return error_;
}
inline bool Info::error() const {
  // @@protoc_insertion_point(field_get:valhalla.Info.error)
  return _internal_error();
}
inline void Info::_internal_set_error(bool value) {
  _has_bits_[0] |= 0x00000001u;
  error_ = value;
}
inline void Info::set_error(bool value) {
  _internal_set_error(value);
  // @@protoc_insertion_point(field_set:valhalla.Info.error)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace valhalla

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::valhalla::StatisticType> : ::std::true_type {};

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_info_2eproto
