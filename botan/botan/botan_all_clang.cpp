/*
* Botan 2.18.1 Amalgamation
* (C) 1999-2020 The Botan Authors
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include "botan_all.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>


namespace Botan {

/**
* If top bit of arg is set, return ~0. Otherwise return 0.
*/
template<typename T>
inline T expand_top_bit(T a)
   {
   return static_cast<T>(0) - (a >> (sizeof(T)*8-1));
   }

/**
* If arg is zero, return ~0. Otherwise return 0
*/
template<typename T>
inline T ct_is_zero(T x)
   {
   return expand_top_bit<T>(~x & (x - 1));
   }

/**
* Power of 2 test. T should be an unsigned integer type
* @param arg an integer value
* @return true iff arg is 2^n for some n > 0
*/
template<typename T>
inline constexpr bool is_power_of_2(T arg)
   {
   return (arg != 0) && (arg != 1) && ((arg & static_cast<T>(arg-1)) == 0);
   }

/**
* Return the index of the highest set bit
* T is an unsigned integer type
* @param n an integer value
* @return index of the highest set bit in n
*/
template<typename T>
inline size_t high_bit(T n)
   {
   size_t hb = 0;

   for(size_t s = 8*sizeof(T) / 2; s > 0; s /= 2)
      {
      const size_t z = s * ((~ct_is_zero(n >> s)) & 1);
      hb += z;
      n >>= z;
      }

   hb += n;

   return hb;
   }

/**
* Return the number of significant bytes in n
* @param n an integer value
* @return number of significant bytes in n
*/
template<typename T>
inline size_t significant_bytes(T n)
   {
   size_t b = 0;

   for(size_t s = 8*sizeof(n) / 2; s >= 8; s /= 2)
      {
      const size_t z = s * (~ct_is_zero(n >> s) & 1);
      b += z/8;
      n >>= z;
      }

   b += (n != 0);

   return b;
   }

/**
* Count the trailing zero bits in n
* @param n an integer value
* @return maximum x st 2^x divides n
*/
template<typename T>
inline size_t ctz(T n)
   {
   /*
   * If n == 0 then this function will compute 8*sizeof(T)-1, so
   * initialize lb to 1 if n == 0 to produce the expected result.
   */
   size_t lb = ct_is_zero(n) & 1;

   for(size_t s = 8*sizeof(T) / 2; s > 0; s /= 2)
      {
      const T mask = (static_cast<T>(1) << s) - 1;
      const size_t z = s * (ct_is_zero(n & mask) & 1);
      lb += z;
      n >>= z;
      }

   return lb;
   }

template<typename T>
uint8_t ceil_log2(T x)
   {
   static_assert(sizeof(T) < 32, "Abnormally large scalar");

   if(x >> (sizeof(T)*8-1))
      return sizeof(T)*8;

   uint8_t result = 0;
   T compare = 1;

   while(compare < x)
      {
      compare <<= 1;
      result++;
      }

   return result;
   }

// Potentially variable time ctz used for OCB
inline size_t var_ctz32(uint32_t n)
   {
#if defined(BOTAN_BUILD_COMPILER_IS_GCC) || defined(BOTAN_BUILD_COMPILER_IS_CLANG)
   if(n == 0)
      return 32;
   return __builtin_ctz(n);
#else
   return ctz<uint32_t>(n);
#endif
   }

template<typename T>
inline T bit_permute_step(T x, T mask, size_t shift)
   {
   /*
   See https://reflectionsonsecurity.wordpress.com/2014/05/11/efficient-bit-permutation-using-delta-swaps/
   and http://programming.sirrida.de/bit_perm.html
   */
   const T swap = ((x >> shift) ^ x) & mask;
   return (x ^ swap) ^ (swap << shift);
   }

template<typename T>
inline void swap_bits(T& x, T& y, T mask, size_t shift)
   {
   const T swap = ((x >> shift) ^ y) & mask;
   x ^= swap << shift;
   y ^= swap;
   }

}

namespace Botan {

/**
* Perform encoding using the base provided
* @param base object giving access to the encodings specifications
* @param output an array of at least base.encode_max_output bytes
* @param input is some binary data
* @param input_length length of input in bytes
* @param input_consumed is an output parameter which says how many
*        bytes of input were actually consumed. If less than
*        input_length, then the range input[consumed:length]
*        should be passed in later along with more input.
* @param final_inputs true iff this is the last input, in which case
         padding chars will be applied if needed
* @return number of bytes written to output
*/
template <class Base>
size_t base_encode(Base&& base,
                   char output[],
                   const uint8_t input[],
                   size_t input_length,
                   size_t& input_consumed,
                   bool final_inputs)
   {
   input_consumed = 0;

   const size_t encoding_bytes_in = base.encoding_bytes_in();
   const size_t encoding_bytes_out = base.encoding_bytes_out();

   size_t input_remaining = input_length;
   size_t output_produced = 0;

   while(input_remaining >= encoding_bytes_in)
      {
      base.encode(output + output_produced, input + input_consumed);

      input_consumed += encoding_bytes_in;
      output_produced += encoding_bytes_out;
      input_remaining -= encoding_bytes_in;
      }

   if(final_inputs && input_remaining)
      {
      std::vector<uint8_t> remainder(encoding_bytes_in, 0);
      for(size_t i = 0; i != input_remaining; ++i)
         { remainder[i] = input[input_consumed + i]; }

      base.encode(output + output_produced, remainder.data());

      const size_t bits_consumed = base.bits_consumed();
      const size_t remaining_bits_before_padding = base.remaining_bits_before_padding();

      size_t empty_bits = 8 * (encoding_bytes_in - input_remaining);
      size_t index = output_produced + encoding_bytes_out - 1;
      while(empty_bits >= remaining_bits_before_padding)
         {
         output[index--] = '=';
         empty_bits -= bits_consumed;
         }

      input_consumed += input_remaining;
      output_produced += encoding_bytes_out;
      }

   return output_produced;
   }


template <typename Base>
std::string base_encode_to_string(Base&& base, const uint8_t input[], size_t input_length)
   {
   const size_t output_length = base.encode_max_output(input_length);
   std::string output(output_length, 0);

   size_t consumed = 0;
   size_t produced = 0;

   if(output_length > 0)
      {
      produced = base_encode(base, &output.front(),
                                   input, input_length,
                                   consumed, true);
      }

   BOTAN_ASSERT_EQUAL(consumed, input_length, "Consumed the entire input");
   BOTAN_ASSERT_EQUAL(produced, output.size(), "Produced expected size");

   return output;
   }

/**
* Perform decoding using the base provided
* @param base object giving access to the encodings specifications
* @param output an array of at least Base::decode_max_output bytes
* @param input some base input
* @param input_length length of input in bytes
* @param input_consumed is an output parameter which says how many
*        bytes of input were actually consumed. If less than
*        input_length, then the range input[consumed:length]
*        should be passed in later along with more input.
* @param final_inputs true iff this is the last input, in which case
         padding is allowed
* @param ignore_ws ignore whitespace on input; if false, throw an
                   exception if whitespace is encountered
* @return number of bytes written to output
*/
template <typename Base>
size_t base_decode(Base&& base,
                   uint8_t output[],
                   const char input[],
                   size_t input_length,
                   size_t& input_consumed,
                   bool final_inputs,
                   bool ignore_ws = true)
   {
   const size_t decoding_bytes_in = base.decoding_bytes_in();
   const size_t decoding_bytes_out = base.decoding_bytes_out();

   uint8_t* out_ptr = output;
   std::vector<uint8_t> decode_buf(decoding_bytes_in, 0);
   size_t decode_buf_pos = 0;
   size_t final_truncate = 0;

   clear_mem(output, base.decode_max_output(input_length));

   for(size_t i = 0; i != input_length; ++i)
      {
      const uint8_t bin = base.lookup_binary_value(input[i]);

      if(base.check_bad_char(bin, input[i], ignore_ws)) // May throw Invalid_Argument
         {
         decode_buf[decode_buf_pos] = bin;
         ++decode_buf_pos;
         }

      /*
      * If we're at the end of the input, pad with 0s and truncate
      */
      if(final_inputs && (i == input_length - 1))
         {
         if(decode_buf_pos)
            {
            for(size_t j = decode_buf_pos; j < decoding_bytes_in; ++j)
               { decode_buf[j] = 0; }

            final_truncate = decoding_bytes_in - decode_buf_pos;
            decode_buf_pos = decoding_bytes_in;
            }
         }

      if(decode_buf_pos == decoding_bytes_in)
         {
         base.decode(out_ptr, decode_buf.data());

         out_ptr += decoding_bytes_out;
         decode_buf_pos = 0;
         input_consumed = i+1;
         }
      }

   while(input_consumed < input_length &&
         base.lookup_binary_value(input[input_consumed]) == 0x80)
      {
      ++input_consumed;
      }

   size_t written = (out_ptr - output) - base.bytes_to_remove(final_truncate);

   return written;
   }

template<typename Base>
size_t base_decode_full(Base&& base, uint8_t output[], const char input[], size_t input_length, bool ignore_ws)
   {
   size_t consumed = 0;
   const size_t written = base_decode(base, output, input, input_length, consumed, true, ignore_ws);

   if(consumed != input_length)
      {
      throw Invalid_Argument(base.name() + " decoding failed, input did not have full bytes");
      }

   return written;
   }

template<typename Vector, typename Base>
Vector base_decode_to_vec(Base&& base,
                          const char input[],
                          size_t input_length,
                          bool ignore_ws)
   {
   const size_t output_length = base.decode_max_output(input_length);
   Vector bin(output_length);

   const size_t written =
      base_decode_full(base, bin.data(), input, input_length, ignore_ws);

   bin.resize(written);
   return bin;
   }

}

#if defined(BOTAN_HAS_VALGRIND)
  #include <valgrind/memcheck.h>
#endif

namespace Botan {

namespace CT {

/**
* Use valgrind to mark the contents of memory as being undefined.
* Valgrind will accept operations which manipulate undefined values,
* but will warn if an undefined value is used to decided a conditional
* jump or a load/store address. So if we poison all of our inputs we
* can confirm that the operations in question are truly const time
* when compiled by whatever compiler is in use.
*
* Even better, the VALGRIND_MAKE_MEM_* macros work even when the
* program is not run under valgrind (though with a few cycles of
* overhead, which is unfortunate in final binaries as these
* annotations tend to be used in fairly important loops).
*
* This approach was first used in ctgrind (https://github.com/agl/ctgrind)
* but calling the valgrind mecheck API directly works just as well and
* doesn't require a custom patched valgrind.
*/
template<typename T>
inline void poison(const T* p, size_t n)
   {
#if defined(BOTAN_HAS_VALGRIND)
   VALGRIND_MAKE_MEM_UNDEFINED(p, n * sizeof(T));
#else
   BOTAN_UNUSED(p);
   BOTAN_UNUSED(n);
#endif
   }

template<typename T>
inline void unpoison(const T* p, size_t n)
   {
#if defined(BOTAN_HAS_VALGRIND)
   VALGRIND_MAKE_MEM_DEFINED(p, n * sizeof(T));
#else
   BOTAN_UNUSED(p);
   BOTAN_UNUSED(n);
#endif
   }

template<typename T>
inline void unpoison(T& p)
   {
#if defined(BOTAN_HAS_VALGRIND)
   VALGRIND_MAKE_MEM_DEFINED(&p, sizeof(T));
#else
   BOTAN_UNUSED(p);
#endif
   }

/**
* A Mask type used for constant-time operations. A Mask<T> always has value
* either 0 (all bits cleared) or ~0 (all bits set). All operations in a Mask<T>
* are intended to compile to code which does not contain conditional jumps.
* This must be verified with tooling (eg binary disassembly or using valgrind)
* since you never know what a compiler might do.
*/
template<typename T>
class Mask
   {
   public:
      static_assert(std::is_unsigned<T>::value, "CT::Mask only defined for unsigned integer types");

      Mask(const Mask<T>& other) = default;
      Mask<T>& operator=(const Mask<T>& other) = default;

      /**
      * Derive a Mask from a Mask of a larger type
      */
      template<typename U>
      Mask(Mask<U> o) : m_mask(static_cast<T>(o.value()))
         {
         static_assert(sizeof(U) > sizeof(T), "sizes ok");
         }

      /**
      * Return a Mask<T> with all bits set
      */
      static Mask<T> set()
         {
         return Mask<T>(static_cast<T>(~0));
         }

      /**
      * Return a Mask<T> with all bits cleared
      */
      static Mask<T> cleared()
         {
         return Mask<T>(0);
         }

      /**
      * Return a Mask<T> which is set if v is != 0
      */
      static Mask<T> expand(T v)
         {
         return ~Mask<T>::is_zero(v);
         }

      /**
      * Return a Mask<T> which is set if m is set
      */
      template<typename U>
      static Mask<T> expand(Mask<U> m)
         {
         static_assert(sizeof(U) < sizeof(T), "sizes ok");
         return ~Mask<T>::is_zero(m.value());
         }

      /**
      * Return a Mask<T> which is set if v is == 0 or cleared otherwise
      */
      static Mask<T> is_zero(T x)
         {
         return Mask<T>(ct_is_zero<T>(x));
         }

      /**
      * Return a Mask<T> which is set if x == y
      */
      static Mask<T> is_equal(T x, T y)
         {
         return Mask<T>::is_zero(static_cast<T>(x ^ y));
         }

      /**
      * Return a Mask<T> which is set if x < y
      */
      static Mask<T> is_lt(T x, T y)
         {
         return Mask<T>(expand_top_bit<T>(x^((x^y) | ((x-y)^x))));
         }

      /**
      * Return a Mask<T> which is set if x > y
      */
      static Mask<T> is_gt(T x, T y)
         {
         return Mask<T>::is_lt(y, x);
         }

      /**
      * Return a Mask<T> which is set if x <= y
      */
      static Mask<T> is_lte(T x, T y)
         {
         return ~Mask<T>::is_gt(x, y);
         }

      /**
      * Return a Mask<T> which is set if x >= y
      */
      static Mask<T> is_gte(T x, T y)
         {
         return ~Mask<T>::is_lt(x, y);
         }

      static Mask<T> is_within_range(T v, T l, T u)
         {
         //return Mask<T>::is_gte(v, l) & Mask<T>::is_lte(v, u);

         const T v_lt_l = v^((v^l) | ((v-l)^v));
         const T v_gt_u = u^((u^v) | ((u-v)^u));
         const T either = v_lt_l | v_gt_u;
         return ~Mask<T>(expand_top_bit(either));
         }

      static Mask<T> is_any_of(T v, std::initializer_list<T> accepted)
         {
         T accept = 0;

         for(auto a: accepted)
            {
            const T diff = a ^ v;
            const T eq_zero = ~diff & (diff - 1);
            accept |= eq_zero;
            }

         return Mask<T>(expand_top_bit(accept));
         }

      /**
      * AND-combine two masks
      */
      Mask<T>& operator&=(Mask<T> o)
         {
         m_mask &= o.value();
         return (*this);
         }

      /**
      * XOR-combine two masks
      */
      Mask<T>& operator^=(Mask<T> o)
         {
         m_mask ^= o.value();
         return (*this);
         }

      /**
      * OR-combine two masks
      */
      Mask<T>& operator|=(Mask<T> o)
         {
         m_mask |= o.value();
         return (*this);
         }

      /**
      * AND-combine two masks
      */
      friend Mask<T> operator&(Mask<T> x, Mask<T> y)
         {
         return Mask<T>(x.value() & y.value());
         }

      /**
      * XOR-combine two masks
      */
      friend Mask<T> operator^(Mask<T> x, Mask<T> y)
         {
         return Mask<T>(x.value() ^ y.value());
         }

      /**
      * OR-combine two masks
      */
      friend Mask<T> operator|(Mask<T> x, Mask<T> y)
         {
         return Mask<T>(x.value() | y.value());
         }

      /**
      * Negate this mask
      */
      Mask<T> operator~() const
         {
         return Mask<T>(~value());
         }

      /**
      * Return x if the mask is set, or otherwise zero
      */
      T if_set_return(T x) const
         {
         return m_mask & x;
         }

      /**
      * Return x if the mask is cleared, or otherwise zero
      */
      T if_not_set_return(T x) const
         {
         return ~m_mask & x;
         }

      /**
      * If this mask is set, return x, otherwise return y
      */
      T select(T x, T y) const
         {
         // (x & value()) | (y & ~value())
         return static_cast<T>(y ^ (value() & (x ^ y)));
         }

      T select_and_unpoison(T x, T y) const
         {
         T r = this->select(x, y);
         CT::unpoison(r);
         return r;
         }

      /**
      * If this mask is set, return x, otherwise return y
      */
      Mask<T> select_mask(Mask<T> x, Mask<T> y) const
         {
         return Mask<T>(select(x.value(), y.value()));
         }

      /**
      * Conditionally set output to x or y, depending on if mask is set or
      * cleared (resp)
      */
      void select_n(T output[], const T x[], const T y[], size_t len) const
         {
         for(size_t i = 0; i != len; ++i)
            output[i] = this->select(x[i], y[i]);
         }

      /**
      * If this mask is set, zero out buf, otherwise do nothing
      */
      void if_set_zero_out(T buf[], size_t elems)
         {
         for(size_t i = 0; i != elems; ++i)
            {
            buf[i] = this->if_not_set_return(buf[i]);
            }
         }

      /**
      * Return the value of the mask, unpoisoned
      */
      T unpoisoned_value() const
         {
         T r = value();
         CT::unpoison(r);
         return r;
         }

      /**
      * Return true iff this mask is set
      */
      bool is_set() const
         {
         return unpoisoned_value() != 0;
         }

      /**
      * Return the underlying value of the mask
      */
      T value() const
         {
         return m_mask;
         }

   private:
      Mask(T m) : m_mask(m) {}

      T m_mask;
   };

template<typename T>
inline Mask<T> conditional_copy_mem(T cnd,
                                    T* to,
                                    const T* from0,
                                    const T* from1,
                                    size_t elems)
   {
   const auto mask = CT::Mask<T>::expand(cnd);
   mask.select_n(to, from0, from1, elems);
   return mask;
   }

template<typename T>
inline void conditional_swap(bool cnd, T& x, T& y)
   {
   const auto swap = CT::Mask<T>::expand(cnd);

   T t0 = swap.select(y, x);
   T t1 = swap.select(x, y);
   x = t0;
   y = t1;
   }

template<typename T>
inline void conditional_swap_ptr(bool cnd, T& x, T& y)
   {
   uintptr_t xp = reinterpret_cast<uintptr_t>(x);
   uintptr_t yp = reinterpret_cast<uintptr_t>(y);

   conditional_swap<uintptr_t>(cnd, xp, yp);

   x = reinterpret_cast<T>(xp);
   y = reinterpret_cast<T>(yp);
   }

/**
* If bad_mask is unset, return in[delim_idx:input_length] copied to
* new buffer. If bad_mask is set, return an all zero vector of
* unspecified length.
*/
secure_vector<uint8_t> copy_output(CT::Mask<uint8_t> bad_input,
                                   const uint8_t input[],
                                   size_t input_length,
                                   size_t delim_idx);

secure_vector<uint8_t> strip_leading_zeros(const uint8_t in[], size_t length);

inline secure_vector<uint8_t> strip_leading_zeros(const secure_vector<uint8_t>& in)
   {
   return strip_leading_zeros(in.data(), in.size());
   }

}

}

namespace Botan {

class donna128 final
   {
   public:
      donna128(uint64_t ll = 0, uint64_t hh = 0) { l = ll; h = hh; }

      donna128(const donna128&) = default;
      donna128& operator=(const donna128&) = default;

      friend donna128 operator>>(const donna128& x, size_t shift)
         {
         donna128 z = x;
         if(shift > 0)
            {
            const uint64_t carry = z.h << (64 - shift);
            z.h = (z.h >> shift);
            z.l = (z.l >> shift) | carry;
            }
         return z;
         }

      friend donna128 operator<<(const donna128& x, size_t shift)
         {
         donna128 z = x;
         if(shift > 0)
            {
            const uint64_t carry = z.l >> (64 - shift);
            z.l = (z.l << shift);
            z.h = (z.h << shift) | carry;
            }
         return z;
         }

      friend uint64_t operator&(const donna128& x, uint64_t mask)
         {
         return x.l & mask;
         }

      uint64_t operator&=(uint64_t mask)
         {
         h = 0;
         l &= mask;
         return l;
         }

      donna128& operator+=(const donna128& x)
         {
         l += x.l;
         h += x.h;

         const uint64_t carry = (l < x.l);
         h += carry;
         return *this;
         }

      donna128& operator+=(uint64_t x)
         {
         l += x;
         const uint64_t carry = (l < x);
         h += carry;
         return *this;
         }

      uint64_t lo() const { return l; }
      uint64_t hi() const { return h; }
   private:
      uint64_t h = 0, l = 0;
   };

inline donna128 operator*(const donna128& x, uint64_t y)
   {
   BOTAN_ARG_CHECK(x.hi() == 0, "High 64 bits of donna128 set to zero during multiply");

   uint64_t lo = 0, hi = 0;
   mul64x64_128(x.lo(), y, &lo, &hi);
   return donna128(lo, hi);
   }

inline donna128 operator*(uint64_t y, const donna128& x)
   {
   return x * y;
   }

inline donna128 operator+(const donna128& x, const donna128& y)
   {
   donna128 z = x;
   z += y;
   return z;
   }

inline donna128 operator+(const donna128& x, uint64_t y)
   {
   donna128 z = x;
   z += y;
   return z;
   }

inline donna128 operator|(const donna128& x, const donna128& y)
   {
   return donna128(x.lo() | y.lo(), x.hi() | y.hi());
   }

inline uint64_t carry_shift(const donna128& a, size_t shift)
   {
   return (a >> shift).lo();
   }

inline uint64_t combine_lower(const donna128& a, size_t s1,
                              const donna128& b, size_t s2)
   {
   donna128 z = (a >> s1) | (b << s2);
   return z.lo();
   }

#if defined(BOTAN_TARGET_HAS_NATIVE_UINT128)
inline uint64_t carry_shift(const uint128_t a, size_t shift)
   {
   return static_cast<uint64_t>(a >> shift);
   }

inline uint64_t combine_lower(const uint128_t a, size_t s1,
                              const uint128_t b, size_t s2)
   {
   return static_cast<uint64_t>((a >> s1) | (b << s2));
   }
#endif

}

namespace Botan {

/**
* No_Filesystem_Access Exception
*/
class BOTAN_PUBLIC_API(2,0) No_Filesystem_Access final : public Exception
   {
   public:
      No_Filesystem_Access() : Exception("No filesystem access enabled.")
         {}
   };

BOTAN_TEST_API bool has_filesystem_impl();

BOTAN_TEST_API std::vector<std::string> get_files_recursive(const std::string& dir);

}

namespace Botan {

class BigInt;
class Modular_Reducer;

class Montgomery_Params;

class Montgomery_Exponentation_State;

/*
* Precompute for calculating values g^x mod p
*/
std::shared_ptr<const Montgomery_Exponentation_State>
monty_precompute(std::shared_ptr<const Montgomery_Params> params_p,
                 const BigInt& g,
                 size_t window_bits,
                 bool const_time = true);

/*
* Return g^k mod p
*/
BigInt monty_execute(const Montgomery_Exponentation_State& precomputed_state,
                     const BigInt& k, size_t max_k_bits);

/*
* Return g^k mod p taking variable time depending on k
* @warning only use this if k is public
*/
BigInt monty_execute_vartime(const Montgomery_Exponentation_State& precomputed_state,
                             const BigInt& k);

/**
* Return (x^z1 * y^z2) % p
*/
BigInt monty_multi_exp(std::shared_ptr<const Montgomery_Params> params_p,
                       const BigInt& x,
                       const BigInt& z1,
                       const BigInt& y,
                       const BigInt& z2);

}

namespace Botan {

#if (BOTAN_MP_WORD_BITS == 32)
  typedef uint64_t dword;
  #define BOTAN_HAS_MP_DWORD

#elif (BOTAN_MP_WORD_BITS == 64)
  #if defined(BOTAN_TARGET_HAS_NATIVE_UINT128)
    typedef uint128_t dword;
    #define BOTAN_HAS_MP_DWORD
  #else
    // No native 128 bit integer type; use mul64x64_128 instead
  #endif

#else
  #error BOTAN_MP_WORD_BITS must be 32 or 64
#endif

#if defined(BOTAN_USE_GCC_INLINE_ASM)

  #if defined(BOTAN_TARGET_ARCH_IS_X86_32) && (BOTAN_MP_WORD_BITS == 32)
    #define BOTAN_MP_USE_X86_32_ASM
  #elif defined(BOTAN_TARGET_ARCH_IS_X86_64) && (BOTAN_MP_WORD_BITS == 64)
    #define BOTAN_MP_USE_X86_64_ASM
  #endif

#endif

/*
* Word Multiply/Add
*/
inline word word_madd2(word a, word b, word* c)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(R"(
      mull %[b]
      addl %[c],%[a]
      adcl $0,%[carry]
      )"
      : [a]"=a"(a), [b]"=rm"(b), [carry]"=&d"(*c)
      : "0"(a), "1"(b), [c]"g"(*c) : "cc");

   return a;

#elif defined(BOTAN_MP_USE_X86_64_ASM)
      asm(R"(
         mulq %[b]
         addq %[c],%[a]
         adcq $0,%[carry]
      )"
      : [a]"=a"(a), [b]"=rm"(b), [carry]"=&d"(*c)
      : "0"(a), "1"(b), [c]"g"(*c) : "cc");

   return a;

#elif defined(BOTAN_HAS_MP_DWORD)
   const dword s = static_cast<dword>(a) * b + *c;
   *c = static_cast<word>(s >> BOTAN_MP_WORD_BITS);
   return static_cast<word>(s);
#else
   static_assert(BOTAN_MP_WORD_BITS == 64, "Unexpected word size");

   word hi = 0, lo = 0;

   mul64x64_128(a, b, &lo, &hi);

   lo += *c;
   hi += (lo < *c); // carry?

   *c = hi;
   return lo;
#endif
   }

/*
* Word Multiply/Add
*/
inline word word_madd3(word a, word b, word c, word* d)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(R"(
      mull %[b]

      addl %[c],%[a]
      adcl $0,%[carry]

      addl %[d],%[a]
      adcl $0,%[carry]
      )"
      : [a]"=a"(a), [b]"=rm"(b), [carry]"=&d"(*d)
      : "0"(a), "1"(b), [c]"g"(c), [d]"g"(*d) : "cc");

   return a;

#elif defined(BOTAN_MP_USE_X86_64_ASM)
   asm(R"(
      mulq %[b]
      addq %[c],%[a]
      adcq $0,%[carry]
      addq %[d],%[a]
      adcq $0,%[carry]
      )"
      : [a]"=a"(a), [b]"=rm"(b), [carry]"=&d"(*d)
      : "0"(a), "1"(b), [c]"g"(c), [d]"g"(*d) : "cc");

   return a;

#elif defined(BOTAN_HAS_MP_DWORD)
   const dword s = static_cast<dword>(a) * b + c + *d;
   *d = static_cast<word>(s >> BOTAN_MP_WORD_BITS);
   return static_cast<word>(s);
#else
   static_assert(BOTAN_MP_WORD_BITS == 64, "Unexpected word size");

   word hi = 0, lo = 0;

   mul64x64_128(a, b, &lo, &hi);

   lo += c;
   hi += (lo < c); // carry?

   lo += *d;
   hi += (lo < *d); // carry?

   *d = hi;
   return lo;
#endif
   }

}

namespace Botan {

#if defined(BOTAN_MP_USE_X86_32_ASM)

#define ADDSUB2_OP(OPERATION, INDEX)                     \
        ASM("movl 4*" #INDEX "(%[y]), %[carry]")         \
        ASM(OPERATION " %[carry], 4*" #INDEX "(%[x])")   \

#define ADDSUB3_OP(OPERATION, INDEX)                     \
        ASM("movl 4*" #INDEX "(%[x]), %[carry]")         \
        ASM(OPERATION " 4*" #INDEX "(%[y]), %[carry]")   \
        ASM("movl %[carry], 4*" #INDEX "(%[z])")         \

#define LINMUL_OP(WRITE_TO, INDEX)                       \
        ASM("movl 4*" #INDEX "(%[x]),%%eax")             \
        ASM("mull %[y]")                                 \
        ASM("addl %[carry],%%eax")                       \
        ASM("adcl $0,%%edx")                             \
        ASM("movl %%edx,%[carry]")                       \
        ASM("movl %%eax, 4*" #INDEX "(%[" WRITE_TO "])")

#define MULADD_OP(IGNORED, INDEX)                        \
        ASM("movl 4*" #INDEX "(%[x]),%%eax")             \
        ASM("mull %[y]")                                 \
        ASM("addl %[carry],%%eax")                       \
        ASM("adcl $0,%%edx")                             \
        ASM("addl 4*" #INDEX "(%[z]),%%eax")             \
        ASM("adcl $0,%%edx")                             \
        ASM("movl %%edx,%[carry]")                       \
        ASM("movl %%eax, 4*" #INDEX " (%[z])")

#define ADD_OR_SUBTRACT(CORE_CODE)     \
        ASM("rorl %[carry]")           \
        CORE_CODE                      \
        ASM("sbbl %[carry],%[carry]")  \
        ASM("negl %[carry]")

#elif defined(BOTAN_MP_USE_X86_64_ASM)

#define ADDSUB2_OP(OPERATION, INDEX)                     \
        ASM("movq 8*" #INDEX "(%[y]), %[carry]")         \
        ASM(OPERATION " %[carry], 8*" #INDEX "(%[x])")   \

#define ADDSUB3_OP(OPERATION, INDEX)                     \
        ASM("movq 8*" #INDEX "(%[x]), %[carry]")         \
        ASM(OPERATION " 8*" #INDEX "(%[y]), %[carry]")   \
        ASM("movq %[carry], 8*" #INDEX "(%[z])")         \

#define LINMUL_OP(WRITE_TO, INDEX)                       \
        ASM("movq 8*" #INDEX "(%[x]),%%rax")             \
        ASM("mulq %[y]")                                 \
        ASM("addq %[carry],%%rax")                       \
        ASM("adcq $0,%%rdx")                             \
        ASM("movq %%rdx,%[carry]")                       \
        ASM("movq %%rax, 8*" #INDEX "(%[" WRITE_TO "])")

#define MULADD_OP(IGNORED, INDEX)                        \
        ASM("movq 8*" #INDEX "(%[x]),%%rax")             \
        ASM("mulq %[y]")                                 \
        ASM("addq %[carry],%%rax")                       \
        ASM("adcq $0,%%rdx")                             \
        ASM("addq 8*" #INDEX "(%[z]),%%rax")             \
        ASM("adcq $0,%%rdx")                             \
        ASM("movq %%rdx,%[carry]")                       \
        ASM("movq %%rax, 8*" #INDEX " (%[z])")

#define ADD_OR_SUBTRACT(CORE_CODE)     \
        ASM("rorq %[carry]")           \
        CORE_CODE                      \
        ASM("sbbq %[carry],%[carry]")  \
        ASM("negq %[carry]")

#endif

#if defined(ADD_OR_SUBTRACT)

#define ASM(x) x "\n\t"

#define DO_8_TIMES(MACRO, ARG) \
        MACRO(ARG, 0) \
        MACRO(ARG, 1) \
        MACRO(ARG, 2) \
        MACRO(ARG, 3) \
        MACRO(ARG, 4) \
        MACRO(ARG, 5) \
        MACRO(ARG, 6) \
        MACRO(ARG, 7)

#endif

/*
* Word Addition
*/
inline word word_add(word x, word y, word* carry)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(
      ADD_OR_SUBTRACT(ASM("adcl %[y],%[x]"))
      : [x]"=r"(x), [carry]"=r"(*carry)
      : "0"(x), [y]"rm"(y), "1"(*carry)
      : "cc");
   return x;

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   asm(
      ADD_OR_SUBTRACT(ASM("adcq %[y],%[x]"))
      : [x]"=r"(x), [carry]"=r"(*carry)
      : "0"(x), [y]"rm"(y), "1"(*carry)
      : "cc");
   return x;

#else
   word z = x + y;
   word c1 = (z < x);
   z += *carry;
   *carry = c1 | (z < *carry);
   return z;
#endif
   }

/*
* Eight Word Block Addition, Two Argument
*/
inline word word8_add2(word x[8], const word y[8], word carry)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(
      ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB2_OP, "adcl"))
      : [carry]"=r"(carry)
      : [x]"r"(x), [y]"r"(y), "0"(carry)
      : "cc", "memory");
   return carry;

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   asm(
      ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB2_OP, "adcq"))
      : [carry]"=r"(carry)
      : [x]"r"(x), [y]"r"(y), "0"(carry)
      : "cc", "memory");
   return carry;

#else
   x[0] = word_add(x[0], y[0], &carry);
   x[1] = word_add(x[1], y[1], &carry);
   x[2] = word_add(x[2], y[2], &carry);
   x[3] = word_add(x[3], y[3], &carry);
   x[4] = word_add(x[4], y[4], &carry);
   x[5] = word_add(x[5], y[5], &carry);
   x[6] = word_add(x[6], y[6], &carry);
   x[7] = word_add(x[7], y[7], &carry);
   return carry;
#endif
   }

/*
* Eight Word Block Addition, Three Argument
*/
inline word word8_add3(word z[8], const word x[8],
                       const word y[8], word carry)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(
      ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB3_OP, "adcl"))
      : [carry]"=r"(carry)
      : [x]"r"(x), [y]"r"(y), [z]"r"(z), "0"(carry)
      : "cc", "memory");
   return carry;

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   asm(
      ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB3_OP, "adcq"))
      : [carry]"=r"(carry)
      : [x]"r"(x), [y]"r"(y), [z]"r"(z), "0"(carry)
      : "cc", "memory");
   return carry;

#else
   z[0] = word_add(x[0], y[0], &carry);
   z[1] = word_add(x[1], y[1], &carry);
   z[2] = word_add(x[2], y[2], &carry);
   z[3] = word_add(x[3], y[3], &carry);
   z[4] = word_add(x[4], y[4], &carry);
   z[5] = word_add(x[5], y[5], &carry);
   z[6] = word_add(x[6], y[6], &carry);
   z[7] = word_add(x[7], y[7], &carry);
   return carry;
#endif
   }

/*
* Word Subtraction
*/
inline word word_sub(word x, word y, word* carry)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(
      ADD_OR_SUBTRACT(ASM("sbbl %[y],%[x]"))
      : [x]"=r"(x), [carry]"=r"(*carry)
      : "0"(x), [y]"rm"(y), "1"(*carry)
      : "cc");
   return x;

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   asm(
      ADD_OR_SUBTRACT(ASM("sbbq %[y],%[x]"))
      : [x]"=r"(x), [carry]"=r"(*carry)
      : "0"(x), [y]"rm"(y), "1"(*carry)
      : "cc");
   return x;

#else
   word t0 = x - y;
   word c1 = (t0 > x);
   word z = t0 - *carry;
   *carry = c1 | (z > t0);
   return z;
#endif
   }

/*
* Eight Word Block Subtraction, Two Argument
*/
inline word word8_sub2(word x[8], const word y[8], word carry)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(
      ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB2_OP, "sbbl"))
      : [carry]"=r"(carry)
      : [x]"r"(x), [y]"r"(y), "0"(carry)
      : "cc", "memory");
   return carry;

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   asm(
      ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB2_OP, "sbbq"))
      : [carry]"=r"(carry)
      : [x]"r"(x), [y]"r"(y), "0"(carry)
      : "cc", "memory");
   return carry;

#else
   x[0] = word_sub(x[0], y[0], &carry);
   x[1] = word_sub(x[1], y[1], &carry);
   x[2] = word_sub(x[2], y[2], &carry);
   x[3] = word_sub(x[3], y[3], &carry);
   x[4] = word_sub(x[4], y[4], &carry);
   x[5] = word_sub(x[5], y[5], &carry);
   x[6] = word_sub(x[6], y[6], &carry);
   x[7] = word_sub(x[7], y[7], &carry);
   return carry;
#endif
   }

/*
* Eight Word Block Subtraction, Two Argument
*/
inline word word8_sub2_rev(word x[8], const word y[8], word carry)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(
      ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB3_OP, "sbbl"))
      : [carry]"=r"(carry)
      : [x]"r"(y), [y]"r"(x), [z]"r"(x), "0"(carry)
      : "cc", "memory");
   return carry;

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   asm(
      ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB3_OP, "sbbq"))
      : [carry]"=r"(carry)
      : [x]"r"(y), [y]"r"(x), [z]"r"(x), "0"(carry)
      : "cc", "memory");
   return carry;

#else
   x[0] = word_sub(y[0], x[0], &carry);
   x[1] = word_sub(y[1], x[1], &carry);
   x[2] = word_sub(y[2], x[2], &carry);
   x[3] = word_sub(y[3], x[3], &carry);
   x[4] = word_sub(y[4], x[4], &carry);
   x[5] = word_sub(y[5], x[5], &carry);
   x[6] = word_sub(y[6], x[6], &carry);
   x[7] = word_sub(y[7], x[7], &carry);
   return carry;
#endif
   }

/*
* Eight Word Block Subtraction, Three Argument
*/
inline word word8_sub3(word z[8], const word x[8],
                       const word y[8], word carry)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(
      ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB3_OP, "sbbl"))
      : [carry]"=r"(carry)
      : [x]"r"(x), [y]"r"(y), [z]"r"(z), "0"(carry)
      : "cc", "memory");
   return carry;

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   asm(
      ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB3_OP, "sbbq"))
      : [carry]"=r"(carry)
      : [x]"r"(x), [y]"r"(y), [z]"r"(z), "0"(carry)
      : "cc", "memory");
   return carry;

#else
   z[0] = word_sub(x[0], y[0], &carry);
   z[1] = word_sub(x[1], y[1], &carry);
   z[2] = word_sub(x[2], y[2], &carry);
   z[3] = word_sub(x[3], y[3], &carry);
   z[4] = word_sub(x[4], y[4], &carry);
   z[5] = word_sub(x[5], y[5], &carry);
   z[6] = word_sub(x[6], y[6], &carry);
   z[7] = word_sub(x[7], y[7], &carry);
   return carry;
#endif
   }

/*
* Eight Word Block Linear Multiplication
*/
inline word word8_linmul2(word x[8], word y, word carry)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(
      DO_8_TIMES(LINMUL_OP, "x")
      : [carry]"=r"(carry)
      : [x]"r"(x), [y]"rm"(y), "0"(carry)
      : "cc", "%eax", "%edx");
   return carry;

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   asm(
      DO_8_TIMES(LINMUL_OP, "x")
      : [carry]"=r"(carry)
      : [x]"r"(x), [y]"rm"(y), "0"(carry)
      : "cc", "%rax", "%rdx");
   return carry;

#else
   x[0] = word_madd2(x[0], y, &carry);
   x[1] = word_madd2(x[1], y, &carry);
   x[2] = word_madd2(x[2], y, &carry);
   x[3] = word_madd2(x[3], y, &carry);
   x[4] = word_madd2(x[4], y, &carry);
   x[5] = word_madd2(x[5], y, &carry);
   x[6] = word_madd2(x[6], y, &carry);
   x[7] = word_madd2(x[7], y, &carry);
   return carry;
#endif
   }

/*
* Eight Word Block Linear Multiplication
*/
inline word word8_linmul3(word z[8], const word x[8], word y, word carry)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(
      DO_8_TIMES(LINMUL_OP, "z")
      : [carry]"=r"(carry)
      : [z]"r"(z), [x]"r"(x), [y]"rm"(y), "0"(carry)
      : "cc", "%eax", "%edx");
   return carry;

#elif defined(BOTAN_MP_USE_X86_64_ASM)
   asm(
      DO_8_TIMES(LINMUL_OP, "z")
      : [carry]"=r"(carry)
      : [z]"r"(z), [x]"r"(x), [y]"rm"(y), "0"(carry)
      : "cc", "%rax", "%rdx");
   return carry;

#else
   z[0] = word_madd2(x[0], y, &carry);
   z[1] = word_madd2(x[1], y, &carry);
   z[2] = word_madd2(x[2], y, &carry);
   z[3] = word_madd2(x[3], y, &carry);
   z[4] = word_madd2(x[4], y, &carry);
   z[5] = word_madd2(x[5], y, &carry);
   z[6] = word_madd2(x[6], y, &carry);
   z[7] = word_madd2(x[7], y, &carry);
   return carry;
#endif
   }

/*
* Eight Word Block Multiply/Add
*/
inline word word8_madd3(word z[8], const word x[8], word y, word carry)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(
      DO_8_TIMES(MULADD_OP, "")
      : [carry]"=r"(carry)
      : [z]"r"(z), [x]"r"(x), [y]"rm"(y), "0"(carry)
      : "cc", "%eax", "%edx");
   return carry;

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   asm(
      DO_8_TIMES(MULADD_OP, "")
      : [carry]"=r"(carry)
      : [z]"r"(z), [x]"r"(x), [y]"rm"(y), "0"(carry)
      : "cc", "%rax", "%rdx");
   return carry;

#else
   z[0] = word_madd3(x[0], y, z[0], &carry);
   z[1] = word_madd3(x[1], y, z[1], &carry);
   z[2] = word_madd3(x[2], y, z[2], &carry);
   z[3] = word_madd3(x[3], y, z[3], &carry);
   z[4] = word_madd3(x[4], y, z[4], &carry);
   z[5] = word_madd3(x[5], y, z[5], &carry);
   z[6] = word_madd3(x[6], y, z[6], &carry);
   z[7] = word_madd3(x[7], y, z[7], &carry);
   return carry;
#endif
   }

/*
* Multiply-Add Accumulator
* (w2,w1,w0) += x * y
*/
inline void word3_muladd(word* w2, word* w1, word* w0, word x, word y)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   word z0 = 0, z1 = 0;

   asm("mull %[y]"
        : "=a"(z0),"=d"(z1)
        : "a"(x), [y]"rm"(y)
        : "cc");

   asm(R"(
       addl %[z0],%[w0]
       adcl %[z1],%[w1]
       adcl $0,%[w2]
       )"
       : [w0]"=r"(*w0), [w1]"=r"(*w1), [w2]"=r"(*w2)
       : [z0]"r"(z0), [z1]"r"(z1), "0"(*w0), "1"(*w1), "2"(*w2)
       : "cc");

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   word z0 = 0, z1 = 0;

   asm("mulq %[y]"
        : "=a"(z0),"=d"(z1)
        : "a"(x), [y]"rm"(y)
        : "cc");

   asm(R"(
       addq %[z0],%[w0]
       adcq %[z1],%[w1]
       adcq $0,%[w2]
       )"
       : [w0]"=r"(*w0), [w1]"=r"(*w1), [w2]"=r"(*w2)
       : [z0]"r"(z0), [z1]"r"(z1), "0"(*w0), "1"(*w1), "2"(*w2)
       : "cc");

#else
   word carry = *w0;
   *w0 = word_madd2(x, y, &carry);
   *w1 += carry;
   *w2 += (*w1 < carry);
#endif
   }

/*
* 3-word addition
* (w2,w1,w0) += x
*/
inline void word3_add(word* w2, word* w1, word* w0, word x)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)
   asm(R"(
      addl %[x],%[w0]
      adcl $0,%[w1]
      adcl $0,%[w2]
      )"
      : [w0]"=r"(*w0), [w1]"=r"(*w1), [w2]"=r"(*w2)
      : [x]"r"(x), "0"(*w0), "1"(*w1), "2"(*w2)
      : "cc");

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   asm(R"(
      addq %[x],%[w0]
      adcq $0,%[w1]
      adcq $0,%[w2]
      )"
      : [w0]"=r"(*w0), [w1]"=r"(*w1), [w2]"=r"(*w2)
      : [x]"r"(x), "0"(*w0), "1"(*w1), "2"(*w2)
      : "cc");

#else
   *w0 += x;
   word c1 = (*w0 < x);
   *w1 += c1;
   word c2 = (*w1 < c1);
   *w2 += c2;
#endif
   }

/*
* Multiply-Add Accumulator
* (w2,w1,w0) += 2 * x * y
*/
inline void word3_muladd_2(word* w2, word* w1, word* w0, word x, word y)
   {
#if defined(BOTAN_MP_USE_X86_32_ASM)

   word z0 = 0, z1 = 0;

   asm("mull %[y]"
        : "=a"(z0),"=d"(z1)
        : "a"(x), [y]"rm"(y)
        : "cc");

   asm(R"(
      addl %[z0],%[w0]
      adcl %[z1],%[w1]
      adcl $0,%[w2]

      addl %[z0],%[w0]
      adcl %[z1],%[w1]
      adcl $0,%[w2]
      )"
      : [w0]"=r"(*w0), [w1]"=r"(*w1), [w2]"=r"(*w2)
      : [z0]"r"(z0), [z1]"r"(z1), "0"(*w0), "1"(*w1), "2"(*w2)
      : "cc");

#elif defined(BOTAN_MP_USE_X86_64_ASM)

   word z0 = 0, z1 = 0;

   asm("mulq %[y]"
        : "=a"(z0),"=d"(z1)
        : "a"(x), [y]"rm"(y)
        : "cc");

   asm(R"(
      addq %[z0],%[w0]
      adcq %[z1],%[w1]
      adcq $0,%[w2]

      addq %[z0],%[w0]
      adcq %[z1],%[w1]
      adcq $0,%[w2]
      )"
      : [w0]"=r"(*w0), [w1]"=r"(*w1), [w2]"=r"(*w2)
      : [z0]"r"(z0), [z1]"r"(z1), "0"(*w0), "1"(*w1), "2"(*w2)
      : "cc");

#else
   word carry = 0;
   x = word_madd2(x, y, &carry);
   y = carry;

   word top = (y >> (BOTAN_MP_WORD_BITS-1));
   y <<= 1;
   y |= (x >> (BOTAN_MP_WORD_BITS-1));
   x <<= 1;

   carry = 0;
   *w0 = word_add(*w0, x, &carry);
   *w1 = word_add(*w1, y, &carry);
   *w2 = word_add(*w2, top, &carry);
#endif
   }

#if defined(ASM)
  #undef ASM
  #undef DO_8_TIMES
  #undef ADD_OR_SUBTRACT
  #undef ADDSUB2_OP
  #undef ADDSUB3_OP
  #undef LINMUL_OP
  #undef MULADD_OP
#endif

}

namespace Botan {

const word MP_WORD_MAX = ~static_cast<word>(0);

/*
* If cond == 0, does nothing.
* If cond > 0, swaps x[0:size] with y[0:size]
* Runs in constant time
*/
inline void bigint_cnd_swap(word cnd, word x[], word y[], size_t size)
   {
   const auto mask = CT::Mask<word>::expand(cnd);

   for(size_t i = 0; i != size; ++i)
      {
      const word a = x[i];
      const word b = y[i];
      x[i] = mask.select(b, a);
      y[i] = mask.select(a, b);
      }
   }

inline word bigint_cnd_add(word cnd, word x[], word x_size,
                           const word y[], size_t y_size)
   {
   BOTAN_ASSERT(x_size >= y_size, "Expected sizes");

   const auto mask = CT::Mask<word>::expand(cnd);

   word carry = 0;

   const size_t blocks = y_size - (y_size % 8);
   word z[8] = { 0 };

   for(size_t i = 0; i != blocks; i += 8)
      {
      carry = word8_add3(z, x + i, y + i, carry);
      mask.select_n(x + i, z, x + i, 8);
      }

   for(size_t i = blocks; i != y_size; ++i)
      {
      z[0] = word_add(x[i], y[i], &carry);
      x[i] = mask.select(z[0], x[i]);
      }

   for(size_t i = y_size; i != x_size; ++i)
      {
      z[0] = word_add(x[i], 0, &carry);
      x[i] = mask.select(z[0], x[i]);
      }

   return mask.if_set_return(carry);
   }

/*
* If cond > 0 adds x[0:size] and y[0:size] and returns carry
* Runs in constant time
*/
inline word bigint_cnd_add(word cnd, word x[], const word y[], size_t size)
   {
   return bigint_cnd_add(cnd, x, size, y, size);
   }

/*
* If cond > 0 subtracts x[0:size] and y[0:size] and returns borrow
* Runs in constant time
*/
inline word bigint_cnd_sub(word cnd,
                           word x[], size_t x_size,
                           const word y[], size_t y_size)
   {
   BOTAN_ASSERT(x_size >= y_size, "Expected sizes");

   const auto mask = CT::Mask<word>::expand(cnd);

   word carry = 0;

   const size_t blocks = y_size - (y_size % 8);
   word z[8] = { 0 };

   for(size_t i = 0; i != blocks; i += 8)
      {
      carry = word8_sub3(z, x + i, y + i, carry);
      mask.select_n(x + i, z, x + i, 8);
      }

   for(size_t i = blocks; i != y_size; ++i)
      {
      z[0] = word_sub(x[i], y[i], &carry);
      x[i] = mask.select(z[0], x[i]);
      }

   for(size_t i = y_size; i != x_size; ++i)
      {
      z[0] = word_sub(x[i], 0, &carry);
      x[i] = mask.select(z[0], x[i]);
      }

   return mask.if_set_return(carry);
   }

/*
* If cond > 0 adds x[0:size] and y[0:size] and returns carry
* Runs in constant time
*/
inline word bigint_cnd_sub(word cnd, word x[], const word y[], size_t size)
   {
   return bigint_cnd_sub(cnd, x, size, y, size);
   }


/*
* Equivalent to
*   bigint_cnd_add( mask, x, y, size);
*   bigint_cnd_sub(~mask, x, y, size);
*
* Mask must be either 0 or all 1 bits
*/
inline void bigint_cnd_add_or_sub(CT::Mask<word> mask, word x[], const word y[], size_t size)
   {
   const size_t blocks = size - (size % 8);

   word carry = 0;
   word borrow = 0;

   word t0[8] = { 0 };
   word t1[8] = { 0 };

   for(size_t i = 0; i != blocks; i += 8)
      {
      carry = word8_add3(t0, x + i, y + i, carry);
      borrow = word8_sub3(t1, x + i, y + i, borrow);

      for(size_t j = 0; j != 8; ++j)
         x[i+j] = mask.select(t0[j], t1[j]);
      }

   for(size_t i = blocks; i != size; ++i)
      {
      const word a = word_add(x[i], y[i], &carry);
      const word s = word_sub(x[i], y[i], &borrow);

      x[i] = mask.select(a, s);
      }
   }

/*
* Equivalent to
*   bigint_cnd_add( mask, x, size, y, size);
*   bigint_cnd_sub(~mask, x, size, z, size);
*
* Mask must be either 0 or all 1 bits
*
* Returns the carry or borrow resp
*/
inline word bigint_cnd_addsub(CT::Mask<word> mask, word x[],
                              const word y[], const word z[],
                              size_t size)
   {
   const size_t blocks = size - (size % 8);

   word carry = 0;
   word borrow = 0;

   word t0[8] = { 0 };
   word t1[8] = { 0 };

   for(size_t i = 0; i != blocks; i += 8)
      {
      carry = word8_add3(t0, x + i, y + i, carry);
      borrow = word8_sub3(t1, x + i, z + i, borrow);

      for(size_t j = 0; j != 8; ++j)
         x[i+j] = mask.select(t0[j], t1[j]);
      }

   for(size_t i = blocks; i != size; ++i)
      {
      t0[0] = word_add(x[i], y[i], &carry);
      t1[0] = word_sub(x[i], z[i], &borrow);
      x[i] = mask.select(t0[0], t1[0]);
      }

   return mask.select(carry, borrow);
   }

/*
* 2s complement absolute value
* If cond > 0 sets x to ~x + 1
* Runs in constant time
*/
inline void bigint_cnd_abs(word cnd, word x[], size_t size)
   {
   const auto mask = CT::Mask<word>::expand(cnd);

   word carry = mask.if_set_return(1);
   for(size_t i = 0; i != size; ++i)
      {
      const word z = word_add(~x[i], 0, &carry);
      x[i] = mask.select(z, x[i]);
      }
   }

/**
* Two operand addition with carry out
*/
inline word bigint_add2_nc(word x[], size_t x_size, const word y[], size_t y_size)
   {
   word carry = 0;

   BOTAN_ASSERT(x_size >= y_size, "Expected sizes");

   const size_t blocks = y_size - (y_size % 8);

   for(size_t i = 0; i != blocks; i += 8)
      carry = word8_add2(x + i, y + i, carry);

   for(size_t i = blocks; i != y_size; ++i)
      x[i] = word_add(x[i], y[i], &carry);

   for(size_t i = y_size; i != x_size; ++i)
      x[i] = word_add(x[i], 0, &carry);

   return carry;
   }

/**
* Three operand addition with carry out
*/
inline word bigint_add3_nc(word z[],
                           const word x[], size_t x_size,
                           const word y[], size_t y_size)
   {
   if(x_size < y_size)
      { return bigint_add3_nc(z, y, y_size, x, x_size); }

   word carry = 0;

   const size_t blocks = y_size - (y_size % 8);

   for(size_t i = 0; i != blocks; i += 8)
      carry = word8_add3(z + i, x + i, y + i, carry);

   for(size_t i = blocks; i != y_size; ++i)
      z[i] = word_add(x[i], y[i], &carry);

   for(size_t i = y_size; i != x_size; ++i)
      z[i] = word_add(x[i], 0, &carry);

   return carry;
   }

/**
* Two operand addition
* @param x the first operand (and output)
* @param x_size size of x
* @param y the second operand
* @param y_size size of y (must be >= x_size)
*/
inline void bigint_add2(word x[], size_t x_size,
                        const word y[], size_t y_size)
   {
   x[x_size] += bigint_add2_nc(x, x_size, y, y_size);
   }

/**
* Three operand addition
*/
inline void bigint_add3(word z[],
                        const word x[], size_t x_size,
                        const word y[], size_t y_size)
   {
   z[x_size > y_size ? x_size : y_size] +=
      bigint_add3_nc(z, x, x_size, y, y_size);
   }

/**
* Two operand subtraction
*/
inline word bigint_sub2(word x[], size_t x_size,
                        const word y[], size_t y_size)
   {
   word borrow = 0;

   BOTAN_ASSERT(x_size >= y_size, "Expected sizes");

   const size_t blocks = y_size - (y_size % 8);

   for(size_t i = 0; i != blocks; i += 8)
      borrow = word8_sub2(x + i, y + i, borrow);

   for(size_t i = blocks; i != y_size; ++i)
      x[i] = word_sub(x[i], y[i], &borrow);

   for(size_t i = y_size; i != x_size; ++i)
      x[i] = word_sub(x[i], 0, &borrow);

   return borrow;
   }

/**
* Two operand subtraction, x = y - x; assumes y >= x
*/
inline void bigint_sub2_rev(word x[], const word y[], size_t y_size)
   {
   word borrow = 0;

   const size_t blocks = y_size - (y_size % 8);

   for(size_t i = 0; i != blocks; i += 8)
      borrow = word8_sub2_rev(x + i, y + i, borrow);

   for(size_t i = blocks; i != y_size; ++i)
      x[i] = word_sub(y[i], x[i], &borrow);

   BOTAN_ASSERT(borrow == 0, "y must be greater than x");
   }

/**
* Three operand subtraction
*/
inline word bigint_sub3(word z[],
                        const word x[], size_t x_size,
                        const word y[], size_t y_size)
   {
   word borrow = 0;

   BOTAN_ASSERT(x_size >= y_size, "Expected sizes");

   const size_t blocks = y_size - (y_size % 8);

   for(size_t i = 0; i != blocks; i += 8)
      borrow = word8_sub3(z + i, x + i, y + i, borrow);

   for(size_t i = blocks; i != y_size; ++i)
      z[i] = word_sub(x[i], y[i], &borrow);

   for(size_t i = y_size; i != x_size; ++i)
      z[i] = word_sub(x[i], 0, &borrow);

   return borrow;
   }

/**
* Return abs(x-y), ie if x >= y, then compute z = x - y
* Otherwise compute z = y - x
* No borrow is possible since the result is always >= 0
*
* Returns ~0 if x >= y or 0 if x < y
* @param z output array of at least N words
* @param x input array of N words
* @param y input array of N words
* @param N length of x and y
* @param ws array of at least 2*N words
*/
inline CT::Mask<word>
bigint_sub_abs(word z[],
               const word x[], const word y[], size_t N,
               word ws[])
   {
   // Subtract in both direction then conditional copy out the result

   word* ws0 = ws;
   word* ws1 = ws + N;

   word borrow0 = 0;
   word borrow1 = 0;

   const size_t blocks = N - (N % 8);

   for(size_t i = 0; i != blocks; i += 8)
      {
      borrow0 = word8_sub3(ws0 + i, x + i, y + i, borrow0);
      borrow1 = word8_sub3(ws1 + i, y + i, x + i, borrow1);
      }

   for(size_t i = blocks; i != N; ++i)
      {
      ws0[i] = word_sub(x[i], y[i], &borrow0);
      ws1[i] = word_sub(y[i], x[i], &borrow1);
      }

   return CT::conditional_copy_mem(borrow0, z, ws1, ws0, N);
   }

/*
* Shift Operations
*/
inline void bigint_shl1(word x[], size_t x_size, size_t x_words,
                        size_t word_shift, size_t bit_shift)
   {
   copy_mem(x + word_shift, x, x_words);
   clear_mem(x, word_shift);

   const auto carry_mask = CT::Mask<word>::expand(bit_shift);
   const size_t carry_shift = carry_mask.if_set_return(BOTAN_MP_WORD_BITS - bit_shift);

   word carry = 0;
   for(size_t i = word_shift; i != x_size; ++i)
      {
      const word w = x[i];
      x[i] = (w << bit_shift) | carry;
      carry = carry_mask.if_set_return(w >> carry_shift);
      }
   }

inline void bigint_shr1(word x[], size_t x_size,
                        size_t word_shift, size_t bit_shift)
   {
   const size_t top = x_size >= word_shift ? (x_size - word_shift) : 0;

   if(top > 0)
      copy_mem(x, x + word_shift, top);
   clear_mem(x + top, std::min(word_shift, x_size));

   const auto carry_mask = CT::Mask<word>::expand(bit_shift);
   const size_t carry_shift = carry_mask.if_set_return(BOTAN_MP_WORD_BITS - bit_shift);

   word carry = 0;

   for(size_t i = 0; i != top; ++i)
      {
      const word w = x[top - i - 1];
      x[top-i-1] = (w >> bit_shift) | carry;
      carry = carry_mask.if_set_return(w << carry_shift);
      }
   }

inline void bigint_shl2(word y[], const word x[], size_t x_size,
                        size_t word_shift, size_t bit_shift)
   {
   copy_mem(y + word_shift, x, x_size);

   const auto carry_mask = CT::Mask<word>::expand(bit_shift);
   const size_t carry_shift = carry_mask.if_set_return(BOTAN_MP_WORD_BITS - bit_shift);

   word carry = 0;
   for(size_t i = word_shift; i != x_size + word_shift + 1; ++i)
      {
      const word w = y[i];
      y[i] = (w << bit_shift) | carry;
      carry = carry_mask.if_set_return(w >> carry_shift);
      }
   }

inline void bigint_shr2(word y[], const word x[], size_t x_size,
                        size_t word_shift, size_t bit_shift)
   {
   const size_t new_size = x_size < word_shift ? 0 : (x_size - word_shift);

   if(new_size > 0)
      copy_mem(y, x + word_shift, new_size);

   const auto carry_mask = CT::Mask<word>::expand(bit_shift);
   const size_t carry_shift = carry_mask.if_set_return(BOTAN_MP_WORD_BITS - bit_shift);

   word carry = 0;
   for(size_t i = new_size; i > 0; --i)
      {
      word w = y[i-1];
      y[i-1] = (w >> bit_shift) | carry;
      carry = carry_mask.if_set_return(w << carry_shift);
      }
   }

/*
* Linear Multiply - returns the carry
*/
inline word BOTAN_WARN_UNUSED_RESULT bigint_linmul2(word x[], size_t x_size, word y)
   {
   const size_t blocks = x_size - (x_size % 8);

   word carry = 0;

   for(size_t i = 0; i != blocks; i += 8)
      carry = word8_linmul2(x + i, y, carry);

   for(size_t i = blocks; i != x_size; ++i)
      x[i] = word_madd2(x[i], y, &carry);

   return carry;
   }

inline void bigint_linmul3(word z[], const word x[], size_t x_size, word y)
   {
   const size_t blocks = x_size - (x_size % 8);

   word carry = 0;

   for(size_t i = 0; i != blocks; i += 8)
      carry = word8_linmul3(z + i, x + i, y, carry);

   for(size_t i = blocks; i != x_size; ++i)
      z[i] = word_madd2(x[i], y, &carry);

   z[x_size] = carry;
   }

/**
* Compare x and y
* Return -1 if x < y
* Return 0 if x == y
* Return 1 if x > y
*/
inline int32_t bigint_cmp(const word x[], size_t x_size,
                          const word y[], size_t y_size)
   {
   static_assert(sizeof(word) >= sizeof(uint32_t), "Size assumption");

   const word LT = static_cast<word>(-1);
   const word EQ = 0;
   const word GT = 1;

   const size_t common_elems = std::min(x_size, y_size);

   word result = EQ; // until found otherwise

   for(size_t i = 0; i != common_elems; i++)
      {
      const auto is_eq = CT::Mask<word>::is_equal(x[i], y[i]);
      const auto is_lt = CT::Mask<word>::is_lt(x[i], y[i]);

      result = is_eq.select(result, is_lt.select(LT, GT));
      }

   if(x_size < y_size)
      {
      word mask = 0;
      for(size_t i = x_size; i != y_size; i++)
         mask |= y[i];

      // If any bits were set in high part of y, then x < y
      result = CT::Mask<word>::is_zero(mask).select(result, LT);
      }
   else if(y_size < x_size)
      {
      word mask = 0;
      for(size_t i = y_size; i != x_size; i++)
         mask |= x[i];

      // If any bits were set in high part of x, then x > y
      result = CT::Mask<word>::is_zero(mask).select(result, GT);
      }

   CT::unpoison(result);
   BOTAN_DEBUG_ASSERT(result == LT || result == GT || result == EQ);
   return static_cast<int32_t>(result);
   }

/**
* Compare x and y
* Return ~0 if x[0:x_size] < y[0:y_size] or 0 otherwise
* If lt_or_equal is true, returns ~0 also for x == y
*/
inline CT::Mask<word>
bigint_ct_is_lt(const word x[], size_t x_size,
                const word y[], size_t y_size,
                bool lt_or_equal = false)
   {
   const size_t common_elems = std::min(x_size, y_size);

   auto is_lt = CT::Mask<word>::expand(lt_or_equal);

   for(size_t i = 0; i != common_elems; i++)
      {
      const auto eq = CT::Mask<word>::is_equal(x[i], y[i]);
      const auto lt = CT::Mask<word>::is_lt(x[i], y[i]);
      is_lt = eq.select_mask(is_lt, lt);
      }

   if(x_size < y_size)
      {
      word mask = 0;
      for(size_t i = x_size; i != y_size; i++)
         mask |= y[i];
      // If any bits were set in high part of y, then is_lt should be forced true
      is_lt |= CT::Mask<word>::expand(mask);
      }
   else if(y_size < x_size)
      {
      word mask = 0;
      for(size_t i = y_size; i != x_size; i++)
         mask |= x[i];

      // If any bits were set in high part of x, then is_lt should be false
      is_lt &= CT::Mask<word>::is_zero(mask);
      }

   return is_lt;
   }

inline CT::Mask<word>
bigint_ct_is_eq(const word x[], size_t x_size,
                const word y[], size_t y_size)
   {
   const size_t common_elems = std::min(x_size, y_size);

   word diff = 0;

   for(size_t i = 0; i != common_elems; i++)
      {
      diff |= (x[i] ^ y[i]);
      }

   // If any bits were set in high part of x/y, then they are not equal
   if(x_size < y_size)
      {
      for(size_t i = x_size; i != y_size; i++)
         diff |= y[i];
      }
   else if(y_size < x_size)
      {
      for(size_t i = y_size; i != x_size; i++)
         diff |= x[i];
      }

   return CT::Mask<word>::is_zero(diff);
   }

/**
* Set z to abs(x-y), ie if x >= y, then compute z = x - y
* Otherwise compute z = y - x
* No borrow is possible since the result is always >= 0
*
* Return the relative size of x vs y (-1, 0, 1)
*
* @param z output array of max(x_size,y_size) words
* @param x input param
* @param x_size length of x
* @param y input param
* @param y_size length of y
*/
inline int32_t
bigint_sub_abs(word z[],
               const word x[], size_t x_size,
               const word y[], size_t y_size)
   {
   const int32_t relative_size = bigint_cmp(x, x_size, y, y_size);

   // Swap if relative_size == -1
   const bool need_swap = relative_size < 0;
   CT::conditional_swap_ptr(need_swap, x, y);
   CT::conditional_swap(need_swap, x_size, y_size);

   /*
   * We know at this point that x >= y so if y_size is larger than
   * x_size, we are guaranteed they are just leading zeros which can
   * be ignored
   */
   y_size = std::min(x_size, y_size);

   bigint_sub3(z, x, x_size, y, y_size);

   return relative_size;
   }

/**
* Set t to t-s modulo mod
*
* @param t first integer
* @param s second integer
* @param mod the modulus
* @param mod_sw size of t, s, and mod
* @param ws workspace of size mod_sw
*/
inline void
bigint_mod_sub(word t[], const word s[], const word mod[], size_t mod_sw, word ws[])
   {
   // is t < s or not?
   const auto is_lt = bigint_ct_is_lt(t, mod_sw, s, mod_sw);

   // ws = p - s
   const word borrow = bigint_sub3(ws, mod, mod_sw, s, mod_sw);

   // Compute either (t - s) or (t + (p - s)) depending on mask
   const word carry = bigint_cnd_addsub(is_lt, t, ws, s, mod_sw);

   BOTAN_DEBUG_ASSERT(borrow == 0 && carry == 0);
   BOTAN_UNUSED(carry, borrow);
   }

template<size_t N>
inline void bigint_mod_sub_n(word t[], const word s[], const word mod[], word ws[])
   {
   // is t < s or not?
   const auto is_lt = bigint_ct_is_lt(t, N, s, N);

   // ws = p - s
   const word borrow = bigint_sub3(ws, mod, N, s, N);

   // Compute either (t - s) or (t + (p - s)) depending on mask
   const word carry = bigint_cnd_addsub(is_lt, t, ws, s, N);

   BOTAN_DEBUG_ASSERT(borrow == 0 && carry == 0);
   BOTAN_UNUSED(carry, borrow);
   }

/**
* Compute ((n1<<bits) + n0) / d
*/
inline word bigint_divop(word n1, word n0, word d)
   {
   if(d == 0)
      throw Invalid_Argument("bigint_divop divide by zero");

#if defined(BOTAN_HAS_MP_DWORD)
   return ((static_cast<dword>(n1) << BOTAN_MP_WORD_BITS) | n0) / d;
#else

   word high = n1 % d;
   word quotient = 0;

   for(size_t i = 0; i != BOTAN_MP_WORD_BITS; ++i)
      {
      const word high_top_bit = high >> (BOTAN_MP_WORD_BITS-1);

      high <<= 1;
      high |= (n0 >> (BOTAN_MP_WORD_BITS-1-i)) & 1;
      quotient <<= 1;

      if(high_top_bit || high >= d)
         {
         high -= d;
         quotient |= 1;
         }
      }

   return quotient;
#endif
   }

/**
* Compute ((n1<<bits) + n0) % d
*/
inline word bigint_modop(word n1, word n0, word d)
   {
   if(d == 0)
      throw Invalid_Argument("bigint_modop divide by zero");

#if defined(BOTAN_HAS_MP_DWORD)
   return ((static_cast<dword>(n1) << BOTAN_MP_WORD_BITS) | n0) % d;
#else
   word z = bigint_divop(n1, n0, d);
   word dummy = 0;
   z = word_madd2(z, d, &dummy);
   return (n0-z);
#endif
   }

/*
* Comba Multiplication / Squaring
*/
void bigint_comba_mul4(word z[8], const word x[4], const word y[4]);
void bigint_comba_mul6(word z[12], const word x[6], const word y[6]);
void bigint_comba_mul8(word z[16], const word x[8], const word y[8]);
void bigint_comba_mul9(word z[18], const word x[9], const word y[9]);
void bigint_comba_mul16(word z[32], const word x[16], const word y[16]);
void bigint_comba_mul24(word z[48], const word x[24], const word y[24]);

void bigint_comba_sqr4(word out[8], const word in[4]);
void bigint_comba_sqr6(word out[12], const word in[6]);
void bigint_comba_sqr8(word out[16], const word in[8]);
void bigint_comba_sqr9(word out[18], const word in[9]);
void bigint_comba_sqr16(word out[32], const word in[16]);
void bigint_comba_sqr24(word out[48], const word in[24]);

/**
* Montgomery Reduction
* @param z integer to reduce, of size exactly 2*(p_size+1).
           Output is in the first p_size+1 words, higher
           words are set to zero.
* @param p modulus
* @param p_size size of p
* @param p_dash Montgomery value
* @param workspace array of at least 2*(p_size+1) words
* @param ws_size size of workspace in words
*/
void bigint_monty_redc(word z[],
                       const word p[], size_t p_size,
                       word p_dash,
                       word workspace[],
                       size_t ws_size);

/*
* High Level Multiplication/Squaring Interfaces
*/

void bigint_mul(word z[], size_t z_size,
                const word x[], size_t x_size, size_t x_sw,
                const word y[], size_t y_size, size_t y_sw,
                word workspace[], size_t ws_size);

void bigint_sqr(word z[], size_t z_size,
                const word x[], size_t x_size, size_t x_sw,
                word workspace[], size_t ws_size);

}

namespace Botan {

/*
* Each of these functions makes the following assumptions:
*
* z_size >= 2*(p_size + 1)
* ws_size >= z_size
*/

void bigint_monty_redc_4(word z[], const word p[], word p_dash, word ws[]);
void bigint_monty_redc_6(word z[], const word p[], word p_dash, word ws[]);
void bigint_monty_redc_8(word z[], const word p[], word p_dash, word ws[]);
void bigint_monty_redc_16(word z[], const word p[], word p_dash, word ws[]);
void bigint_monty_redc_24(word z[], const word p[], word p_dash, word ws[]);
void bigint_monty_redc_32(word z[], const word p[], word p_dash, word ws[]);


}

namespace Botan {

namespace OS {

/*
* This header is internal (not installed) and these functions are not
* intended to be called by applications. However they are given public
* visibility (using BOTAN_TEST_API macro) for the tests. This also probably
* allows them to be overridden by the application on ELF systems, but
* this hasn't been tested.
*/

/**
* @return process ID assigned by the operating system.
* On Unix and Windows systems, this always returns a result
* On IncludeOS it returns 0 since there is no process ID to speak of
* in a unikernel.
*/
uint32_t BOTAN_TEST_API get_process_id();

/**
* Test if we are currently running with elevated permissions
* eg setuid, setgid, or with POSIX caps set.
*/
bool running_in_privileged_state();

/**
* @return CPU processor clock, if available
*
* On Windows, calls QueryPerformanceCounter.
*
* Under GCC or Clang on supported platforms the hardware cycle counter is queried.
* Currently supported processors are x86, PPC, Alpha, SPARC, IA-64, S/390x, and HP-PA.
* If no CPU cycle counter is available on this system, returns zero.
*/
uint64_t BOTAN_TEST_API get_cpu_cycle_counter();

size_t BOTAN_TEST_API get_cpu_total();
size_t BOTAN_TEST_API get_cpu_available();

/**
* Return the ELF auxiliary vector cooresponding to the given ID.
* This only makes sense on Unix-like systems and is currently
* only supported on Linux, Android, and FreeBSD.
*
* Returns zero if not supported on the current system or if
* the id provided is not known.
*/
unsigned long get_auxval(unsigned long id);

/*
* @return best resolution timestamp available
*
* The epoch and update rate of this clock is arbitrary and depending
* on the hardware it may not tick at a constant rate.
*
* Uses hardware cycle counter, if available.
* On POSIX platforms clock_gettime is used with a monotonic timer
* As a final fallback std::chrono::high_resolution_clock is used.
*/
uint64_t BOTAN_TEST_API get_high_resolution_clock();

/**
* @return system clock (reflecting wall clock) with best resolution
* available, normalized to nanoseconds resolution.
*/
uint64_t BOTAN_TEST_API get_system_timestamp_ns();

/**
* @return maximum amount of memory (in bytes) Botan could/should
* hyptothetically allocate for the memory poool. Reads environment
* variable "BOTAN_MLOCK_POOL_SIZE", set to "0" to disable pool.
*/
size_t get_memory_locking_limit();

/**
* Return the size of a memory page, if that can be derived on the
* current system. Otherwise returns some default value (eg 4096)
*/
size_t system_page_size();

/**
* Read the value of an environment variable, setting it to value_out if it
* exists.  Returns false and sets value_out to empty string if no such variable
* is set. If the process seems to be running in a privileged state (such as
* setuid) then always returns false and does not examine the environment.
*/
bool read_env_variable(std::string& value_out, const std::string& var_name);

/**
* Read the value of an environment variable and convert it to an
* integer. If not set or conversion fails, returns the default value.
*
* If the process seems to be running in a privileged state (such as setuid)
* then always returns nullptr, similiar to glibc's secure_getenv.
*/
size_t read_env_variable_sz(const std::string& var_name, size_t def_value = 0);

/**
* Request count pages of RAM which are locked into memory using mlock,
* VirtualLock, or some similar OS specific API. Free it with free_locked_pages.
*
* Returns an empty list on failure. This function is allowed to return fewer
* than count pages.
*
* The contents of the allocated pages are undefined.
*
* Each page is preceded by and followed by a page which is marked
* as noaccess, such that accessing it will cause a crash. This turns
* out of bound reads/writes into crash events.
*
* @param count requested number of locked pages
*/
std::vector<void*> allocate_locked_pages(size_t count);

/**
* Free memory allocated by allocate_locked_pages
* @param pages a list of pages returned by allocate_locked_pages
*/
void free_locked_pages(const std::vector<void*>& pages);

/**
* Set the MMU to prohibit access to this page
*/
void page_prohibit_access(void* page);

/**
* Set the MMU to allow R/W access to this page
*/
void page_allow_access(void* page);


/**
* Run a probe instruction to test for support for a CPU instruction.
* Runs in system-specific env that catches illegal instructions; this
* function always fails if the OS doesn't provide this.
* Returns value of probe_fn, if it could run.
* If error occurs, returns negative number.
* This allows probe_fn to indicate errors of its own, if it wants.
* For example the instruction might not only be only available on some
* CPUs, but also buggy on some subset of these - the probe function
* can test to make sure the instruction works properly before
* indicating that the instruction is available.
*
* @warning on Unix systems uses signal handling in a way that is not
* thread safe. It should only be called in a single-threaded context
* (ie, at static init time).
*
* If probe_fn throws an exception the result is undefined.
*
* Return codes:
* -1 illegal instruction detected
*/
int BOTAN_TEST_API run_cpu_instruction_probe(std::function<int ()> probe_fn);

/**
* Represents a terminal state
*/
class BOTAN_UNSTABLE_API Echo_Suppression
   {
   public:
      /**
      * Reenable echo on this terminal. Can be safely called
      * multiple times. May throw if an error occurs.
      */
      virtual void reenable_echo() = 0;

      /**
      * Implicitly calls reenable_echo, but swallows/ignored all
      * errors which would leave the terminal in an invalid state.
      */
      virtual ~Echo_Suppression() = default;
   };

/**
* Suppress echo on the terminal
* Returns null if this operation is not supported on the current system.
*/
std::unique_ptr<Echo_Suppression> BOTAN_UNSTABLE_API suppress_echo_on_terminal();

}

}

namespace Botan {

/**
* Returns the allowed padding schemes when using the given
* algorithm (key type) for creating digital signatures.
*
* @param algo the algorithm for which to look up supported padding schemes
* @return a vector of supported padding schemes
*/
BOTAN_TEST_API const std::vector<std::string> get_sig_paddings(const std::string algo);

/**
* Returns true iff the given padding scheme is valid for the given
* signature algorithm (key type).
*
* @param algo the signature algorithm to be used
* @param padding the padding scheme to be used
*/
bool sig_algo_and_pad_ok(const std::string algo, const std::string padding);

}

namespace Botan {

namespace PK_Ops {

class Encryption_with_EME : public Encryption
   {
   public:
      size_t max_input_bits() const override;

      secure_vector<uint8_t> encrypt(const uint8_t msg[], size_t msg_len,
                                  RandomNumberGenerator& rng) override;

      ~Encryption_with_EME() = default;
   protected:
      explicit Encryption_with_EME(const std::string& eme);
   private:
      virtual size_t max_raw_input_bits() const = 0;

      virtual secure_vector<uint8_t> raw_encrypt(const uint8_t msg[], size_t len,
                                              RandomNumberGenerator& rng) = 0;
      std::unique_ptr<EME> m_eme;
   };

class Decryption_with_EME : public Decryption
   {
   public:
      secure_vector<uint8_t> decrypt(uint8_t& valid_mask,
                                  const uint8_t msg[], size_t msg_len) override;

      ~Decryption_with_EME() = default;
   protected:
      explicit Decryption_with_EME(const std::string& eme);
   private:
      virtual secure_vector<uint8_t> raw_decrypt(const uint8_t msg[], size_t len) = 0;
      std::unique_ptr<EME> m_eme;
   };

class Verification_with_EMSA : public Verification
   {
   public:
      ~Verification_with_EMSA() = default;

      void update(const uint8_t msg[], size_t msg_len) override;
      bool is_valid_signature(const uint8_t sig[], size_t sig_len) override;

      bool do_check(const secure_vector<uint8_t>& msg,
                    const uint8_t sig[], size_t sig_len);

      std::string hash_for_signature() { return m_hash; }

   protected:
      explicit Verification_with_EMSA(const std::string& emsa);

      /**
      * Get the maximum message size in bits supported by this public key.
      * @return maximum message in bits
      */
      virtual size_t max_input_bits() const = 0;

      /**
      * @return boolean specifying if this signature scheme uses
      * a message prefix returned by message_prefix()
      */
      virtual bool has_prefix() { return false; }

      /**
      * @return the message prefix if this signature scheme uses
      * a message prefix, signaled via has_prefix()
      */
      virtual secure_vector<uint8_t> message_prefix() const { throw Invalid_State("No prefix"); }

      /**
      * @return boolean specifying if this key type supports message
      * recovery and thus if you need to call verify() or verify_mr()
      */
      virtual bool with_recovery() const = 0;

      /*
      * Perform a signature check operation
      * @param msg the message
      * @param msg_len the length of msg in bytes
      * @param sig the signature
      * @param sig_len the length of sig in bytes
      * @returns if signature is a valid one for message
      */
      virtual bool verify(const uint8_t[], size_t,
                          const uint8_t[], size_t)
         {
         throw Invalid_State("Message recovery required");
         }

      /*
      * Perform a signature operation (with message recovery)
      * Only call this if with_recovery() returns true
      * @param msg the message
      * @param msg_len the length of msg in bytes
      * @returns recovered message
      */
      virtual secure_vector<uint8_t> verify_mr(const uint8_t[], size_t)
         {
         throw Invalid_State("Message recovery not supported");
         }

      std::unique_ptr<EMSA> clone_emsa() const { return std::unique_ptr<EMSA>(m_emsa->clone()); }

   private:
      std::unique_ptr<EMSA> m_emsa;
      const std::string m_hash;
      bool m_prefix_used;
   };

class Signature_with_EMSA : public Signature
   {
   public:
      void update(const uint8_t msg[], size_t msg_len) override;

      secure_vector<uint8_t> sign(RandomNumberGenerator& rng) override;
   protected:
      explicit Signature_with_EMSA(const std::string& emsa);
      ~Signature_with_EMSA() = default;

      std::string hash_for_signature() { return m_hash; }

      /**
      * @return boolean specifying if this signature scheme uses
      * a message prefix returned by message_prefix()
      */
      virtual bool has_prefix() { return false; }

      /**
      * @return the message prefix if this signature scheme uses
      * a message prefix, signaled via has_prefix()
      */
      virtual secure_vector<uint8_t> message_prefix() const { throw Invalid_State("No prefix"); }

      std::unique_ptr<EMSA> clone_emsa() const { return std::unique_ptr<EMSA>(m_emsa->clone()); }

   private:

      /**
      * Get the maximum message size in bits supported by this public key.
      * @return maximum message in bits
      */
      virtual size_t max_input_bits() const = 0;

      bool self_test_signature(const std::vector<uint8_t>& msg,
                               const std::vector<uint8_t>& sig) const;

      virtual secure_vector<uint8_t> raw_sign(const uint8_t msg[], size_t msg_len,
                                           RandomNumberGenerator& rng) = 0;

      std::unique_ptr<EMSA> m_emsa;
      const std::string m_hash;
      bool m_prefix_used;
   };

class Key_Agreement_with_KDF : public Key_Agreement
   {
   public:
      secure_vector<uint8_t> agree(size_t key_len,
                                const uint8_t other_key[], size_t other_key_len,
                                const uint8_t salt[], size_t salt_len) override;

   protected:
      explicit Key_Agreement_with_KDF(const std::string& kdf);
      ~Key_Agreement_with_KDF() = default;
   private:
      virtual secure_vector<uint8_t> raw_agree(const uint8_t w[], size_t w_len) = 0;
      std::unique_ptr<KDF> m_kdf;
   };

class KEM_Encryption_with_KDF : public KEM_Encryption
   {
   public:
      void kem_encrypt(secure_vector<uint8_t>& out_encapsulated_key,
                       secure_vector<uint8_t>& out_shared_key,
                       size_t desired_shared_key_len,
                       Botan::RandomNumberGenerator& rng,
                       const uint8_t salt[],
                       size_t salt_len) override;

   protected:
      virtual void raw_kem_encrypt(secure_vector<uint8_t>& out_encapsulated_key,
                                   secure_vector<uint8_t>& raw_shared_key,
                                   Botan::RandomNumberGenerator& rng) = 0;

      explicit KEM_Encryption_with_KDF(const std::string& kdf);
      ~KEM_Encryption_with_KDF() = default;
   private:
      std::unique_ptr<KDF> m_kdf;
   };

class KEM_Decryption_with_KDF : public KEM_Decryption
   {
   public:
      secure_vector<uint8_t> kem_decrypt(const uint8_t encap_key[],
                                      size_t len,
                                      size_t desired_shared_key_len,
                                      const uint8_t salt[],
                                      size_t salt_len) override;

   protected:
      virtual secure_vector<uint8_t>
      raw_kem_decrypt(const uint8_t encap_key[], size_t len) = 0;

      explicit KEM_Decryption_with_KDF(const std::string& kdf);
      ~KEM_Decryption_with_KDF() = default;
   private:
      std::unique_ptr<KDF> m_kdf;
   };

}

}

namespace Botan {

template<typename T>
inline void prefetch_readonly(const T* addr, size_t length)
   {
#if defined(__GNUG__)
   const size_t Ts_per_cache_line = CPUID::cache_line_size() / sizeof(T);

   for(size_t i = 0; i <= length; i += Ts_per_cache_line)
      __builtin_prefetch(addr + i, 0);
#endif
   }

template<typename T>
inline void prefetch_readwrite(const T* addr, size_t length)
   {
#if defined(__GNUG__)
   const size_t Ts_per_cache_line = CPUID::cache_line_size() / sizeof(T);

   for(size_t i = 0; i <= length; i += Ts_per_cache_line)
      __builtin_prefetch(addr + i, 1);
#endif
   }

}

namespace Botan {

class BigInt;
class Modular_Reducer;
class Montgomery_Params;
class RandomNumberGenerator;

/**
* Perform Lucas primality test
* @see FIPS 186-4 C.3.3
*
* @warning it is possible to construct composite integers which pass
* this test alone.
*
* @param n the positive integer to test
* @param mod_n a pre-created Modular_Reducer for n
* @return true if n seems probably prime, false if n is composite
*/
bool BOTAN_TEST_API is_lucas_probable_prime(const BigInt& n, const Modular_Reducer& mod_n);

/**
* Perform Bailie-PSW primality test
*
* This is a combination of Miller-Rabin with base 2 and a Lucas test. No known
* composite integer passes both tests, though it is conjectured that infinitely
* many composite counterexamples exist.
*
* @param n the positive integer to test
* @param mod_n a pre-created Modular_Reducer for n
* @return true if n seems probably prime, false if n is composite
*/
bool BOTAN_TEST_API is_bailie_psw_probable_prime(const BigInt& n, const Modular_Reducer& mod_n);

/**
* Perform Bailie-PSW primality test
*
* This is a combination of Miller-Rabin with base 2 and a Lucas test. No known
* composite integer passes both tests, though it is conjectured that infinitely
* many composite counterexamples exist.
*
* @param n the positive integer to test
* @return true if n seems probably prime, false if n is composite
*/
bool is_bailie_psw_probable_prime(const BigInt& n);

/**
* Return required number of Miller-Rabin tests in order to
* reach the specified probability of error.
*
* @param n_bits the bit-length of the integer being tested
* @param prob chance of false positive is bounded by 1/2**prob
* @param random is set if (and only if) the integer was randomly generated by us
*        and thus cannot have been maliciously constructed.
*/
size_t miller_rabin_test_iterations(size_t n_bits, size_t prob, bool random);

/**
* Perform a single Miller-Rabin test with specified base
*
* @param n the positive integer to test
* @param mod_n a pre-created Modular_Reducer for n
* @param monty_n Montgomery parameters for n
* @param a the base to check
* @return result of primality test
*/
bool passes_miller_rabin_test(const BigInt& n,
                              const Modular_Reducer& mod_n,
                              const std::shared_ptr<Montgomery_Params>& monty_n,
                              const BigInt& a);

/**
* Perform t iterations of a Miller-Rabin primality test with random bases
*
* @param n the positive integer to test
* @param mod_n a pre-created Modular_Reducer for n
* @param rng a random number generator
* @param t number of tests to perform
*
* @return result of primality test
*/
bool BOTAN_TEST_API is_miller_rabin_probable_prime(const BigInt& n,
                                                   const Modular_Reducer& mod_n,
                                                   RandomNumberGenerator& rng,
                                                   size_t t);

}

namespace Botan {

/**
* Round up
* @param n a non-negative integer
* @param align_to the alignment boundary
* @return n rounded up to a multiple of align_to
*/
inline size_t round_up(size_t n, size_t align_to)
   {
   BOTAN_ARG_CHECK(align_to != 0, "align_to must not be 0");

   if(n % align_to)
      n += align_to - (n % align_to);
   return n;
   }

/**
* Round down
* @param n an integer
* @param align_to the alignment boundary
* @return n rounded down to a multiple of align_to
*/
template<typename T>
inline constexpr T round_down(T n, T align_to)
   {
   return (align_to == 0) ? n : (n - (n % align_to));
   }

/**
* Clamp
*/
inline size_t clamp(size_t n, size_t lower_bound, size_t upper_bound)
   {
   if(n < lower_bound)
      return lower_bound;
   if(n > upper_bound)
      return upper_bound;
   return n;
   }

}

namespace Botan {

class BOTAN_PUBLIC_API(2,0) Integer_Overflow_Detected final : public Exception
   {
   public:
      Integer_Overflow_Detected(const std::string& file, int line) :
         Exception("Integer overflow detected at " + file + ":" + std::to_string(line))
         {}

      ErrorType error_type() const noexcept override { return ErrorType::InternalError; }
   };

inline size_t checked_add(size_t x, size_t y, const char* file, int line)
   {
   // TODO: use __builtin_x_overflow on GCC and Clang
   size_t z = x + y;
   if(z < x)
      {
      throw Integer_Overflow_Detected(file, line);
      }
   return z;
   }

#define BOTAN_CHECKED_ADD(x,y) checked_add(x,y,__FILE__,__LINE__)

}

namespace Botan {

inline std::vector<uint8_t> to_byte_vector(const std::string& s)
   {
   return std::vector<uint8_t>(s.cbegin(), s.cend());
   }

inline std::string to_string(const secure_vector<uint8_t> &bytes)
   {
   return std::string(bytes.cbegin(), bytes.cend());
   }

/**
* Return the keys of a map as a std::set
*/
template<typename K, typename V>
std::set<K> map_keys_as_set(const std::map<K, V>& kv)
   {
   std::set<K> s;
   for(auto&& i : kv)
      {
      s.insert(i.first);
      }
   return s;
   }

/*
* Searching through a std::map
* @param mapping the map to search
* @param key is what to look for
* @param null_result is the value to return if key is not in mapping
* @return mapping[key] or null_result
*/
template<typename K, typename V>
inline V search_map(const std::map<K, V>& mapping,
                    const K& key,
                    const V& null_result = V())
   {
   auto i = mapping.find(key);
   if(i == mapping.end())
      return null_result;
   return i->second;
   }

template<typename K, typename V, typename R>
inline R search_map(const std::map<K, V>& mapping, const K& key,
                    const R& null_result, const R& found_result)
   {
   auto i = mapping.find(key);
   if(i == mapping.end())
      return null_result;
   return found_result;
   }

/*
* Insert a key/value pair into a multimap
*/
template<typename K, typename V>
void multimap_insert(std::multimap<K, V>& multimap,
                     const K& key, const V& value)
   {
   multimap.insert(std::make_pair(key, value));
   }

/**
* Existence check for values
*/
template<typename T>
bool value_exists(const std::vector<T>& vec,
                  const T& val)
   {
   for(size_t i = 0; i != vec.size(); ++i)
      if(vec[i] == val)
         return true;
   return false;
   }

template<typename T, typename Pred>
void map_remove_if(Pred pred, T& assoc)
   {
   auto i = assoc.begin();
   while(i != assoc.end())
      {
      if(pred(i->first))
         assoc.erase(i++);
      else
         i++;
      }
   }

}

namespace Botan {

class BOTAN_TEST_API Timer final
   {
   public:
      Timer(const std::string& name,
            const std::string& provider,
            const std::string& doing,
            uint64_t event_mult,
            size_t buf_size,
            double clock_cycle_ratio,
            uint64_t clock_speed)
         : m_name(name + ((provider.empty() || provider == "base") ? "" : " [" + provider + "]"))
         , m_doing(doing)
         , m_buf_size(buf_size)
         , m_event_mult(event_mult)
         , m_clock_cycle_ratio(clock_cycle_ratio)
         , m_clock_speed(clock_speed)
         {}

      Timer(const std::string& name) :
         Timer(name, "", "", 1, 0, 0.0, 0)
         {}

      Timer(const std::string& name, size_t buf_size) :
         Timer(name, "", "", buf_size, buf_size, 0.0, 0)
         {}

      Timer(const Timer& other) = default;
      Timer& operator=(const Timer& other) = default;

      void start();

      void stop();

      bool under(std::chrono::milliseconds msec)
         {
         return (milliseconds() < msec.count());
         }

      class Timer_Scope final
         {
         public:
            explicit Timer_Scope(Timer& timer)
               : m_timer(timer)
               {
               m_timer.start();
               }
            ~Timer_Scope()
               {
               try
                  {
                  m_timer.stop();
                  }
               catch(...) {}
               }
         private:
            Timer& m_timer;
         };

      template<typename F>
      auto run(F f) -> decltype(f())
         {
         Timer_Scope timer(*this);
         return f();
         }

      template<typename F>
      void run_until_elapsed(std::chrono::milliseconds msec, F f)
         {
         while(this->under(msec))
            {
            run(f);
            }
         }

      uint64_t value() const
         {
         return m_time_used;
         }

      double seconds() const
         {
         return milliseconds() / 1000.0;
         }

      double milliseconds() const
         {
         return value() / 1000000.0;
         }

      double ms_per_event() const
         {
         return milliseconds() / events();
         }

      uint64_t cycles_consumed() const
         {
         if(m_clock_speed != 0)
            {
            return static_cast<uint64_t>((m_clock_speed * value()) / 1000.0);
            }
         return m_cpu_cycles_used;
         }

      uint64_t events() const
         {
         return m_event_count * m_event_mult;
         }

      const std::string& get_name() const
         {
         return m_name;
         }

      const std::string& doing() const
         {
         return m_doing;
         }

      size_t buf_size() const
         {
         return m_buf_size;
         }

      double bytes_per_second() const
         {
         return seconds() > 0.0 ? events() / seconds() : 0.0;
         }

      double events_per_second() const
         {
         return seconds() > 0.0 ? events() / seconds() : 0.0;
         }

      double seconds_per_event() const
         {
         return events() > 0 ? seconds() / events() : 0.0;
         }

      void set_custom_msg(const std::string& s)
         {
         m_custom_msg = s;
         }

      bool operator<(const Timer& other) const;

      std::string to_string() const;

   private:
      std::string result_string_bps() const;
      std::string result_string_ops() const;

      // const data
      std::string m_name, m_doing;
      size_t m_buf_size;
      uint64_t m_event_mult;
      double m_clock_cycle_ratio;
      uint64_t m_clock_speed;

      // set at runtime
      std::string m_custom_msg;
      uint64_t m_time_used = 0, m_timer_start = 0;
      uint64_t m_event_count = 0;

      uint64_t m_max_time = 0, m_min_time = 0;
      uint64_t m_cpu_cycles_start = 0, m_cpu_cycles_used = 0;
   };

}
/*
* Algorithm Identifier
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/*
* Create an AlgorithmIdentifier
*/
AlgorithmIdentifier::AlgorithmIdentifier(const OID& alg_id,
                                         const std::vector<uint8_t>& param) :
   oid(alg_id),
   parameters(param)
   {}

/*
* Create an AlgorithmIdentifier
*/
AlgorithmIdentifier::AlgorithmIdentifier(const std::string& alg_id,
                                         const std::vector<uint8_t>& param) :
   AlgorithmIdentifier(OID::from_string(alg_id), param)
   {}

/*
* Create an AlgorithmIdentifier
*/
AlgorithmIdentifier::AlgorithmIdentifier(const OID& alg_id,
                                         Encoding_Option option) :
   oid(alg_id),
   parameters()
   {
   const uint8_t DER_NULL[] = { 0x05, 0x00 };

   if(option == USE_NULL_PARAM)
      parameters.assign(DER_NULL, DER_NULL + 2);
   }

/*
* Create an AlgorithmIdentifier
*/
AlgorithmIdentifier::AlgorithmIdentifier(const std::string& alg_id,
                                         Encoding_Option option) :
   oid(OID::from_string(alg_id)),
   parameters()
   {
   const uint8_t DER_NULL[] = { 0x05, 0x00 };

   if(option == USE_NULL_PARAM)
      parameters.assign(DER_NULL, DER_NULL + 2);
   }

bool AlgorithmIdentifier::parameters_are_null() const
   {
   return (parameters.size() == 2 && (parameters[0] == 0x05) && (parameters[1] == 0x00));
   }

bool operator==(const AlgorithmIdentifier& a1, const AlgorithmIdentifier& a2)
   {
   if(a1.get_oid() != a2.get_oid())
      return false;

   /*
   * Treat NULL and empty as equivalent
   */
   if(a1.parameters_are_null_or_empty() &&
      a2.parameters_are_null_or_empty())
      {
      return true;
      }

   return (a1.get_parameters() == a2.get_parameters());
   }

bool operator!=(const AlgorithmIdentifier& a1, const AlgorithmIdentifier& a2)
   {
   return !(a1 == a2);
   }

/*
* DER encode an AlgorithmIdentifier
*/
void AlgorithmIdentifier::encode_into(DER_Encoder& codec) const
   {
   codec.start_cons(SEQUENCE)
      .encode(get_oid())
      .raw_bytes(get_parameters())
   .end_cons();
   }

/*
* Decode a BER encoded AlgorithmIdentifier
*/
void AlgorithmIdentifier::decode_from(BER_Decoder& codec)
   {
   codec.start_cons(SEQUENCE)
      .decode(oid)
      .raw_bytes(parameters)
   .end_cons();
   }

}
/*
* ASN.1 Internals
* (C) 1999-2007,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <sstream>

namespace Botan {

std::vector<uint8_t> ASN1_Object::BER_encode() const
   {
   std::vector<uint8_t> output;
   DER_Encoder der(output);
   this->encode_into(der);
   return output;
   }

/*
* Check a type invariant on BER data
*/
void BER_Object::assert_is_a(ASN1_Tag type_tag_, ASN1_Tag class_tag_,
                             const std::string& descr) const
   {
   if(this->is_a(type_tag_, class_tag_) == false)
      {
      std::stringstream msg;

      msg << "Tag mismatch when decoding " << descr << " got ";

      if(class_tag == NO_OBJECT && type_tag == NO_OBJECT)
         {
         msg << "EOF";
         }
      else
         {
         if(class_tag == UNIVERSAL || class_tag == CONSTRUCTED)
            {
            msg << asn1_tag_to_string(type_tag);
            }
         else
            {
            msg << std::to_string(type_tag);
            }

         msg << "/" << asn1_class_to_string(class_tag);
         }

      msg << " expected ";

      if(class_tag_ == UNIVERSAL || class_tag_ == CONSTRUCTED)
         {
         msg << asn1_tag_to_string(type_tag_);
         }
      else
         {
         msg << std::to_string(type_tag_);
         }

      msg << "/" << asn1_class_to_string(class_tag_);

      throw BER_Decoding_Error(msg.str());
      }
   }

bool BER_Object::is_a(ASN1_Tag type_tag_, ASN1_Tag class_tag_) const
   {
   return (type_tag == type_tag_ && class_tag == class_tag_);
   }

bool BER_Object::is_a(int type_tag_, ASN1_Tag class_tag_) const
   {
   return is_a(ASN1_Tag(type_tag_), class_tag_);
   }

void BER_Object::set_tagging(ASN1_Tag t, ASN1_Tag c)
   {
   type_tag = t;
   class_tag = c;
   }

std::string asn1_class_to_string(ASN1_Tag type)
   {
   switch(type)
      {
      case UNIVERSAL:
         return "UNIVERSAL";
      case CONSTRUCTED:
         return "CONSTRUCTED";
      case CONTEXT_SPECIFIC:
         return "CONTEXT_SPECIFIC";
      case APPLICATION:
         return "APPLICATION";
      case CONSTRUCTED | CONTEXT_SPECIFIC:
         return "PRIVATE";
      case Botan::NO_OBJECT:
         return "NO_OBJECT";
      default:
         return "CLASS(" + std::to_string(static_cast<size_t>(type)) + ")";
      }
   }

std::string asn1_tag_to_string(ASN1_Tag type)
   {
   switch(type)
      {
      case Botan::SEQUENCE:
         return "SEQUENCE";

      case Botan::SET:
         return "SET";

      case Botan::PRINTABLE_STRING:
         return "PRINTABLE STRING";

      case Botan::NUMERIC_STRING:
         return "NUMERIC STRING";

      case Botan::IA5_STRING:
         return "IA5 STRING";

      case Botan::T61_STRING:
         return "T61 STRING";

      case Botan::UTF8_STRING:
         return "UTF8 STRING";

      case Botan::VISIBLE_STRING:
         return "VISIBLE STRING";

      case Botan::BMP_STRING:
         return "BMP STRING";

      case Botan::UNIVERSAL_STRING:
         return "UNIVERSAL STRING";

      case Botan::UTC_TIME:
         return "UTC TIME";

      case Botan::GENERALIZED_TIME:
         return "GENERALIZED TIME";

      case Botan::OCTET_STRING:
         return "OCTET STRING";

      case Botan::BIT_STRING:
         return "BIT STRING";

      case Botan::ENUMERATED:
         return "ENUMERATED";

      case Botan::INTEGER:
         return "INTEGER";

      case Botan::NULL_TAG:
         return "NULL";

      case Botan::OBJECT_ID:
         return "OBJECT";

      case Botan::BOOLEAN:
         return "BOOLEAN";

      case Botan::NO_OBJECT:
         return "NO_OBJECT";

      default:
         return "TAG(" + std::to_string(static_cast<size_t>(type)) + ")";
      }
   }

/*
* BER Decoding Exceptions
*/
BER_Decoding_Error::BER_Decoding_Error(const std::string& str) :
   Decoding_Error("BER: " + str) {}

BER_Bad_Tag::BER_Bad_Tag(const std::string& str, ASN1_Tag tag) :
      BER_Decoding_Error(str + ": " + std::to_string(tag)) {}

BER_Bad_Tag::BER_Bad_Tag(const std::string& str,
                         ASN1_Tag tag1, ASN1_Tag tag2) :
   BER_Decoding_Error(str + ": " + std::to_string(tag1) + "/" + std::to_string(tag2)) {}

namespace ASN1 {

/*
* Put some arbitrary bytes into a SEQUENCE
*/
std::vector<uint8_t> put_in_sequence(const std::vector<uint8_t>& contents)
   {
   return ASN1::put_in_sequence(contents.data(), contents.size());
   }

std::vector<uint8_t> put_in_sequence(const uint8_t bits[], size_t len)
   {
   std::vector<uint8_t> output;
   DER_Encoder(output)
      .start_cons(SEQUENCE)
         .raw_bytes(bits, len)
      .end_cons();
   return output;
   }

/*
* Convert a BER object into a string object
*/
std::string to_string(const BER_Object& obj)
   {
   return std::string(cast_uint8_ptr_to_char(obj.bits()),
                      obj.length());
   }

/*
* Do heuristic tests for BER data
*/
bool maybe_BER(DataSource& source)
   {
   uint8_t first_u8;
   if(!source.peek_byte(first_u8))
      {
      BOTAN_ASSERT_EQUAL(source.read_byte(first_u8), 0, "Expected EOF");
      throw Stream_IO_Error("ASN1::maybe_BER: Source was empty");
      }

   if(first_u8 == (SEQUENCE | CONSTRUCTED))
      return true;
   return false;
   }

}

}
/*
* ASN.1 OID
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <algorithm>

namespace Botan {

namespace {

// returns empty on invalid
std::vector<uint32_t> parse_oid_str(const std::string& oid)
   {
   try
      {
      std::string elem;
      std::vector<uint32_t> oid_elems;

      for(char c : oid)
         {
         if(c == '.')
            {
            if(elem.empty())
               return std::vector<uint32_t>();
            oid_elems.push_back(to_u32bit(elem));
            elem.clear();
            }
         else
            {
            elem += c;
            }
         }

      if(elem.empty())
         return std::vector<uint32_t>();
      oid_elems.push_back(to_u32bit(elem));

      if(oid_elems.size() < 2)
         return std::vector<uint32_t>();

      return oid_elems;
      }
   catch(Invalid_Argument&) // thrown by to_u32bit
      {
      return std::vector<uint32_t>();
      }
   }

}

//static
OID OID::from_string(const std::string& str)
   {
   if(str.empty())
      throw Invalid_Argument("OID::from_string argument must be non-empty");

   const OID o = OIDS::str2oid_or_empty(str);
   if(o.has_value())
      return o;

   std::vector<uint32_t> raw = parse_oid_str(str);

   if(raw.size() > 0)
      return OID(std::move(raw));

   throw Lookup_Error("No OID associated with name " + str);
   }

/*
* ASN.1 OID Constructor
*/
OID::OID(const std::string& oid_str)
   {
   if(!oid_str.empty())
      {
      m_id = parse_oid_str(oid_str);

      if(m_id.size() < 2 || m_id[0] > 2)
         throw Invalid_OID(oid_str);
      if((m_id[0] == 0 || m_id[0] == 1) && m_id[1] > 39)
         throw Invalid_OID(oid_str);
      }
   }

/*
* Return this OID as a string
*/
std::string OID::to_string() const
   {
   std::ostringstream oss;
   oss.imbue(std::locale("C"));
   for(size_t i = 0; i != m_id.size(); ++i)
      {
      oss << m_id[i];
      if(i != m_id.size() - 1)
         oss << ".";
      }
   return oss.str();
   }

std::string OID::to_formatted_string() const
   {
   const std::string s = OIDS::oid2str_or_empty(*this);
   if(!s.empty())
      return s;
   return this->to_string();
   }

/*
* Append another component to the OID
*/
OID operator+(const OID& oid, uint32_t new_component)
   {
   std::vector<uint32_t> val = oid.get_components();
   val.push_back(new_component);
   return OID(std::move(val));
   }

/*
* Compare two OIDs
*/
bool operator<(const OID& a, const OID& b)
   {
   const std::vector<uint32_t>& oid1 = a.get_components();
   const std::vector<uint32_t>& oid2 = b.get_components();

   return std::lexicographical_compare(oid1.begin(), oid1.end(),
                                       oid2.begin(), oid2.end());
   }

/*
* DER encode an OBJECT IDENTIFIER
*/
void OID::encode_into(DER_Encoder& der) const
   {
   if(m_id.size() < 2)
      throw Invalid_Argument("OID::encode_into: OID is invalid");

   std::vector<uint8_t> encoding;

   if(m_id[0] > 2 || m_id[1] >= 40)
      throw Encoding_Error("Invalid OID prefix, cannot encode");

   encoding.push_back(static_cast<uint8_t>(40 * m_id[0] + m_id[1]));

   for(size_t i = 2; i != m_id.size(); ++i)
      {
      if(m_id[i] == 0)
         encoding.push_back(0);
      else
         {
         size_t blocks = high_bit(m_id[i]) + 6;
         blocks = (blocks - (blocks % 7)) / 7;

         BOTAN_ASSERT(blocks > 0, "Math works");

         for(size_t j = 0; j != blocks - 1; ++j)
            encoding.push_back(0x80 | ((m_id[i] >> 7*(blocks-j-1)) & 0x7F));
         encoding.push_back(m_id[i] & 0x7F);
         }
      }
   der.add_object(OBJECT_ID, UNIVERSAL, encoding);
   }

/*
* Decode a BER encoded OBJECT IDENTIFIER
*/
void OID::decode_from(BER_Decoder& decoder)
   {
   BER_Object obj = decoder.get_next_object();
   if(obj.tagging() != OBJECT_ID)
      throw BER_Bad_Tag("Error decoding OID, unknown tag", obj.tagging());

   const size_t length = obj.length();
   const uint8_t* bits = obj.bits();

   if(length < 2 && !(length == 1 && bits[0] == 0))
      {
      throw BER_Decoding_Error("OID encoding is too short");
      }

   m_id.clear();
   m_id.push_back(bits[0] / 40);
   m_id.push_back(bits[0] % 40);

   size_t i = 0;
   while(i != length - 1)
      {
      uint32_t component = 0;
      while(i != length - 1)
         {
         ++i;

         if(component >> (32-7))
            throw Decoding_Error("OID component overflow");

         component = (component << 7) + (bits[i] & 0x7F);

         if(!(bits[i] & 0x80))
            break;
         }
      m_id.push_back(component);
      }
   }

}
/*
* (C) 2014,2015,2017 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <iomanip>
#include <cctype>

namespace Botan {

namespace {

bool all_printable_chars(const uint8_t bits[], size_t bits_len)
   {
   for(size_t i = 0; i != bits_len; ++i)
      {
      int c = bits[i];
      if(c > 127)
         return false;

      if((std::isalnum(c) || c == '.' || c == ':' || c == '/' || c == '-') == false)
         return false;
      }
   return true;
   }

/*
* Special hack to handle GeneralName [2] and [6] (DNS name and URI)
*/
bool possibly_a_general_name(const uint8_t bits[], size_t bits_len)
   {
   if(bits_len <= 2)
      return false;

   if(bits[0] != 0x82 && bits[0] != 0x86)
      return false;

   if(bits[1] != bits_len - 2)
      return false;

   if(all_printable_chars(bits + 2, bits_len - 2) == false)
      return false;

   return true;
   }

}

std::string ASN1_Formatter::print(const uint8_t in[], size_t len) const
   {
   std::ostringstream output;
   print_to_stream(output, in, len);
   return output.str();
   }

void ASN1_Formatter::print_to_stream(std::ostream& output,
                                     const uint8_t in[],
                                     size_t len) const
   {
   BER_Decoder dec(in, len);
   decode(output, dec, 0);
   }

void ASN1_Formatter::decode(std::ostream& output,
                            BER_Decoder& decoder,
                            size_t level) const
   {
   BER_Object obj = decoder.get_next_object();

   const bool recurse_deeper = (m_max_depth == 0 || level < m_max_depth);

   while(obj.is_set())
      {
      const ASN1_Tag type_tag = obj.type();
      const ASN1_Tag class_tag = obj.get_class();
      const size_t length = obj.length();

      /* hack to insert the tag+length back in front of the stuff now
         that we've gotten the type info */
      std::vector<uint8_t> bits;
      DER_Encoder(bits).add_object(type_tag, class_tag, obj.bits(), obj.length());

      BER_Decoder data(bits);

      if(class_tag & CONSTRUCTED)
         {
         BER_Decoder cons_info(obj.bits(), obj.length());

         if(recurse_deeper)
            {
            output << format(type_tag, class_tag, level, length, "");
            decode(output, cons_info, level + 1); // recurse
            }
         else
            {
            output << format(type_tag, class_tag, level, length,
                             format_bin(type_tag, class_tag, bits));
            }
         }
      else if((class_tag & APPLICATION) || (class_tag & CONTEXT_SPECIFIC))
         {
         bool success_parsing_cs = false;

         if(m_print_context_specific)
            {
            try
               {
               if(possibly_a_general_name(bits.data(), bits.size()))
                  {
                  output << format(type_tag, class_tag, level, level,
                                   std::string(cast_uint8_ptr_to_char(&bits[2]), bits.size() - 2));
                  success_parsing_cs = true;
                  }
               else if(recurse_deeper)
                  {
                  std::vector<uint8_t> inner_bits;
                  data.decode(inner_bits, type_tag);

                  BER_Decoder inner(inner_bits);
                  std::ostringstream inner_data;
                  decode(inner_data, inner, level + 1); // recurse
                  output << inner_data.str();
                  success_parsing_cs = true;
                  }
               }
            catch(...)
               {
               }
            }

         if(success_parsing_cs == false)
            {
            output << format(type_tag, class_tag, level, length,
                             format_bin(type_tag, class_tag, bits));
            }
         }
      else if(type_tag == OBJECT_ID)
         {
         OID oid;
         data.decode(oid);

         std::string out = OIDS::oid2str_or_empty(oid);
         if(out.empty())
            {
            out = oid.to_string();
            }
         else
            {
            out += " [" + oid.to_string() + "]";
            }

         output << format(type_tag, class_tag, level, length, out);
         }
      else if(type_tag == INTEGER || type_tag == ENUMERATED)
         {
         BigInt number;

         if(type_tag == INTEGER)
            {
            data.decode(number);
            }
         else if(type_tag == ENUMERATED)
            {
            data.decode(number, ENUMERATED, class_tag);
            }

         std::vector<uint8_t> rep = BigInt::encode(number);
         if(rep.empty()) // if zero
            rep.resize(1);

         output << format(type_tag, class_tag, level, length, hex_encode(rep));
         }
      else if(type_tag == BOOLEAN)
         {
         bool boolean;
         data.decode(boolean);
         output << format(type_tag, class_tag, level, length, (boolean ? "true" : "false"));
         }
      else if(type_tag == NULL_TAG)
         {
         output << format(type_tag, class_tag, level, length, "");
         }
      else if(type_tag == OCTET_STRING || type_tag == BIT_STRING)
         {
         std::vector<uint8_t> decoded_bits;
         data.decode(decoded_bits, type_tag);
         bool printing_octet_string_worked = false;

         if(recurse_deeper)
            {
            try
               {
               BER_Decoder inner(decoded_bits);

               std::ostringstream inner_data;
               decode(inner_data, inner, level + 1); // recurse

               output << format(type_tag, class_tag, level, length, "");
               output << inner_data.str();
               printing_octet_string_worked = true;
               }
            catch(...)
               {
               }
            }

         if(!printing_octet_string_worked)
            {
            output << format(type_tag, class_tag, level, length,
                             format_bin(type_tag, class_tag, decoded_bits));
            }
         }
      else if(ASN1_String::is_string_type(type_tag))
         {
         ASN1_String str;
         data.decode(str);
         output << format(type_tag, class_tag, level, length, str.value());
         }
      else if(type_tag == UTC_TIME || type_tag == GENERALIZED_TIME)
         {
         ASN1_Time time;
         data.decode(time);
         output << format(type_tag, class_tag, level, length, time.readable_string());
         }
      else
         {
         output << "Unknown ASN.1 tag class=" << static_cast<int>(class_tag)
                << " type=" << static_cast<int>(type_tag) << "\n";
         }

      obj = decoder.get_next_object();
      }
   }

namespace {

std::string format_type(ASN1_Tag type_tag, ASN1_Tag class_tag)
   {
   if(class_tag == UNIVERSAL)
      return asn1_tag_to_string(type_tag);

   if(class_tag == CONSTRUCTED && (type_tag == SEQUENCE || type_tag == SET))
      return asn1_tag_to_string(type_tag);

   std::string name;

   if(class_tag & CONSTRUCTED)
      name += "cons ";

   name += "[" + std::to_string(type_tag) + "]";

   if(class_tag & APPLICATION)
      {
      name += " appl";
      }
   if(class_tag & CONTEXT_SPECIFIC)
      {
      name += " context";
      }

   return name;
   }

}

std::string ASN1_Pretty_Printer::format(ASN1_Tag type_tag,
                                        ASN1_Tag class_tag,
                                        size_t level,
                                        size_t length,
                                        const std::string& value) const
   {
   bool should_skip = false;

   if(value.length() > m_print_limit)
      {
      should_skip = true;
      }

   if((type_tag == OCTET_STRING || type_tag == BIT_STRING) &&
      value.length() > m_print_binary_limit)
      {
      should_skip = true;
      }

   level += m_initial_level;

   std::ostringstream oss;

   oss << "  d=" << std::setw(2) << level
       << ", l=" << std::setw(4) << length << ":"
       << std::string(level + 1, ' ') << format_type(type_tag, class_tag);

   if(value != "" && !should_skip)
      {
      const size_t current_pos = static_cast<size_t>(oss.tellp());
      const size_t spaces_to_align =
         (current_pos >= m_value_column) ? 1 : (m_value_column - current_pos);

      oss << std::string(spaces_to_align, ' ') << value;
      }

   oss << "\n";

   return oss.str();
   }

std::string ASN1_Pretty_Printer::format_bin(ASN1_Tag /*type_tag*/,
                                            ASN1_Tag /*class_tag*/,
                                            const std::vector<uint8_t>& vec) const
   {
   if(all_printable_chars(vec.data(), vec.size()))
      {
      return std::string(cast_uint8_ptr_to_char(vec.data()), vec.size());
      }
   else
      return hex_encode(vec);
   }

}
/*
* Simple ASN.1 String Types
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

/*
* Choose an encoding for the string
*/
ASN1_Tag choose_encoding(const std::string& str)
   {
   static const uint8_t IS_PRINTABLE[256] = {
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01,
      0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
      0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
      0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
      0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
      0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
      0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00 };

   for(size_t i = 0; i != str.size(); ++i)
      {
      if(!IS_PRINTABLE[static_cast<uint8_t>(str[i])])
         {
         return UTF8_STRING;
         }
      }
   return PRINTABLE_STRING;
   }

void assert_is_string_type(ASN1_Tag tag)
   {
   if(!ASN1_String::is_string_type(tag))
      {
      throw Invalid_Argument("ASN1_String: Unknown string type " +
                             std::to_string(tag));
      }
   }

}

//static
bool ASN1_String::is_string_type(ASN1_Tag tag)
   {
   return (tag == NUMERIC_STRING ||
           tag == PRINTABLE_STRING ||
           tag == VISIBLE_STRING ||
           tag == T61_STRING ||
           tag == IA5_STRING ||
           tag == UTF8_STRING ||
           tag == BMP_STRING ||
           tag == UNIVERSAL_STRING);
   }


/*
* Create an ASN1_String
*/
ASN1_String::ASN1_String(const std::string& str, ASN1_Tag t) : m_utf8_str(str), m_tag(t)
   {
   if(m_tag == DIRECTORY_STRING)
      {
      m_tag = choose_encoding(m_utf8_str);
      }

   assert_is_string_type(m_tag);
   }

/*
* Create an ASN1_String
*/
ASN1_String::ASN1_String(const std::string& str) :
   m_utf8_str(str),
   m_tag(choose_encoding(m_utf8_str))
   {}

/*
* Return this string in ISO 8859-1 encoding
*/
std::string ASN1_String::iso_8859() const
   {
   return utf8_to_latin1(m_utf8_str);
   }

/*
* DER encode an ASN1_String
*/
void ASN1_String::encode_into(DER_Encoder& encoder) const
   {
   if(m_data.empty())
      {
      encoder.add_object(tagging(), UNIVERSAL, m_utf8_str);
      }
   else
      {
      // If this string was decoded, reserialize using original encoding
      encoder.add_object(tagging(), UNIVERSAL, m_data.data(), m_data.size());
      }
   }

/*
* Decode a BER encoded ASN1_String
*/
void ASN1_String::decode_from(BER_Decoder& source)
   {
   BER_Object obj = source.get_next_object();

   assert_is_string_type(obj.type());

   m_tag = obj.type();
   m_data.assign(obj.bits(), obj.bits() + obj.length());

   if(m_tag == BMP_STRING)
      {
      m_utf8_str = ucs2_to_utf8(m_data.data(), m_data.size());
      }
   else if(m_tag == UNIVERSAL_STRING)
      {
      m_utf8_str = ucs4_to_utf8(m_data.data(), m_data.size());
      }
   else
      {
      // All other supported string types are UTF-8 or some subset thereof
      m_utf8_str = ASN1::to_string(obj);
      }
   }

}
/*
* X.509 Time Types
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

ASN1_Time::ASN1_Time(const std::chrono::system_clock::time_point& time)
   {
   calendar_point cal = calendar_value(time);

   m_year   = cal.get_year();
   m_month  = cal.get_month();
   m_day    = cal.get_day();
   m_hour   = cal.get_hour();
   m_minute = cal.get_minutes();
   m_second = cal.get_seconds();

   m_tag = (m_year >= 2050) ? GENERALIZED_TIME : UTC_TIME;
   }

ASN1_Time::ASN1_Time(const std::string& t_spec, ASN1_Tag tag)
   {
   set_to(t_spec, tag);
   }

void ASN1_Time::encode_into(DER_Encoder& der) const
   {
   BOTAN_ARG_CHECK(m_tag == UTC_TIME || m_tag == GENERALIZED_TIME,
                   "ASN1_Time: Bad encoding tag");

   der.add_object(m_tag, UNIVERSAL, to_string());
   }

void ASN1_Time::decode_from(BER_Decoder& source)
   {
   BER_Object ber_time = source.get_next_object();

   set_to(ASN1::to_string(ber_time), ber_time.type());
   }

std::string ASN1_Time::to_string() const
   {
   if(time_is_set() == false)
      throw Invalid_State("ASN1_Time::to_string: No time set");

   uint32_t full_year = m_year;

   if(m_tag == UTC_TIME)
      {
      if(m_year < 1950 || m_year >= 2050)
         throw Encoding_Error("ASN1_Time: The time " + readable_string() +
                              " cannot be encoded as a UTCTime");

      full_year = (m_year >= 2000) ? (m_year - 2000) : (m_year - 1900);
      }

   const uint64_t YEAR_FACTOR = 10000000000ULL;
   const uint64_t MON_FACTOR  = 100000000;
   const uint64_t DAY_FACTOR  = 1000000;
   const uint64_t HOUR_FACTOR = 10000;
   const uint64_t MIN_FACTOR  = 100;

   const uint64_t int_repr =
      YEAR_FACTOR * full_year +
      MON_FACTOR * m_month +
      DAY_FACTOR * m_day +
      HOUR_FACTOR * m_hour +
      MIN_FACTOR * m_minute +
      m_second;

   std::string repr = std::to_string(int_repr) + "Z";

   uint32_t desired_size = (m_tag == UTC_TIME) ? 13 : 15;

   while(repr.size() < desired_size)
      repr = "0" + repr;

   return repr;
   }

std::string ASN1_Time::readable_string() const
   {
   if(time_is_set() == false)
      throw Invalid_State("ASN1_Time::readable_string: No time set");

   // desired format: "%04d/%02d/%02d %02d:%02d:%02d UTC"
   std::stringstream output;
   output << std::setfill('0')
          << std::setw(4) << m_year << "/"
          << std::setw(2) << m_month << "/"
          << std::setw(2) << m_day
          << " "
          << std::setw(2) << m_hour << ":"
          << std::setw(2) << m_minute << ":"
          << std::setw(2) << m_second
          << " UTC";

   return output.str();
   }

bool ASN1_Time::time_is_set() const
   {
   return (m_year != 0);
   }

int32_t ASN1_Time::cmp(const ASN1_Time& other) const
   {
   if(time_is_set() == false)
      throw Invalid_State("ASN1_Time::cmp: No time set");

   const int32_t EARLIER = -1, LATER = 1, SAME_TIME = 0;

   if(m_year < other.m_year)     return EARLIER;
   if(m_year > other.m_year)     return LATER;
   if(m_month < other.m_month)   return EARLIER;
   if(m_month > other.m_month)   return LATER;
   if(m_day < other.m_day)       return EARLIER;
   if(m_day > other.m_day)       return LATER;
   if(m_hour < other.m_hour)     return EARLIER;
   if(m_hour > other.m_hour)     return LATER;
   if(m_minute < other.m_minute) return EARLIER;
   if(m_minute > other.m_minute) return LATER;
   if(m_second < other.m_second) return EARLIER;
   if(m_second > other.m_second) return LATER;

   return SAME_TIME;
   }

void ASN1_Time::set_to(const std::string& t_spec, ASN1_Tag spec_tag)
   {
   if(spec_tag == UTC_OR_GENERALIZED_TIME)
      {
      try
         {
         set_to(t_spec, GENERALIZED_TIME);
         return;
         }
      catch(Invalid_Argument&) {} // Not a generalized time. Continue

      try
         {
         set_to(t_spec, UTC_TIME);
         return;
         }
      catch(Invalid_Argument&) {} // Not a UTC time. Continue

      throw Invalid_Argument("Time string could not be parsed as GeneralizedTime or UTCTime.");
      }

   BOTAN_ASSERT(spec_tag == UTC_TIME || spec_tag == GENERALIZED_TIME, "Invalid tag.");

   BOTAN_ARG_CHECK(t_spec.size() > 0, "Time string must not be empty.");

   BOTAN_ARG_CHECK(t_spec.back() == 'Z', "Botan does not support times with timezones other than Z");

   if(spec_tag == GENERALIZED_TIME)
      {
      BOTAN_ARG_CHECK(t_spec.size() == 15, "Invalid GeneralizedTime string");
      }
   else if(spec_tag == UTC_TIME)
      {
      BOTAN_ARG_CHECK(t_spec.size() == 13, "Invalid UTCTime string");
      }

   const size_t YEAR_SIZE = (spec_tag == UTC_TIME) ? 2 : 4;

   std::vector<std::string> params;
   std::string current;

   for(size_t j = 0; j != YEAR_SIZE; ++j)
      current += t_spec[j];
   params.push_back(current);
   current.clear();

   for(size_t j = YEAR_SIZE; j != t_spec.size() - 1; ++j)
      {
      current += t_spec[j];
      if(current.size() == 2)
         {
         params.push_back(current);
         current.clear();
         }
      }

   m_year   = to_u32bit(params[0]);
   m_month  = to_u32bit(params[1]);
   m_day    = to_u32bit(params[2]);
   m_hour   = to_u32bit(params[3]);
   m_minute = to_u32bit(params[4]);
   m_second = (params.size() == 6) ? to_u32bit(params[5]) : 0;
   m_tag    = spec_tag;

   if(spec_tag == UTC_TIME)
      {
      if(m_year >= 50) m_year += 1900;
      else             m_year += 2000;
      }

   if(!passes_sanity_check())
      throw Invalid_Argument("Time " + t_spec + " does not seem to be valid");
   }

/*
* Do a general sanity check on the time
*/
bool ASN1_Time::passes_sanity_check() const
   {
   // AppVeyor's trust store includes a cert with expiration date in 3016 ...
   if(m_year < 1950 || m_year > 3100)
      return false;
   if(m_month == 0 || m_month > 12)
      return false;

   const uint32_t days_in_month[12] = { 31, 28+1, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

   if(m_day == 0 || m_day > days_in_month[m_month-1])
      return false;

   if(m_month == 2 && m_day == 29)
      {
      if(m_year % 4 != 0)
         return false; // not a leap year

      if(m_year % 100 == 0 && m_year % 400 != 0)
         return false;
      }

   if(m_hour >= 24 || m_minute >= 60 || m_second > 60)
      return false;

   if (m_tag == UTC_TIME)
      {
      /*
      UTCTime limits the value of components such that leap seconds
      are not covered. See "UNIVERSAL 23" in "Information technology
      Abstract Syntax Notation One (ASN.1): Specification of basic notation"

      http://www.itu.int/ITU-T/studygroups/com17/languages/
      */
      if(m_second > 59)
         {
         return false;
         }
      }

   return true;
   }

std::chrono::system_clock::time_point ASN1_Time::to_std_timepoint() const
   {
   return calendar_point(m_year, m_month, m_day, m_hour, m_minute, m_second).to_std_timepoint();
   }

uint64_t ASN1_Time::time_since_epoch() const
   {
   auto tp = this->to_std_timepoint();
   return std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch()).count();
   }

/*
* Compare two ASN1_Times for in various ways
*/
bool operator==(const ASN1_Time& t1, const ASN1_Time& t2)
   { return (t1.cmp(t2) == 0); }
bool operator!=(const ASN1_Time& t1, const ASN1_Time& t2)
   { return (t1.cmp(t2) != 0); }

bool operator<=(const ASN1_Time& t1, const ASN1_Time& t2)
   { return (t1.cmp(t2) <= 0); }
bool operator>=(const ASN1_Time& t1, const ASN1_Time& t2)
   { return (t1.cmp(t2) >= 0); }

bool operator<(const ASN1_Time& t1, const ASN1_Time& t2)
   { return (t1.cmp(t2) < 0); }
bool operator>(const ASN1_Time& t1, const ASN1_Time& t2)
   { return (t1.cmp(t2) > 0); }

}
/*
* BER Decoder
* (C) 1999-2008,2015,2017,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

/*
* This value is somewhat arbitrary. OpenSSL allows up to 128 nested
* indefinite length sequences. If you increase this, also increase the
* limit in the test in test_asn1.cpp
*/
const size_t ALLOWED_EOC_NESTINGS = 16;

/*
* BER decode an ASN.1 type tag
*/
size_t decode_tag(DataSource* ber, ASN1_Tag& type_tag, ASN1_Tag& class_tag)
   {
   uint8_t b;
   if(!ber->read_byte(b))
      {
      class_tag = type_tag = NO_OBJECT;
      return 0;
      }

   if((b & 0x1F) != 0x1F)
      {
      type_tag = ASN1_Tag(b & 0x1F);
      class_tag = ASN1_Tag(b & 0xE0);
      return 1;
      }

   size_t tag_bytes = 1;
   class_tag = ASN1_Tag(b & 0xE0);

   size_t tag_buf = 0;
   while(true)
      {
      if(!ber->read_byte(b))
         throw BER_Decoding_Error("Long-form tag truncated");
      if(tag_buf & 0xFF000000)
         throw BER_Decoding_Error("Long-form tag overflowed 32 bits");
      ++tag_bytes;
      tag_buf = (tag_buf << 7) | (b & 0x7F);
      if((b & 0x80) == 0) break;
      }
   type_tag = ASN1_Tag(tag_buf);
   return tag_bytes;
   }

/*
* Find the EOC marker
*/
size_t find_eoc(DataSource* src, size_t allow_indef);

/*
* BER decode an ASN.1 length field
*/
size_t decode_length(DataSource* ber, size_t& field_size, size_t allow_indef)
   {
   uint8_t b;
   if(!ber->read_byte(b))
      throw BER_Decoding_Error("Length field not found");
   field_size = 1;
   if((b & 0x80) == 0)
      return b;

   field_size += (b & 0x7F);
   if(field_size > 5)
      throw BER_Decoding_Error("Length field is too large");

   if(field_size == 1)
      {
      if(allow_indef == 0)
         {
         throw BER_Decoding_Error("Nested EOC markers too deep, rejecting to avoid stack exhaustion");
         }
      else
         {
         return find_eoc(ber, allow_indef - 1);
         }
      }

   size_t length = 0;

   for(size_t i = 0; i != field_size - 1; ++i)
      {
      if(get_byte(0, length) != 0)
         throw BER_Decoding_Error("Field length overflow");
      if(!ber->read_byte(b))
         throw BER_Decoding_Error("Corrupted length field");
      length = (length << 8) | b;
      }
   return length;
   }

/*
* Find the EOC marker
*/
size_t find_eoc(DataSource* ber, size_t allow_indef)
   {
   secure_vector<uint8_t> buffer(BOTAN_DEFAULT_BUFFER_SIZE), data;

   while(true)
      {
      const size_t got = ber->peek(buffer.data(), buffer.size(), data.size());
      if(got == 0)
         break;

      data += std::make_pair(buffer.data(), got);
      }

   DataSource_Memory source(data);
   data.clear();

   size_t length = 0;
   while(true)
      {
      ASN1_Tag type_tag, class_tag;
      size_t tag_size = decode_tag(&source, type_tag, class_tag);
      if(type_tag == NO_OBJECT)
         break;

      size_t length_size = 0;
      size_t item_size = decode_length(&source, length_size, allow_indef);
      source.discard_next(item_size);

      length = BOTAN_CHECKED_ADD(length, item_size);
      length = BOTAN_CHECKED_ADD(length, tag_size);
      length = BOTAN_CHECKED_ADD(length, length_size);

      if(type_tag == EOC && class_tag == UNIVERSAL)
         break;
      }
   return length;
   }

class DataSource_BERObject final : public DataSource
   {
   public:
      size_t read(uint8_t out[], size_t length) override
         {
         BOTAN_ASSERT_NOMSG(m_offset <= m_obj.length());
         const size_t got = std::min<size_t>(m_obj.length() - m_offset, length);
         copy_mem(out, m_obj.bits() + m_offset, got);
         m_offset += got;
         return got;
         }

      size_t peek(uint8_t out[], size_t length, size_t peek_offset) const override
         {
         BOTAN_ASSERT_NOMSG(m_offset <= m_obj.length());
         const size_t bytes_left = m_obj.length() - m_offset;

         if(peek_offset >= bytes_left)
            return 0;

         const size_t got = std::min(bytes_left - peek_offset, length);
         copy_mem(out, m_obj.bits() + peek_offset, got);
         return got;
         }

      bool check_available(size_t n) override
         {
         BOTAN_ASSERT_NOMSG(m_offset <= m_obj.length());
         return (n <= (m_obj.length() - m_offset));
         }

      bool end_of_data() const override
         {
         return get_bytes_read() == m_obj.length();
         }

      size_t get_bytes_read() const override { return m_offset; }

      explicit DataSource_BERObject(BER_Object&& obj) : m_obj(std::move(obj)), m_offset(0) {}

   private:
      BER_Object m_obj;
      size_t m_offset;
   };

}

/*
* Check if more objects are there
*/
bool BER_Decoder::more_items() const
   {
   if(m_source->end_of_data() && !m_pushed.is_set())
      return false;
   return true;
   }

/*
* Verify that no bytes remain in the source
*/
BER_Decoder& BER_Decoder::verify_end()
   {
   return verify_end("BER_Decoder::verify_end called, but data remains");
   }

/*
* Verify that no bytes remain in the source
*/
BER_Decoder& BER_Decoder::verify_end(const std::string& err)
   {
   if(!m_source->end_of_data() || m_pushed.is_set())
      throw Decoding_Error(err);
   return (*this);
   }

/*
* Discard all the bytes remaining in the source
*/
BER_Decoder& BER_Decoder::discard_remaining()
   {
   uint8_t buf;
   while(m_source->read_byte(buf))
      {}
   return (*this);
   }

/*
* Return the BER encoding of the next object
*/
BER_Object BER_Decoder::get_next_object()
   {
   BER_Object next;

   if(m_pushed.is_set())
      {
      std::swap(next, m_pushed);
      return next;
      }

   for(;;)
      {
      ASN1_Tag type_tag, class_tag;
      decode_tag(m_source, type_tag, class_tag);
      next.set_tagging(type_tag, class_tag);
      if(next.is_set() == false) // no more objects
         return next;

      size_t field_size;
      const size_t length = decode_length(m_source, field_size, ALLOWED_EOC_NESTINGS);
      if(!m_source->check_available(length))
         throw BER_Decoding_Error("Value truncated");

      uint8_t* out = next.mutable_bits(length);
      if(m_source->read(out, length) != length)
         throw BER_Decoding_Error("Value truncated");

      if(next.tagging() == EOC)
         continue;
      else
         break;
      }

   return next;
   }

/*
* Push a object back into the stream
*/
void BER_Decoder::push_back(const BER_Object& obj)
   {
   if(m_pushed.is_set())
      throw Invalid_State("BER_Decoder: Only one push back is allowed");
   m_pushed = obj;
   }

void BER_Decoder::push_back(BER_Object&& obj)
   {
   if(m_pushed.is_set())
      throw Invalid_State("BER_Decoder: Only one push back is allowed");
   m_pushed = std::move(obj);
   }

BER_Decoder BER_Decoder::start_cons(ASN1_Tag type_tag, ASN1_Tag class_tag)
   {
   BER_Object obj = get_next_object();
   obj.assert_is_a(type_tag, ASN1_Tag(class_tag | CONSTRUCTED));
   return BER_Decoder(std::move(obj), this);
   }

/*
* Finish decoding a CONSTRUCTED type
*/
BER_Decoder& BER_Decoder::end_cons()
   {
   if(!m_parent)
      throw Invalid_State("BER_Decoder::end_cons called with null parent");
   if(!m_source->end_of_data())
      throw Decoding_Error("BER_Decoder::end_cons called with data left");
   return (*m_parent);
   }

BER_Decoder::BER_Decoder(BER_Object&& obj, BER_Decoder* parent)
   {
   m_data_src.reset(new DataSource_BERObject(std::move(obj)));
   m_source = m_data_src.get();
   m_parent = parent;
   }

/*
* BER_Decoder Constructor
*/
BER_Decoder::BER_Decoder(DataSource& src)
   {
   m_source = &src;
   }

/*
* BER_Decoder Constructor
 */
BER_Decoder::BER_Decoder(const uint8_t data[], size_t length)
   {
   m_data_src.reset(new DataSource_Memory(data, length));
   m_source = m_data_src.get();
   }

/*
* BER_Decoder Constructor
*/
BER_Decoder::BER_Decoder(const secure_vector<uint8_t>& data)
   {
   m_data_src.reset(new DataSource_Memory(data));
   m_source = m_data_src.get();
   }

/*
* BER_Decoder Constructor
*/
BER_Decoder::BER_Decoder(const std::vector<uint8_t>& data)
   {
   m_data_src.reset(new DataSource_Memory(data.data(), data.size()));
   m_source = m_data_src.get();
   }

/*
* BER_Decoder Copy Constructor
*/
BER_Decoder::BER_Decoder(const BER_Decoder& other)
   {
   m_source = other.m_source;

   // take ownership
   std::swap(m_data_src, other.m_data_src);
   m_parent = other.m_parent;
   }

/*
* Request for an object to decode itself
*/
BER_Decoder& BER_Decoder::decode(ASN1_Object& obj,
                                 ASN1_Tag, ASN1_Tag)
   {
   obj.decode_from(*this);
   return (*this);
   }

/*
* Decode a BER encoded NULL
*/
BER_Decoder& BER_Decoder::decode_null()
   {
   BER_Object obj = get_next_object();
   obj.assert_is_a(NULL_TAG, UNIVERSAL);
   if(obj.length() > 0)
      throw BER_Decoding_Error("NULL object had nonzero size");
   return (*this);
   }

BER_Decoder& BER_Decoder::decode_octet_string_bigint(BigInt& out)
   {
   secure_vector<uint8_t> out_vec;
   decode(out_vec, OCTET_STRING);
   out = BigInt::decode(out_vec.data(), out_vec.size());
   return (*this);
   }

/*
* Decode a BER encoded BOOLEAN
*/
BER_Decoder& BER_Decoder::decode(bool& out,
                                 ASN1_Tag type_tag, ASN1_Tag class_tag)
   {
   BER_Object obj = get_next_object();
   obj.assert_is_a(type_tag, class_tag);

   if(obj.length() != 1)
      throw BER_Decoding_Error("BER boolean value had invalid size");

   out = (obj.bits()[0]) ? true : false;
   return (*this);
   }

/*
* Decode a small BER encoded INTEGER
*/
BER_Decoder& BER_Decoder::decode(size_t& out,
                                 ASN1_Tag type_tag,
                                 ASN1_Tag class_tag)
   {
   BigInt integer;
   decode(integer, type_tag, class_tag);

   if(integer.is_negative())
      throw BER_Decoding_Error("Decoded small integer value was negative");

   if(integer.bits() > 32)
      throw BER_Decoding_Error("Decoded integer value larger than expected");

   out = 0;
   for(size_t i = 0; i != 4; ++i)
      out = (out << 8) | integer.byte_at(3-i);

   return (*this);
   }

/*
* Decode a small BER encoded INTEGER
*/
uint64_t BER_Decoder::decode_constrained_integer(ASN1_Tag type_tag,
                                                 ASN1_Tag class_tag,
                                                 size_t T_bytes)
   {
   if(T_bytes > 8)
      throw BER_Decoding_Error("Can't decode small integer over 8 bytes");

   BigInt integer;
   decode(integer, type_tag, class_tag);

   if(integer.bits() > 8*T_bytes)
      throw BER_Decoding_Error("Decoded integer value larger than expected");

   uint64_t out = 0;
   for(size_t i = 0; i != 8; ++i)
      out = (out << 8) | integer.byte_at(7-i);

   return out;
   }

/*
* Decode a BER encoded INTEGER
*/
BER_Decoder& BER_Decoder::decode(BigInt& out,
                                 ASN1_Tag type_tag,
                                 ASN1_Tag class_tag)
   {
   BER_Object obj = get_next_object();
   obj.assert_is_a(type_tag, class_tag);

   if(obj.length() == 0)
      {
      out = 0;
      }
   else
      {
      const bool negative = (obj.bits()[0] & 0x80) ? true : false;

      if(negative)
         {
         secure_vector<uint8_t> vec(obj.bits(), obj.bits() + obj.length());
         for(size_t i = obj.length(); i > 0; --i)
            if(vec[i-1]--)
               break;
         for(size_t i = 0; i != obj.length(); ++i)
            vec[i] = ~vec[i];
         out = BigInt(vec.data(), vec.size());
         out.flip_sign();
         }
      else
         {
         out = BigInt(obj.bits(), obj.length());
         }
      }

   return (*this);
   }

namespace {

template<typename Alloc>
void asn1_decode_binary_string(std::vector<uint8_t, Alloc>& buffer,
                               const BER_Object& obj,
                               ASN1_Tag real_type,
                               ASN1_Tag type_tag,
                               ASN1_Tag class_tag)
   {
   obj.assert_is_a(type_tag, class_tag);

   if(real_type == OCTET_STRING)
      {
      buffer.assign(obj.bits(), obj.bits() + obj.length());
      }
   else
      {
      if(obj.length() == 0)
         throw BER_Decoding_Error("Invalid BIT STRING");
      if(obj.bits()[0] >= 8)
         throw BER_Decoding_Error("Bad number of unused bits in BIT STRING");

      buffer.resize(obj.length() - 1);

      if(obj.length() > 1)
         copy_mem(buffer.data(), obj.bits() + 1, obj.length() - 1);
      }
   }

}

/*
* BER decode a BIT STRING or OCTET STRING
*/
BER_Decoder& BER_Decoder::decode(secure_vector<uint8_t>& buffer,
                                 ASN1_Tag real_type,
                                 ASN1_Tag type_tag, ASN1_Tag class_tag)
   {
   if(real_type != OCTET_STRING && real_type != BIT_STRING)
      throw BER_Bad_Tag("Bad tag for {BIT,OCTET} STRING", real_type);

   asn1_decode_binary_string(buffer, get_next_object(), real_type, type_tag, class_tag);
   return (*this);
   }

BER_Decoder& BER_Decoder::decode(std::vector<uint8_t>& buffer,
                                 ASN1_Tag real_type,
                                 ASN1_Tag type_tag, ASN1_Tag class_tag)
   {
   if(real_type != OCTET_STRING && real_type != BIT_STRING)
      throw BER_Bad_Tag("Bad tag for {BIT,OCTET} STRING", real_type);

   asn1_decode_binary_string(buffer, get_next_object(), real_type, type_tag, class_tag);
   return (*this);
   }

}
/*
* DER Encoder
* (C) 1999-2007,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

/*
* DER encode an ASN.1 type tag
*/
void encode_tag(std::vector<uint8_t>& encoded_tag,
                ASN1_Tag type_tag, ASN1_Tag class_tag)
   {
   if((class_tag | 0xE0) != 0xE0)
      throw Encoding_Error("DER_Encoder: Invalid class tag " +
                           std::to_string(class_tag));

   if(type_tag <= 30)
      {
      encoded_tag.push_back(static_cast<uint8_t>(type_tag | class_tag));
      }
   else
      {
      size_t blocks = high_bit(static_cast<uint32_t>(type_tag)) + 6;
      blocks = (blocks - (blocks % 7)) / 7;

      BOTAN_ASSERT_NOMSG(blocks > 0);

      encoded_tag.push_back(static_cast<uint8_t>(class_tag | 0x1F));
      for(size_t i = 0; i != blocks - 1; ++i)
         encoded_tag.push_back(0x80 | ((type_tag >> 7*(blocks-i-1)) & 0x7F));
      encoded_tag.push_back(type_tag & 0x7F);
      }
   }

/*
* DER encode an ASN.1 length field
*/
void encode_length(std::vector<uint8_t>& encoded_length, size_t length)
   {
   if(length <= 127)
      {
      encoded_length.push_back(static_cast<uint8_t>(length));
      }
   else
      {
      const size_t bytes_needed = significant_bytes(length);

      encoded_length.push_back(static_cast<uint8_t>(0x80 | bytes_needed));

      for(size_t i = sizeof(length) - bytes_needed; i < sizeof(length); ++i)
         encoded_length.push_back(get_byte(i, length));
      }
   }

}

DER_Encoder::DER_Encoder(secure_vector<uint8_t>& vec)
   {
   m_append_output = [&vec](const uint8_t b[], size_t l)
      {
      vec.insert(vec.end(), b, b + l);
      };
   }

DER_Encoder::DER_Encoder(std::vector<uint8_t>& vec)
   {
   m_append_output = [&vec](const uint8_t b[], size_t l)
      {
      vec.insert(vec.end(), b, b + l);
      };
   }

/*
* Push the encoded SEQUENCE/SET to the encoder stream
*/
void DER_Encoder::DER_Sequence::push_contents(DER_Encoder& der)
   {
   const ASN1_Tag real_class_tag = ASN1_Tag(m_class_tag | CONSTRUCTED);

   if(m_type_tag == SET)
      {
      std::sort(m_set_contents.begin(), m_set_contents.end());
      for(size_t i = 0; i != m_set_contents.size(); ++i)
         m_contents += m_set_contents[i];
      m_set_contents.clear();
      }

   der.add_object(m_type_tag, real_class_tag, m_contents.data(), m_contents.size());
   m_contents.clear();
   }

/*
* Add an encoded value to the SEQUENCE/SET
*/
void DER_Encoder::DER_Sequence::add_bytes(const uint8_t data[], size_t length)
   {
   if(m_type_tag == SET)
      m_set_contents.push_back(secure_vector<uint8_t>(data, data + length));
   else
      m_contents += std::make_pair(data, length);
   }

void DER_Encoder::DER_Sequence::add_bytes(const uint8_t hdr[], size_t hdr_len,
                                          const uint8_t val[], size_t val_len)
   {
   if(m_type_tag == SET)
      {
      secure_vector<uint8_t> m;
      m.reserve(hdr_len + val_len);
      m += std::make_pair(hdr, hdr_len);
      m += std::make_pair(val, val_len);
      m_set_contents.push_back(std::move(m));
      }
   else
      {
      m_contents += std::make_pair(hdr, hdr_len);
      m_contents += std::make_pair(val, val_len);
      }
   }

/*
* Return the type and class taggings
*/
ASN1_Tag DER_Encoder::DER_Sequence::tag_of() const
   {
   return ASN1_Tag(m_type_tag | m_class_tag);
   }

/*
* DER_Sequence Constructor
*/
DER_Encoder::DER_Sequence::DER_Sequence(ASN1_Tag t1, ASN1_Tag t2) :
   m_type_tag(t1), m_class_tag(t2)
   {
   }

/*
* Return the encoded contents
*/
secure_vector<uint8_t> DER_Encoder::get_contents()
   {
   if(m_subsequences.size() != 0)
      throw Invalid_State("DER_Encoder: Sequence hasn't been marked done");

   if(m_append_output)
      throw Invalid_State("DER_Encoder Cannot get contents when using output vector");

   secure_vector<uint8_t> output;
   std::swap(output, m_default_outbuf);
   return output;
   }

std::vector<uint8_t> DER_Encoder::get_contents_unlocked()
   {
   if(m_subsequences.size() != 0)
      throw Invalid_State("DER_Encoder: Sequence hasn't been marked done");

   if(m_append_output)
      throw Invalid_State("DER_Encoder Cannot get contents when using output vector");

   std::vector<uint8_t> output(m_default_outbuf.begin(), m_default_outbuf.end());
   m_default_outbuf.clear();
   return output;
   }

/*
* Start a new ASN.1 SEQUENCE/SET/EXPLICIT
*/
DER_Encoder& DER_Encoder::start_cons(ASN1_Tag type_tag,
                                     ASN1_Tag class_tag)
   {
   m_subsequences.push_back(DER_Sequence(type_tag, class_tag));
   return (*this);
   }

/*
* Finish the current ASN.1 SEQUENCE/SET/EXPLICIT
*/
DER_Encoder& DER_Encoder::end_cons()
   {
   if(m_subsequences.empty())
      throw Invalid_State("DER_Encoder::end_cons: No such sequence");

   DER_Sequence last_seq = std::move(m_subsequences[m_subsequences.size()-1]);
   m_subsequences.pop_back();
   last_seq.push_contents(*this);

   return (*this);
   }

/*
* Start a new ASN.1 EXPLICIT encoding
*/
DER_Encoder& DER_Encoder::start_explicit(uint16_t type_no)
   {
   ASN1_Tag type_tag = static_cast<ASN1_Tag>(type_no);

   // This would confuse DER_Sequence
   if(type_tag == SET)
      throw Internal_Error("DER_Encoder.start_explicit(SET) not supported");

   return start_cons(type_tag, CONTEXT_SPECIFIC);
   }

/*
* Finish the current ASN.1 EXPLICIT encoding
*/
DER_Encoder& DER_Encoder::end_explicit()
   {
   return end_cons();
   }

/*
* Write raw bytes into the stream
*/
DER_Encoder& DER_Encoder::raw_bytes(const uint8_t bytes[], size_t length)
   {
   if(m_subsequences.size())
      {
      m_subsequences[m_subsequences.size()-1].add_bytes(bytes, length);
      }
   else if(m_append_output)
      {
      m_append_output(bytes, length);
      }
   else
      {
      m_default_outbuf += std::make_pair(bytes, length);
      }

   return (*this);
   }

/*
* Write the encoding of the byte(s)
*/
DER_Encoder& DER_Encoder::add_object(ASN1_Tag type_tag, ASN1_Tag class_tag,
                                     const uint8_t rep[], size_t length)
   {
   std::vector<uint8_t> hdr;
   encode_tag(hdr, type_tag, class_tag);
   encode_length(hdr, length);

   if(m_subsequences.size())
      {
      m_subsequences[m_subsequences.size()-1].add_bytes(hdr.data(), hdr.size(), rep, length);
      }
   else if(m_append_output)
      {
      m_append_output(hdr.data(), hdr.size());
      m_append_output(rep, length);
      }
   else
      {
      m_default_outbuf += hdr;
      m_default_outbuf += std::make_pair(rep, length);
      }

   return (*this);
   }

/*
* Encode a NULL object
*/
DER_Encoder& DER_Encoder::encode_null()
   {
   return add_object(NULL_TAG, UNIVERSAL, nullptr, 0);
   }

/*
* DER encode a BOOLEAN
*/
DER_Encoder& DER_Encoder::encode(bool is_true)
   {
   return encode(is_true, BOOLEAN, UNIVERSAL);
   }

/*
* DER encode a small INTEGER
*/
DER_Encoder& DER_Encoder::encode(size_t n)
   {
   return encode(BigInt(n), INTEGER, UNIVERSAL);
   }

/*
* DER encode a small INTEGER
*/
DER_Encoder& DER_Encoder::encode(const BigInt& n)
   {
   return encode(n, INTEGER, UNIVERSAL);
   }

/*
* Encode this object
*/
DER_Encoder& DER_Encoder::encode(const uint8_t bytes[], size_t length,
                                 ASN1_Tag real_type)
   {
   return encode(bytes, length, real_type, real_type, UNIVERSAL);
   }

/*
* DER encode a BOOLEAN
*/
DER_Encoder& DER_Encoder::encode(bool is_true,
                                 ASN1_Tag type_tag, ASN1_Tag class_tag)
   {
   uint8_t val = is_true ? 0xFF : 0x00;
   return add_object(type_tag, class_tag, &val, 1);
   }

/*
* DER encode a small INTEGER
*/
DER_Encoder& DER_Encoder::encode(size_t n,
                                 ASN1_Tag type_tag, ASN1_Tag class_tag)
   {
   return encode(BigInt(n), type_tag, class_tag);
   }

/*
* DER encode an INTEGER
*/
DER_Encoder& DER_Encoder::encode(const BigInt& n,
                                 ASN1_Tag type_tag, ASN1_Tag class_tag)
   {
   if(n == 0)
      return add_object(type_tag, class_tag, 0);

   const size_t extra_zero = (n.bits() % 8 == 0) ? 1 : 0;
   secure_vector<uint8_t> contents(extra_zero + n.bytes());
   n.binary_encode(&contents[extra_zero]);
   if(n < 0)
      {
      for(size_t i = 0; i != contents.size(); ++i)
         contents[i] = ~contents[i];
      for(size_t i = contents.size(); i > 0; --i)
         if(++contents[i-1])
            break;
      }

   return add_object(type_tag, class_tag, contents);
   }

/*
* DER encode an OCTET STRING or BIT STRING
*/
DER_Encoder& DER_Encoder::encode(const uint8_t bytes[], size_t length,
                                 ASN1_Tag real_type,
                                 ASN1_Tag type_tag, ASN1_Tag class_tag)
   {
   if(real_type != OCTET_STRING && real_type != BIT_STRING)
      throw Invalid_Argument("DER_Encoder: Invalid tag for byte/bit string");

   if(real_type == BIT_STRING)
      {
      secure_vector<uint8_t> encoded;
      encoded.push_back(0);
      encoded += std::make_pair(bytes, length);
      return add_object(type_tag, class_tag, encoded);
      }
   else
      return add_object(type_tag, class_tag, bytes, length);
   }

DER_Encoder& DER_Encoder::encode(const ASN1_Object& obj)
   {
   obj.encode_into(*this);
   return (*this);
   }

/*
* Write the encoding of the byte(s)
*/
DER_Encoder& DER_Encoder::add_object(ASN1_Tag type_tag, ASN1_Tag class_tag,
                                     const std::string& rep_str)
   {
   const uint8_t* rep = cast_char_ptr_to_uint8(rep_str.data());
   const size_t rep_len = rep_str.size();
   return add_object(type_tag, class_tag, rep, rep_len);
   }

/*
* Write the encoding of the byte
*/
DER_Encoder& DER_Encoder::add_object(ASN1_Tag type_tag,
                                     ASN1_Tag class_tag, uint8_t rep)
   {
   return add_object(type_tag, class_tag, &rep, 1);
   }

}
/*
* OID maps
*
* This file was automatically generated by ./src/scripts/oids.py on 2019-10-21
*
* All manual edits to this file will be lost. Edit the script
* then regenerate this source file.
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <unordered_map>

namespace Botan {

std::unordered_map<std::string, std::string> OIDS::load_oid2str_map()
   {
   return std::unordered_map<std::string,std::string>{
      { "0.3.4401.5.3.1.9.26", "Camellia-192/GCM" },
      { "0.3.4401.5.3.1.9.46", "Camellia-256/GCM" },
      { "0.3.4401.5.3.1.9.6", "Camellia-128/GCM" },
      { "0.4.0.127.0.15.1.1.13.0", "XMSS" },
      { "1.0.14888.3.0.5", "ECKCDSA" },
      { "1.2.156.10197.1.104.100", "SM4/OCB" },
      { "1.2.156.10197.1.104.2", "SM4/CBC" },
      { "1.2.156.10197.1.104.8", "SM4/GCM" },
      { "1.2.156.10197.1.301", "sm2p256v1" },
      { "1.2.156.10197.1.301.1", "SM2" },
      { "1.2.156.10197.1.301.2", "SM2_Kex" },
      { "1.2.156.10197.1.301.3", "SM2_Enc" },
      { "1.2.156.10197.1.401", "SM3" },
      { "1.2.156.10197.1.501", "SM2_Sig/SM3" },
      { "1.2.156.10197.1.504", "RSA/EMSA3(SM3)" },
      { "1.2.250.1.223.101.256.1", "frp256v1" },
      { "1.2.392.200011.61.1.1.1.2", "Camellia-128/CBC" },
      { "1.2.392.200011.61.1.1.1.3", "Camellia-192/CBC" },
      { "1.2.392.200011.61.1.1.1.4", "Camellia-256/CBC" },
      { "1.2.410.200004.1.100.4.3", "ECKCDSA/EMSA1(SHA-1)" },
      { "1.2.410.200004.1.100.4.4", "ECKCDSA/EMSA1(SHA-224)" },
      { "1.2.410.200004.1.100.4.5", "ECKCDSA/EMSA1(SHA-256)" },
      { "1.2.410.200004.1.4", "SEED/CBC" },
      { "1.2.643.100.1", "GOST.OGRN" },
      { "1.2.643.100.111", "GOST.SubjectSigningTool" },
      { "1.2.643.100.112", "GOST.IssuerSigningTool" },
      { "1.2.643.2.2.19", "GOST-34.10" },
      { "1.2.643.2.2.3", "GOST-34.10/EMSA1(GOST-R-34.11-94)" },
      { "1.2.643.2.2.35.1", "gost_256A" },
      { "1.2.643.2.2.36.0", "gost_256A" },
      { "1.2.643.3.131.1.1", "GOST.INN" },
      { "1.2.643.7.1.1.1.1", "GOST-34.10-2012-256" },
      { "1.2.643.7.1.1.1.2", "GOST-34.10-2012-512" },
      { "1.2.643.7.1.1.2.2", "Streebog-256" },
      { "1.2.643.7.1.1.2.3", "Streebog-512" },
      { "1.2.643.7.1.1.3.2", "GOST-34.10-2012-256/EMSA1(Streebog-256)" },
      { "1.2.643.7.1.1.3.3", "GOST-34.10-2012-512/EMSA1(Streebog-512)" },
      { "1.2.643.7.1.2.1.1.1", "gost_256A" },
      { "1.2.643.7.1.2.1.1.2", "gost_256B" },
      { "1.2.643.7.1.2.1.2.1", "gost_512A" },
      { "1.2.643.7.1.2.1.2.2", "gost_512B" },
      { "1.2.840.10040.4.1", "DSA" },
      { "1.2.840.10040.4.3", "DSA/EMSA1(SHA-160)" },
      { "1.2.840.10045.2.1", "ECDSA" },
      { "1.2.840.10045.3.1.1", "secp192r1" },
      { "1.2.840.10045.3.1.2", "x962_p192v2" },
      { "1.2.840.10045.3.1.3", "x962_p192v3" },
      { "1.2.840.10045.3.1.4", "x962_p239v1" },
      { "1.2.840.10045.3.1.5", "x962_p239v2" },
      { "1.2.840.10045.3.1.6", "x962_p239v3" },
      { "1.2.840.10045.3.1.7", "secp256r1" },
      { "1.2.840.10045.4.1", "ECDSA/EMSA1(SHA-160)" },
      { "1.2.840.10045.4.3.1", "ECDSA/EMSA1(SHA-224)" },
      { "1.2.840.10045.4.3.2", "ECDSA/EMSA1(SHA-256)" },
      { "1.2.840.10045.4.3.3", "ECDSA/EMSA1(SHA-384)" },
      { "1.2.840.10045.4.3.4", "ECDSA/EMSA1(SHA-512)" },
      { "1.2.840.10046.2.1", "DH" },
      { "1.2.840.113533.7.66.10", "CAST-128/CBC" },
      { "1.2.840.113533.7.66.15", "KeyWrap.CAST-128" },
      { "1.2.840.113549.1.1.1", "RSA" },
      { "1.2.840.113549.1.1.10", "RSA/EMSA4" },
      { "1.2.840.113549.1.1.11", "RSA/EMSA3(SHA-256)" },
      { "1.2.840.113549.1.1.12", "RSA/EMSA3(SHA-384)" },
      { "1.2.840.113549.1.1.13", "RSA/EMSA3(SHA-512)" },
      { "1.2.840.113549.1.1.14", "RSA/EMSA3(SHA-224)" },
      { "1.2.840.113549.1.1.16", "RSA/EMSA3(SHA-512-256)" },
      { "1.2.840.113549.1.1.4", "RSA/EMSA3(MD5)" },
      { "1.2.840.113549.1.1.5", "RSA/EMSA3(SHA-160)" },
      { "1.2.840.113549.1.1.7", "RSA/OAEP" },
      { "1.2.840.113549.1.1.8", "MGF1" },
      { "1.2.840.113549.1.5.12", "PKCS5.PBKDF2" },
      { "1.2.840.113549.1.5.13", "PBE-PKCS5v20" },
      { "1.2.840.113549.1.9.1", "PKCS9.EmailAddress" },
      { "1.2.840.113549.1.9.14", "PKCS9.ExtensionRequest" },
      { "1.2.840.113549.1.9.16.3.18", "ChaCha20Poly1305" },
      { "1.2.840.113549.1.9.16.3.6", "KeyWrap.TripleDES" },
      { "1.2.840.113549.1.9.16.3.8", "Compression.Zlib" },
      { "1.2.840.113549.1.9.2", "PKCS9.UnstructuredName" },
      { "1.2.840.113549.1.9.3", "PKCS9.ContentType" },
      { "1.2.840.113549.1.9.4", "PKCS9.MessageDigest" },
      { "1.2.840.113549.1.9.7", "PKCS9.ChallengePassword" },
      { "1.2.840.113549.2.10", "HMAC(SHA-384)" },
      { "1.2.840.113549.2.11", "HMAC(SHA-512)" },
      { "1.2.840.113549.2.13", "HMAC(SHA-512-256)" },
      { "1.2.840.113549.2.5", "MD5" },
      { "1.2.840.113549.2.7", "HMAC(SHA-160)" },
      { "1.2.840.113549.2.8", "HMAC(SHA-224)" },
      { "1.2.840.113549.2.9", "HMAC(SHA-256)" },
      { "1.2.840.113549.3.7", "TripleDES/CBC" },
      { "1.3.101.110", "Curve25519" },
      { "1.3.101.112", "Ed25519" },
      { "1.3.132.0.10", "secp256k1" },
      { "1.3.132.0.30", "secp160r2" },
      { "1.3.132.0.31", "secp192k1" },
      { "1.3.132.0.32", "secp224k1" },
      { "1.3.132.0.33", "secp224r1" },
      { "1.3.132.0.34", "secp384r1" },
      { "1.3.132.0.35", "secp521r1" },
      { "1.3.132.0.8", "secp160r1" },
      { "1.3.132.0.9", "secp160k1" },
      { "1.3.132.1.12", "ECDH" },
      { "1.3.14.3.2.26", "SHA-160" },
      { "1.3.14.3.2.7", "DES/CBC" },
      { "1.3.36.3.2.1", "RIPEMD-160" },
      { "1.3.36.3.3.1.2", "RSA/EMSA3(RIPEMD-160)" },
      { "1.3.36.3.3.2.5.2.1", "ECGDSA" },
      { "1.3.36.3.3.2.5.4.1", "ECGDSA/EMSA1(RIPEMD-160)" },
      { "1.3.36.3.3.2.5.4.2", "ECGDSA/EMSA1(SHA-160)" },
      { "1.3.36.3.3.2.5.4.3", "ECGDSA/EMSA1(SHA-224)" },
      { "1.3.36.3.3.2.5.4.4", "ECGDSA/EMSA1(SHA-256)" },
      { "1.3.36.3.3.2.5.4.5", "ECGDSA/EMSA1(SHA-384)" },
      { "1.3.36.3.3.2.5.4.6", "ECGDSA/EMSA1(SHA-512)" },
      { "1.3.36.3.3.2.8.1.1.1", "brainpool160r1" },
      { "1.3.36.3.3.2.8.1.1.11", "brainpool384r1" },
      { "1.3.36.3.3.2.8.1.1.13", "brainpool512r1" },
      { "1.3.36.3.3.2.8.1.1.3", "brainpool192r1" },
      { "1.3.36.3.3.2.8.1.1.5", "brainpool224r1" },
      { "1.3.36.3.3.2.8.1.1.7", "brainpool256r1" },
      { "1.3.36.3.3.2.8.1.1.9", "brainpool320r1" },
      { "1.3.6.1.4.1.11591.12.2", "Tiger(24,3)" },
      { "1.3.6.1.4.1.11591.15.1", "OpenPGP.Ed25519" },
      { "1.3.6.1.4.1.11591.4.11", "Scrypt" },
      { "1.3.6.1.4.1.25258.1.3", "McEliece" },
      { "1.3.6.1.4.1.25258.1.5", "XMSS-draft6" },
      { "1.3.6.1.4.1.25258.1.6.1", "GOST-34.10-2012-256/EMSA1(SHA-256)" },
      { "1.3.6.1.4.1.25258.1.8", "XMSS-draft12" },
      { "1.3.6.1.4.1.25258.3.1", "Serpent/CBC" },
      { "1.3.6.1.4.1.25258.3.101", "Serpent/GCM" },
      { "1.3.6.1.4.1.25258.3.102", "Twofish/GCM" },
      { "1.3.6.1.4.1.25258.3.2", "Threefish-512/CBC" },
      { "1.3.6.1.4.1.25258.3.2.1", "AES-128/OCB" },
      { "1.3.6.1.4.1.25258.3.2.2", "AES-192/OCB" },
      { "1.3.6.1.4.1.25258.3.2.3", "AES-256/OCB" },
      { "1.3.6.1.4.1.25258.3.2.4", "Serpent/OCB" },
      { "1.3.6.1.4.1.25258.3.2.5", "Twofish/OCB" },
      { "1.3.6.1.4.1.25258.3.2.6", "Camellia-128/OCB" },
      { "1.3.6.1.4.1.25258.3.2.7", "Camellia-192/OCB" },
      { "1.3.6.1.4.1.25258.3.2.8", "Camellia-256/OCB" },
      { "1.3.6.1.4.1.25258.3.3", "Twofish/CBC" },
      { "1.3.6.1.4.1.25258.3.4.1", "AES-128/SIV" },
      { "1.3.6.1.4.1.25258.3.4.2", "AES-192/SIV" },
      { "1.3.6.1.4.1.25258.3.4.3", "AES-256/SIV" },
      { "1.3.6.1.4.1.25258.3.4.4", "Serpent/SIV" },
      { "1.3.6.1.4.1.25258.3.4.5", "Twofish/SIV" },
      { "1.3.6.1.4.1.25258.3.4.6", "Camellia-128/SIV" },
      { "1.3.6.1.4.1.25258.3.4.7", "Camellia-192/SIV" },
      { "1.3.6.1.4.1.25258.3.4.8", "Camellia-256/SIV" },
      { "1.3.6.1.4.1.25258.3.4.9", "SM4/SIV" },
      { "1.3.6.1.4.1.3029.1.2.1", "ElGamal" },
      { "1.3.6.1.4.1.3029.1.5.1", "OpenPGP.Curve25519" },
      { "1.3.6.1.4.1.311.20.2.2", "Microsoft SmartcardLogon" },
      { "1.3.6.1.4.1.311.20.2.3", "Microsoft UPN" },
      { "1.3.6.1.4.1.8301.3.1.2.9.0.38", "secp521r1" },
      { "1.3.6.1.5.5.7.1.1", "PKIX.AuthorityInformationAccess" },
      { "1.3.6.1.5.5.7.3.1", "PKIX.ServerAuth" },
      { "1.3.6.1.5.5.7.3.2", "PKIX.ClientAuth" },
      { "1.3.6.1.5.5.7.3.3", "PKIX.CodeSigning" },
      { "1.3.6.1.5.5.7.3.4", "PKIX.EmailProtection" },
      { "1.3.6.1.5.5.7.3.5", "PKIX.IPsecEndSystem" },
      { "1.3.6.1.5.5.7.3.6", "PKIX.IPsecTunnel" },
      { "1.3.6.1.5.5.7.3.7", "PKIX.IPsecUser" },
      { "1.3.6.1.5.5.7.3.8", "PKIX.TimeStamping" },
      { "1.3.6.1.5.5.7.3.9", "PKIX.OCSPSigning" },
      { "1.3.6.1.5.5.7.48.1", "PKIX.OCSP" },
      { "1.3.6.1.5.5.7.48.1.1", "PKIX.OCSP.BasicResponse" },
      { "1.3.6.1.5.5.7.48.2", "PKIX.CertificateAuthorityIssuers" },
      { "1.3.6.1.5.5.7.8.5", "PKIX.XMPPAddr" },
      { "2.16.840.1.101.3.4.1.2", "AES-128/CBC" },
      { "2.16.840.1.101.3.4.1.22", "AES-192/CBC" },
      { "2.16.840.1.101.3.4.1.25", "KeyWrap.AES-192" },
      { "2.16.840.1.101.3.4.1.26", "AES-192/GCM" },
      { "2.16.840.1.101.3.4.1.27", "AES-192/CCM" },
      { "2.16.840.1.101.3.4.1.42", "AES-256/CBC" },
      { "2.16.840.1.101.3.4.1.45", "KeyWrap.AES-256" },
      { "2.16.840.1.101.3.4.1.46", "AES-256/GCM" },
      { "2.16.840.1.101.3.4.1.47", "AES-256/CCM" },
      { "2.16.840.1.101.3.4.1.5", "KeyWrap.AES-128" },
      { "2.16.840.1.101.3.4.1.6", "AES-128/GCM" },
      { "2.16.840.1.101.3.4.1.7", "AES-128/CCM" },
      { "2.16.840.1.101.3.4.2.1", "SHA-256" },
      { "2.16.840.1.101.3.4.2.10", "SHA-3(512)" },
      { "2.16.840.1.101.3.4.2.11", "SHAKE-128" },
      { "2.16.840.1.101.3.4.2.12", "SHAKE-256" },
      { "2.16.840.1.101.3.4.2.2", "SHA-384" },
      { "2.16.840.1.101.3.4.2.3", "SHA-512" },
      { "2.16.840.1.101.3.4.2.4", "SHA-224" },
      { "2.16.840.1.101.3.4.2.6", "SHA-512-256" },
      { "2.16.840.1.101.3.4.2.7", "SHA-3(224)" },
      { "2.16.840.1.101.3.4.2.8", "SHA-3(256)" },
      { "2.16.840.1.101.3.4.2.9", "SHA-3(384)" },
      { "2.16.840.1.101.3.4.3.1", "DSA/EMSA1(SHA-224)" },
      { "2.16.840.1.101.3.4.3.10", "ECDSA/EMSA1(SHA-3(256))" },
      { "2.16.840.1.101.3.4.3.11", "ECDSA/EMSA1(SHA-3(384))" },
      { "2.16.840.1.101.3.4.3.12", "ECDSA/EMSA1(SHA-3(512))" },
      { "2.16.840.1.101.3.4.3.13", "RSA/EMSA3(SHA-3(224))" },
      { "2.16.840.1.101.3.4.3.14", "RSA/EMSA3(SHA-3(256))" },
      { "2.16.840.1.101.3.4.3.15", "RSA/EMSA3(SHA-3(384))" },
      { "2.16.840.1.101.3.4.3.16", "RSA/EMSA3(SHA-3(512))" },
      { "2.16.840.1.101.3.4.3.2", "DSA/EMSA1(SHA-256)" },
      { "2.16.840.1.101.3.4.3.3", "DSA/EMSA1(SHA-384)" },
      { "2.16.840.1.101.3.4.3.4", "DSA/EMSA1(SHA-512)" },
      { "2.16.840.1.101.3.4.3.5", "DSA/EMSA1(SHA-3(224))" },
      { "2.16.840.1.101.3.4.3.6", "DSA/EMSA1(SHA-3(256))" },
      { "2.16.840.1.101.3.4.3.7", "DSA/EMSA1(SHA-3(384))" },
      { "2.16.840.1.101.3.4.3.8", "DSA/EMSA1(SHA-3(512))" },
      { "2.16.840.1.101.3.4.3.9", "ECDSA/EMSA1(SHA-3(224))" },
      { "2.16.840.1.113730.1.13", "Certificate Comment" },
      { "2.5.29.14", "X509v3.SubjectKeyIdentifier" },
      { "2.5.29.15", "X509v3.KeyUsage" },
      { "2.5.29.16", "X509v3.PrivateKeyUsagePeriod" },
      { "2.5.29.17", "X509v3.SubjectAlternativeName" },
      { "2.5.29.18", "X509v3.IssuerAlternativeName" },
      { "2.5.29.19", "X509v3.BasicConstraints" },
      { "2.5.29.20", "X509v3.CRLNumber" },
      { "2.5.29.21", "X509v3.ReasonCode" },
      { "2.5.29.23", "X509v3.HoldInstructionCode" },
      { "2.5.29.24", "X509v3.InvalidityDate" },
      { "2.5.29.28", "X509v3.CRLIssuingDistributionPoint" },
      { "2.5.29.30", "X509v3.NameConstraints" },
      { "2.5.29.31", "X509v3.CRLDistributionPoints" },
      { "2.5.29.32", "X509v3.CertificatePolicies" },
      { "2.5.29.32.0", "X509v3.AnyPolicy" },
      { "2.5.29.35", "X509v3.AuthorityKeyIdentifier" },
      { "2.5.29.36", "X509v3.PolicyConstraints" },
      { "2.5.29.37", "X509v3.ExtendedKeyUsage" },
      { "2.5.4.10", "X520.Organization" },
      { "2.5.4.11", "X520.OrganizationalUnit" },
      { "2.5.4.12", "X520.Title" },
      { "2.5.4.3", "X520.CommonName" },
      { "2.5.4.4", "X520.Surname" },
      { "2.5.4.42", "X520.GivenName" },
      { "2.5.4.43", "X520.Initials" },
      { "2.5.4.44", "X520.GenerationalQualifier" },
      { "2.5.4.46", "X520.DNQualifier" },
      { "2.5.4.5", "X520.SerialNumber" },
      { "2.5.4.6", "X520.Country" },
      { "2.5.4.65", "X520.Pseudonym" },
      { "2.5.4.7", "X520.Locality" },
      { "2.5.4.8", "X520.State" },
      { "2.5.4.9", "X520.StreetAddress" },
      { "2.5.8.1.1", "RSA" }
      };
   }

std::unordered_map<std::string, OID> OIDS::load_str2oid_map()
   {
   return std::unordered_map<std::string,OID>{
      { "AES-128/CBC", OID({2,16,840,1,101,3,4,1,2}) },
      { "AES-128/CCM", OID({2,16,840,1,101,3,4,1,7}) },
      { "AES-128/GCM", OID({2,16,840,1,101,3,4,1,6}) },
      { "AES-128/OCB", OID({1,3,6,1,4,1,25258,3,2,1}) },
      { "AES-128/SIV", OID({1,3,6,1,4,1,25258,3,4,1}) },
      { "AES-192/CBC", OID({2,16,840,1,101,3,4,1,22}) },
      { "AES-192/CCM", OID({2,16,840,1,101,3,4,1,27}) },
      { "AES-192/GCM", OID({2,16,840,1,101,3,4,1,26}) },
      { "AES-192/OCB", OID({1,3,6,1,4,1,25258,3,2,2}) },
      { "AES-192/SIV", OID({1,3,6,1,4,1,25258,3,4,2}) },
      { "AES-256/CBC", OID({2,16,840,1,101,3,4,1,42}) },
      { "AES-256/CCM", OID({2,16,840,1,101,3,4,1,47}) },
      { "AES-256/GCM", OID({2,16,840,1,101,3,4,1,46}) },
      { "AES-256/OCB", OID({1,3,6,1,4,1,25258,3,2,3}) },
      { "AES-256/SIV", OID({1,3,6,1,4,1,25258,3,4,3}) },
      { "CAST-128/CBC", OID({1,2,840,113533,7,66,10}) },
      { "Camellia-128/CBC", OID({1,2,392,200011,61,1,1,1,2}) },
      { "Camellia-128/GCM", OID({0,3,4401,5,3,1,9,6}) },
      { "Camellia-128/OCB", OID({1,3,6,1,4,1,25258,3,2,6}) },
      { "Camellia-128/SIV", OID({1,3,6,1,4,1,25258,3,4,6}) },
      { "Camellia-192/CBC", OID({1,2,392,200011,61,1,1,1,3}) },
      { "Camellia-192/GCM", OID({0,3,4401,5,3,1,9,26}) },
      { "Camellia-192/OCB", OID({1,3,6,1,4,1,25258,3,2,7}) },
      { "Camellia-192/SIV", OID({1,3,6,1,4,1,25258,3,4,7}) },
      { "Camellia-256/CBC", OID({1,2,392,200011,61,1,1,1,4}) },
      { "Camellia-256/GCM", OID({0,3,4401,5,3,1,9,46}) },
      { "Camellia-256/OCB", OID({1,3,6,1,4,1,25258,3,2,8}) },
      { "Camellia-256/SIV", OID({1,3,6,1,4,1,25258,3,4,8}) },
      { "Certificate Comment", OID({2,16,840,1,113730,1,13}) },
      { "ChaCha20Poly1305", OID({1,2,840,113549,1,9,16,3,18}) },
      { "Compression.Zlib", OID({1,2,840,113549,1,9,16,3,8}) },
      { "Curve25519", OID({1,3,101,110}) },
      { "DES/CBC", OID({1,3,14,3,2,7}) },
      { "DH", OID({1,2,840,10046,2,1}) },
      { "DSA", OID({1,2,840,10040,4,1}) },
      { "DSA/EMSA1(SHA-160)", OID({1,2,840,10040,4,3}) },
      { "DSA/EMSA1(SHA-224)", OID({2,16,840,1,101,3,4,3,1}) },
      { "DSA/EMSA1(SHA-256)", OID({2,16,840,1,101,3,4,3,2}) },
      { "DSA/EMSA1(SHA-3(224))", OID({2,16,840,1,101,3,4,3,5}) },
      { "DSA/EMSA1(SHA-3(256))", OID({2,16,840,1,101,3,4,3,6}) },
      { "DSA/EMSA1(SHA-3(384))", OID({2,16,840,1,101,3,4,3,7}) },
      { "DSA/EMSA1(SHA-3(512))", OID({2,16,840,1,101,3,4,3,8}) },
      { "DSA/EMSA1(SHA-384)", OID({2,16,840,1,101,3,4,3,3}) },
      { "DSA/EMSA1(SHA-512)", OID({2,16,840,1,101,3,4,3,4}) },
      { "ECDH", OID({1,3,132,1,12}) },
      { "ECDSA", OID({1,2,840,10045,2,1}) },
      { "ECDSA/EMSA1(SHA-160)", OID({1,2,840,10045,4,1}) },
      { "ECDSA/EMSA1(SHA-224)", OID({1,2,840,10045,4,3,1}) },
      { "ECDSA/EMSA1(SHA-256)", OID({1,2,840,10045,4,3,2}) },
      { "ECDSA/EMSA1(SHA-3(224))", OID({2,16,840,1,101,3,4,3,9}) },
      { "ECDSA/EMSA1(SHA-3(256))", OID({2,16,840,1,101,3,4,3,10}) },
      { "ECDSA/EMSA1(SHA-3(384))", OID({2,16,840,1,101,3,4,3,11}) },
      { "ECDSA/EMSA1(SHA-3(512))", OID({2,16,840,1,101,3,4,3,12}) },
      { "ECDSA/EMSA1(SHA-384)", OID({1,2,840,10045,4,3,3}) },
      { "ECDSA/EMSA1(SHA-512)", OID({1,2,840,10045,4,3,4}) },
      { "ECGDSA", OID({1,3,36,3,3,2,5,2,1}) },
      { "ECGDSA/EMSA1(RIPEMD-160)", OID({1,3,36,3,3,2,5,4,1}) },
      { "ECGDSA/EMSA1(SHA-160)", OID({1,3,36,3,3,2,5,4,2}) },
      { "ECGDSA/EMSA1(SHA-224)", OID({1,3,36,3,3,2,5,4,3}) },
      { "ECGDSA/EMSA1(SHA-256)", OID({1,3,36,3,3,2,5,4,4}) },
      { "ECGDSA/EMSA1(SHA-384)", OID({1,3,36,3,3,2,5,4,5}) },
      { "ECGDSA/EMSA1(SHA-512)", OID({1,3,36,3,3,2,5,4,6}) },
      { "ECKCDSA", OID({1,0,14888,3,0,5}) },
      { "ECKCDSA/EMSA1(SHA-1)", OID({1,2,410,200004,1,100,4,3}) },
      { "ECKCDSA/EMSA1(SHA-224)", OID({1,2,410,200004,1,100,4,4}) },
      { "ECKCDSA/EMSA1(SHA-256)", OID({1,2,410,200004,1,100,4,5}) },
      { "Ed25519", OID({1,3,101,112}) },
      { "ElGamal", OID({1,3,6,1,4,1,3029,1,2,1}) },
      { "GOST-34.10", OID({1,2,643,2,2,19}) },
      { "GOST-34.10-2012-256", OID({1,2,643,7,1,1,1,1}) },
      { "GOST-34.10-2012-256/EMSA1(SHA-256)", OID({1,3,6,1,4,1,25258,1,6,1}) },
      { "GOST-34.10-2012-256/EMSA1(Streebog-256)", OID({1,2,643,7,1,1,3,2}) },
      { "GOST-34.10-2012-512", OID({1,2,643,7,1,1,1,2}) },
      { "GOST-34.10-2012-512/EMSA1(Streebog-512)", OID({1,2,643,7,1,1,3,3}) },
      { "GOST-34.10/EMSA1(GOST-R-34.11-94)", OID({1,2,643,2,2,3}) },
      { "GOST.INN", OID({1,2,643,3,131,1,1}) },
      { "GOST.IssuerSigningTool", OID({1,2,643,100,112}) },
      { "GOST.OGRN", OID({1,2,643,100,1}) },
      { "GOST.SubjectSigningTool", OID({1,2,643,100,111}) },
      { "HMAC(SHA-160)", OID({1,2,840,113549,2,7}) },
      { "HMAC(SHA-224)", OID({1,2,840,113549,2,8}) },
      { "HMAC(SHA-256)", OID({1,2,840,113549,2,9}) },
      { "HMAC(SHA-384)", OID({1,2,840,113549,2,10}) },
      { "HMAC(SHA-512)", OID({1,2,840,113549,2,11}) },
      { "HMAC(SHA-512-256)", OID({1,2,840,113549,2,13}) },
      { "KeyWrap.AES-128", OID({2,16,840,1,101,3,4,1,5}) },
      { "KeyWrap.AES-192", OID({2,16,840,1,101,3,4,1,25}) },
      { "KeyWrap.AES-256", OID({2,16,840,1,101,3,4,1,45}) },
      { "KeyWrap.CAST-128", OID({1,2,840,113533,7,66,15}) },
      { "KeyWrap.TripleDES", OID({1,2,840,113549,1,9,16,3,6}) },
      { "MD5", OID({1,2,840,113549,2,5}) },
      { "MGF1", OID({1,2,840,113549,1,1,8}) },
      { "McEliece", OID({1,3,6,1,4,1,25258,1,3}) },
      { "Microsoft SmartcardLogon", OID({1,3,6,1,4,1,311,20,2,2}) },
      { "Microsoft UPN", OID({1,3,6,1,4,1,311,20,2,3}) },
      { "OpenPGP.Curve25519", OID({1,3,6,1,4,1,3029,1,5,1}) },
      { "OpenPGP.Ed25519", OID({1,3,6,1,4,1,11591,15,1}) },
      { "PBE-PKCS5v20", OID({1,2,840,113549,1,5,13}) },
      { "PBES2", OID({1,2,840,113549,1,5,13}) },
      { "PKCS5.PBKDF2", OID({1,2,840,113549,1,5,12}) },
      { "PKCS9.ChallengePassword", OID({1,2,840,113549,1,9,7}) },
      { "PKCS9.ContentType", OID({1,2,840,113549,1,9,3}) },
      { "PKCS9.EmailAddress", OID({1,2,840,113549,1,9,1}) },
      { "PKCS9.ExtensionRequest", OID({1,2,840,113549,1,9,14}) },
      { "PKCS9.MessageDigest", OID({1,2,840,113549,1,9,4}) },
      { "PKCS9.UnstructuredName", OID({1,2,840,113549,1,9,2}) },
      { "PKIX.AuthorityInformationAccess", OID({1,3,6,1,5,5,7,1,1}) },
      { "PKIX.CertificateAuthorityIssuers", OID({1,3,6,1,5,5,7,48,2}) },
      { "PKIX.ClientAuth", OID({1,3,6,1,5,5,7,3,2}) },
      { "PKIX.CodeSigning", OID({1,3,6,1,5,5,7,3,3}) },
      { "PKIX.EmailProtection", OID({1,3,6,1,5,5,7,3,4}) },
      { "PKIX.IPsecEndSystem", OID({1,3,6,1,5,5,7,3,5}) },
      { "PKIX.IPsecTunnel", OID({1,3,6,1,5,5,7,3,6}) },
      { "PKIX.IPsecUser", OID({1,3,6,1,5,5,7,3,7}) },
      { "PKIX.OCSP", OID({1,3,6,1,5,5,7,48,1}) },
      { "PKIX.OCSP.BasicResponse", OID({1,3,6,1,5,5,7,48,1,1}) },
      { "PKIX.OCSPSigning", OID({1,3,6,1,5,5,7,3,9}) },
      { "PKIX.ServerAuth", OID({1,3,6,1,5,5,7,3,1}) },
      { "PKIX.TimeStamping", OID({1,3,6,1,5,5,7,3,8}) },
      { "PKIX.XMPPAddr", OID({1,3,6,1,5,5,7,8,5}) },
      { "RIPEMD-160", OID({1,3,36,3,2,1}) },
      { "RSA", OID({1,2,840,113549,1,1,1}) },
      { "RSA/EMSA3(MD5)", OID({1,2,840,113549,1,1,4}) },
      { "RSA/EMSA3(RIPEMD-160)", OID({1,3,36,3,3,1,2}) },
      { "RSA/EMSA3(SHA-160)", OID({1,2,840,113549,1,1,5}) },
      { "RSA/EMSA3(SHA-224)", OID({1,2,840,113549,1,1,14}) },
      { "RSA/EMSA3(SHA-256)", OID({1,2,840,113549,1,1,11}) },
      { "RSA/EMSA3(SHA-3(224))", OID({2,16,840,1,101,3,4,3,13}) },
      { "RSA/EMSA3(SHA-3(256))", OID({2,16,840,1,101,3,4,3,14}) },
      { "RSA/EMSA3(SHA-3(384))", OID({2,16,840,1,101,3,4,3,15}) },
      { "RSA/EMSA3(SHA-3(512))", OID({2,16,840,1,101,3,4,3,16}) },
      { "RSA/EMSA3(SHA-384)", OID({1,2,840,113549,1,1,12}) },
      { "RSA/EMSA3(SHA-512)", OID({1,2,840,113549,1,1,13}) },
      { "RSA/EMSA3(SHA-512-256)", OID({1,2,840,113549,1,1,16}) },
      { "RSA/EMSA3(SM3)", OID({1,2,156,10197,1,504}) },
      { "RSA/EMSA4", OID({1,2,840,113549,1,1,10}) },
      { "RSA/OAEP", OID({1,2,840,113549,1,1,7}) },
      { "SEED/CBC", OID({1,2,410,200004,1,4}) },
      { "SHA-160", OID({1,3,14,3,2,26}) },
      { "SHA-224", OID({2,16,840,1,101,3,4,2,4}) },
      { "SHA-256", OID({2,16,840,1,101,3,4,2,1}) },
      { "SHA-3(224)", OID({2,16,840,1,101,3,4,2,7}) },
      { "SHA-3(256)", OID({2,16,840,1,101,3,4,2,8}) },
      { "SHA-3(384)", OID({2,16,840,1,101,3,4,2,9}) },
      { "SHA-3(512)", OID({2,16,840,1,101,3,4,2,10}) },
      { "SHA-384", OID({2,16,840,1,101,3,4,2,2}) },
      { "SHA-512", OID({2,16,840,1,101,3,4,2,3}) },
      { "SHA-512-256", OID({2,16,840,1,101,3,4,2,6}) },
      { "SHAKE-128", OID({2,16,840,1,101,3,4,2,11}) },
      { "SHAKE-256", OID({2,16,840,1,101,3,4,2,12}) },
      { "SM2", OID({1,2,156,10197,1,301,1}) },
      { "SM2_Enc", OID({1,2,156,10197,1,301,3}) },
      { "SM2_Kex", OID({1,2,156,10197,1,301,2}) },
      { "SM2_Sig", OID({1,2,156,10197,1,301,1}) },
      { "SM2_Sig/SM3", OID({1,2,156,10197,1,501}) },
      { "SM3", OID({1,2,156,10197,1,401}) },
      { "SM4/CBC", OID({1,2,156,10197,1,104,2}) },
      { "SM4/GCM", OID({1,2,156,10197,1,104,8}) },
      { "SM4/OCB", OID({1,2,156,10197,1,104,100}) },
      { "SM4/SIV", OID({1,3,6,1,4,1,25258,3,4,9}) },
      { "Scrypt", OID({1,3,6,1,4,1,11591,4,11}) },
      { "Serpent/CBC", OID({1,3,6,1,4,1,25258,3,1}) },
      { "Serpent/GCM", OID({1,3,6,1,4,1,25258,3,101}) },
      { "Serpent/OCB", OID({1,3,6,1,4,1,25258,3,2,4}) },
      { "Serpent/SIV", OID({1,3,6,1,4,1,25258,3,4,4}) },
      { "Streebog-256", OID({1,2,643,7,1,1,2,2}) },
      { "Streebog-512", OID({1,2,643,7,1,1,2,3}) },
      { "Threefish-512/CBC", OID({1,3,6,1,4,1,25258,3,2}) },
      { "Tiger(24,3)", OID({1,3,6,1,4,1,11591,12,2}) },
      { "TripleDES/CBC", OID({1,2,840,113549,3,7}) },
      { "Twofish/CBC", OID({1,3,6,1,4,1,25258,3,3}) },
      { "Twofish/GCM", OID({1,3,6,1,4,1,25258,3,102}) },
      { "Twofish/OCB", OID({1,3,6,1,4,1,25258,3,2,5}) },
      { "Twofish/SIV", OID({1,3,6,1,4,1,25258,3,4,5}) },
      { "X509v3.AnyPolicy", OID({2,5,29,32,0}) },
      { "X509v3.AuthorityKeyIdentifier", OID({2,5,29,35}) },
      { "X509v3.BasicConstraints", OID({2,5,29,19}) },
      { "X509v3.CRLDistributionPoints", OID({2,5,29,31}) },
      { "X509v3.CRLIssuingDistributionPoint", OID({2,5,29,28}) },
      { "X509v3.CRLNumber", OID({2,5,29,20}) },
      { "X509v3.CertificatePolicies", OID({2,5,29,32}) },
      { "X509v3.ExtendedKeyUsage", OID({2,5,29,37}) },
      { "X509v3.HoldInstructionCode", OID({2,5,29,23}) },
      { "X509v3.InvalidityDate", OID({2,5,29,24}) },
      { "X509v3.IssuerAlternativeName", OID({2,5,29,18}) },
      { "X509v3.KeyUsage", OID({2,5,29,15}) },
      { "X509v3.NameConstraints", OID({2,5,29,30}) },
      { "X509v3.PolicyConstraints", OID({2,5,29,36}) },
      { "X509v3.PrivateKeyUsagePeriod", OID({2,5,29,16}) },
      { "X509v3.ReasonCode", OID({2,5,29,21}) },
      { "X509v3.SubjectAlternativeName", OID({2,5,29,17}) },
      { "X509v3.SubjectKeyIdentifier", OID({2,5,29,14}) },
      { "X520.CommonName", OID({2,5,4,3}) },
      { "X520.Country", OID({2,5,4,6}) },
      { "X520.DNQualifier", OID({2,5,4,46}) },
      { "X520.GenerationalQualifier", OID({2,5,4,44}) },
      { "X520.GivenName", OID({2,5,4,42}) },
      { "X520.Initials", OID({2,5,4,43}) },
      { "X520.Locality", OID({2,5,4,7}) },
      { "X520.Organization", OID({2,5,4,10}) },
      { "X520.OrganizationalUnit", OID({2,5,4,11}) },
      { "X520.Pseudonym", OID({2,5,4,65}) },
      { "X520.SerialNumber", OID({2,5,4,5}) },
      { "X520.State", OID({2,5,4,8}) },
      { "X520.StreetAddress", OID({2,5,4,9}) },
      { "X520.Surname", OID({2,5,4,4}) },
      { "X520.Title", OID({2,5,4,12}) },
      { "XMSS", OID({0,4,0,127,0,15,1,1,13,0}) },
      { "XMSS-draft12", OID({1,3,6,1,4,1,25258,1,8}) },
      { "XMSS-draft6", OID({1,3,6,1,4,1,25258,1,5}) },
      { "brainpool160r1", OID({1,3,36,3,3,2,8,1,1,1}) },
      { "brainpool192r1", OID({1,3,36,3,3,2,8,1,1,3}) },
      { "brainpool224r1", OID({1,3,36,3,3,2,8,1,1,5}) },
      { "brainpool256r1", OID({1,3,36,3,3,2,8,1,1,7}) },
      { "brainpool320r1", OID({1,3,36,3,3,2,8,1,1,9}) },
      { "brainpool384r1", OID({1,3,36,3,3,2,8,1,1,11}) },
      { "brainpool512r1", OID({1,3,36,3,3,2,8,1,1,13}) },
      { "frp256v1", OID({1,2,250,1,223,101,256,1}) },
      { "gost_256A", OID({1,2,643,7,1,2,1,1,1}) },
      { "gost_256B", OID({1,2,643,7,1,2,1,1,2}) },
      { "gost_512A", OID({1,2,643,7,1,2,1,2,1}) },
      { "gost_512B", OID({1,2,643,7,1,2,1,2,2}) },
      { "secp160k1", OID({1,3,132,0,9}) },
      { "secp160r1", OID({1,3,132,0,8}) },
      { "secp160r2", OID({1,3,132,0,30}) },
      { "secp192k1", OID({1,3,132,0,31}) },
      { "secp192r1", OID({1,2,840,10045,3,1,1}) },
      { "secp224k1", OID({1,3,132,0,32}) },
      { "secp224r1", OID({1,3,132,0,33}) },
      { "secp256k1", OID({1,3,132,0,10}) },
      { "secp256r1", OID({1,2,840,10045,3,1,7}) },
      { "secp384r1", OID({1,3,132,0,34}) },
      { "secp521r1", OID({1,3,132,0,35}) },
      { "sm2p256v1", OID({1,2,156,10197,1,301}) },
      { "x962_p192v2", OID({1,2,840,10045,3,1,2}) },
      { "x962_p192v3", OID({1,2,840,10045,3,1,3}) },
      { "x962_p239v1", OID({1,2,840,10045,3,1,4}) },
      { "x962_p239v2", OID({1,2,840,10045,3,1,5}) },
      { "x962_p239v3", OID({1,2,840,10045,3,1,6}) }
      };
   }

}

/*
* OID Registry
* (C) 1999-2008,2013 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

class OID_Map final
   {
   public:
      void add_oid(const OID& oid, const std::string& str)
         {
         add_str2oid(oid, str);
         add_oid2str(oid, str);
         }

      void add_str2oid(const OID& oid, const std::string& str)
         {
         lock_guard_type<mutex_type> lock(m_mutex);
         auto i = m_str2oid.find(str);
         if(i == m_str2oid.end())
            m_str2oid.insert(std::make_pair(str, oid));
         }

      void add_oid2str(const OID& oid, const std::string& str)
         {
         const std::string oid_str = oid.to_string();
         lock_guard_type<mutex_type> lock(m_mutex);
         auto i = m_oid2str.find(oid_str);
         if(i == m_oid2str.end())
            m_oid2str.insert(std::make_pair(oid_str, str));
         }

      std::string oid2str(const OID& oid)
         {
         const std::string oid_str = oid.to_string();

         lock_guard_type<mutex_type> lock(m_mutex);

         auto i = m_oid2str.find(oid_str);
         if(i != m_oid2str.end())
            return i->second;

         return "";
         }

      OID str2oid(const std::string& str)
         {
         lock_guard_type<mutex_type> lock(m_mutex);
         auto i = m_str2oid.find(str);
         if(i != m_str2oid.end())
            return i->second;

         return OID();
         }

      bool have_oid(const std::string& str)
         {
         lock_guard_type<mutex_type> lock(m_mutex);
         return m_str2oid.find(str) != m_str2oid.end();
         }

      static OID_Map& global_registry()
         {
         static OID_Map g_map;
         return g_map;
         }

   private:

      OID_Map()
         {
         m_str2oid = OIDS::load_str2oid_map();
         m_oid2str = OIDS::load_oid2str_map();
         }

      mutex_type m_mutex;
      std::unordered_map<std::string, OID> m_str2oid;
      std::unordered_map<std::string, std::string> m_oid2str;
   };

}

void OIDS::add_oid(const OID& oid, const std::string& name)
   {
   OID_Map::global_registry().add_oid(oid, name);
   }

void OIDS::add_oidstr(const char* oidstr, const char* name)
   {
   add_oid(OID(oidstr), name);
   }

void OIDS::add_oid2str(const OID& oid, const std::string& name)
   {
   OID_Map::global_registry().add_oid2str(oid, name);
   }

void OIDS::add_str2oid(const OID& oid, const std::string& name)
   {
   OID_Map::global_registry().add_str2oid(oid, name);
   }

std::string OIDS::oid2str_or_empty(const OID& oid)
   {
   return OID_Map::global_registry().oid2str(oid);
   }

OID OIDS::str2oid_or_empty(const std::string& name)
   {
   return OID_Map::global_registry().str2oid(name);
   }

std::string OIDS::oid2str_or_throw(const OID& oid)
   {
   const std::string s = OIDS::oid2str_or_empty(oid);
   if(s.empty())
      throw Lookup_Error("No name associated with OID " + oid.to_string());
   return s;
   }

bool OIDS::have_oid(const std::string& name)
   {
   return OID_Map::global_registry().have_oid(name);
   }

}
/*
* (C) 2019 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

void Buffered_Computation::update_be(uint16_t val)
   {
   uint8_t inb[sizeof(val)];
   store_be(val, inb);
   add_data(inb, sizeof(inb));
   }

void Buffered_Computation::update_be(uint32_t val)
   {
   uint8_t inb[sizeof(val)];
   store_be(val, inb);
   add_data(inb, sizeof(inb));
   }

void Buffered_Computation::update_be(uint64_t val)
   {
   uint8_t inb[sizeof(val)];
   store_be(val, inb);
   add_data(inb, sizeof(inb));
   }

void Buffered_Computation::update_le(uint16_t val)
   {
   uint8_t inb[sizeof(val)];
   store_le(val, inb);
   add_data(inb, sizeof(inb));
   }

void Buffered_Computation::update_le(uint32_t val)
   {
   uint8_t inb[sizeof(val)];
   store_le(val, inb);
   add_data(inb, sizeof(inb));
   }

void Buffered_Computation::update_le(uint64_t val)
   {
   uint8_t inb[sizeof(val)];
   store_le(val, inb);
   add_data(inb, sizeof(inb));
   }

}
/*
* SCAN Name Abstraction
* (C) 2008-2009,2015 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

std::string make_arg(const std::vector<std::pair<size_t, std::string>>& name, size_t start)
   {
   std::string output = name[start].second;
   size_t level = name[start].first;

   size_t paren_depth = 0;

   for(size_t i = start + 1; i != name.size(); ++i)
      {
      if(name[i].first <= name[start].first)
         break;

      if(name[i].first > level)
         {
         output += "(" + name[i].second;
         ++paren_depth;
         }
      else if(name[i].first < level)
         {
         for (size_t j = name[i].first; j < level; j++) {
            output += ")";
            --paren_depth;
         }
         output += "," + name[i].second;
         }
      else
         {
         if(output[output.size() - 1] != '(')
            output += ",";
         output += name[i].second;
         }

      level = name[i].first;
      }

   for(size_t i = 0; i != paren_depth; ++i)
      output += ")";

   return output;
   }

}

SCAN_Name::SCAN_Name(const char* algo_spec) : SCAN_Name(std::string(algo_spec))
   {
   }

SCAN_Name::SCAN_Name(std::string algo_spec) : m_orig_algo_spec(algo_spec), m_alg_name(), m_args(), m_mode_info()
   { 
   if(algo_spec.size() == 0)
      throw Invalid_Argument("Expected algorithm name, got empty string");

   std::vector<std::pair<size_t, std::string>> name;
   size_t level = 0;
   std::pair<size_t, std::string> accum = std::make_pair(level, "");

   const std::string decoding_error = "Bad SCAN name '" + algo_spec + "': ";

   for(size_t i = 0; i != algo_spec.size(); ++i)
      {
      char c = algo_spec[i];

      if(c == '/' || c == ',' || c == '(' || c == ')')
         {
         if(c == '(')
            ++level;
         else if(c == ')')
            {
            if(level == 0)
               throw Decoding_Error(decoding_error + "Mismatched parens");
            --level;
            }

         if(c == '/' && level > 0)
            accum.second.push_back(c);
         else
            {
            if(accum.second != "")
               name.push_back(accum);
            accum = std::make_pair(level, "");
            }
         }
      else
         accum.second.push_back(c);
      }

   if(accum.second != "")
      name.push_back(accum);

   if(level != 0)
      throw Decoding_Error(decoding_error + "Missing close paren");

   if(name.size() == 0)
      throw Decoding_Error(decoding_error + "Empty name");

   m_alg_name = name[0].second;

   bool in_modes = false;

   for(size_t i = 1; i != name.size(); ++i)
      {
      if(name[i].first == 0)
         {
         m_mode_info.push_back(make_arg(name, i));
         in_modes = true;
         }
      else if(name[i].first == 1 && !in_modes)
         m_args.push_back(make_arg(name, i));
      }
   }

std::string SCAN_Name::arg(size_t i) const
   {
   if(i >= arg_count())
      throw Invalid_Argument("SCAN_Name::arg " + std::to_string(i) +
                             " out of range for '" + to_string() + "'");
   return m_args[i];
   }

std::string SCAN_Name::arg(size_t i, const std::string& def_value) const
   {
   if(i >= arg_count())
      return def_value;
   return m_args[i];
   }

size_t SCAN_Name::arg_as_integer(size_t i, size_t def_value) const
   {
   if(i >= arg_count())
      return def_value;
   return to_u32bit(m_args[i]);
   }

}
/*
* (C) 2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

void SymmetricAlgorithm::throw_key_not_set_error() const
   {
   throw Key_Not_Set(name());
   }

void SymmetricAlgorithm::set_key(const uint8_t key[], size_t length)
   {
   if(!valid_keylength(length))
      throw Invalid_Key_Length(name(), length);
   key_schedule(key, length);
   }

}
/*
* OctetString
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/*
* Create an OctetString from RNG output
*/
OctetString::OctetString(RandomNumberGenerator& rng,
                         size_t len)
   {
   rng.random_vec(m_data, len);
   }

/*
* Create an OctetString from a hex string
*/
OctetString::OctetString(const std::string& hex_string)
   {
   if(!hex_string.empty())
      {
      m_data.resize(1 + hex_string.length() / 2);
      m_data.resize(hex_decode(m_data.data(), hex_string));
      }
   }

/*
* Create an OctetString from a byte string
*/
OctetString::OctetString(const uint8_t in[], size_t n)
   {
   m_data.assign(in, in + n);
   }

/*
* Set the parity of each key byte to odd
*/
void OctetString::set_odd_parity()
   {
   const uint8_t ODD_PARITY[256] = {
      0x01, 0x01, 0x02, 0x02, 0x04, 0x04, 0x07, 0x07, 0x08, 0x08, 0x0B, 0x0B,
      0x0D, 0x0D, 0x0E, 0x0E, 0x10, 0x10, 0x13, 0x13, 0x15, 0x15, 0x16, 0x16,
      0x19, 0x19, 0x1A, 0x1A, 0x1C, 0x1C, 0x1F, 0x1F, 0x20, 0x20, 0x23, 0x23,
      0x25, 0x25, 0x26, 0x26, 0x29, 0x29, 0x2A, 0x2A, 0x2C, 0x2C, 0x2F, 0x2F,
      0x31, 0x31, 0x32, 0x32, 0x34, 0x34, 0x37, 0x37, 0x38, 0x38, 0x3B, 0x3B,
      0x3D, 0x3D, 0x3E, 0x3E, 0x40, 0x40, 0x43, 0x43, 0x45, 0x45, 0x46, 0x46,
      0x49, 0x49, 0x4A, 0x4A, 0x4C, 0x4C, 0x4F, 0x4F, 0x51, 0x51, 0x52, 0x52,
      0x54, 0x54, 0x57, 0x57, 0x58, 0x58, 0x5B, 0x5B, 0x5D, 0x5D, 0x5E, 0x5E,
      0x61, 0x61, 0x62, 0x62, 0x64, 0x64, 0x67, 0x67, 0x68, 0x68, 0x6B, 0x6B,
      0x6D, 0x6D, 0x6E, 0x6E, 0x70, 0x70, 0x73, 0x73, 0x75, 0x75, 0x76, 0x76,
      0x79, 0x79, 0x7A, 0x7A, 0x7C, 0x7C, 0x7F, 0x7F, 0x80, 0x80, 0x83, 0x83,
      0x85, 0x85, 0x86, 0x86, 0x89, 0x89, 0x8A, 0x8A, 0x8C, 0x8C, 0x8F, 0x8F,
      0x91, 0x91, 0x92, 0x92, 0x94, 0x94, 0x97, 0x97, 0x98, 0x98, 0x9B, 0x9B,
      0x9D, 0x9D, 0x9E, 0x9E, 0xA1, 0xA1, 0xA2, 0xA2, 0xA4, 0xA4, 0xA7, 0xA7,
      0xA8, 0xA8, 0xAB, 0xAB, 0xAD, 0xAD, 0xAE, 0xAE, 0xB0, 0xB0, 0xB3, 0xB3,
      0xB5, 0xB5, 0xB6, 0xB6, 0xB9, 0xB9, 0xBA, 0xBA, 0xBC, 0xBC, 0xBF, 0xBF,
      0xC1, 0xC1, 0xC2, 0xC2, 0xC4, 0xC4, 0xC7, 0xC7, 0xC8, 0xC8, 0xCB, 0xCB,
      0xCD, 0xCD, 0xCE, 0xCE, 0xD0, 0xD0, 0xD3, 0xD3, 0xD5, 0xD5, 0xD6, 0xD6,
      0xD9, 0xD9, 0xDA, 0xDA, 0xDC, 0xDC, 0xDF, 0xDF, 0xE0, 0xE0, 0xE3, 0xE3,
      0xE5, 0xE5, 0xE6, 0xE6, 0xE9, 0xE9, 0xEA, 0xEA, 0xEC, 0xEC, 0xEF, 0xEF,
      0xF1, 0xF1, 0xF2, 0xF2, 0xF4, 0xF4, 0xF7, 0xF7, 0xF8, 0xF8, 0xFB, 0xFB,
      0xFD, 0xFD, 0xFE, 0xFE };

   for(size_t j = 0; j != m_data.size(); ++j)
      m_data[j] = ODD_PARITY[m_data[j]];
   }

/*
* Hex encode an OctetString
*/
std::string OctetString::to_string() const
   {
   return hex_encode(m_data.data(), m_data.size());
   }

/*
* XOR Operation for OctetStrings
*/
OctetString& OctetString::operator^=(const OctetString& k)
   {
   if(&k == this) { zeroise(m_data); return (*this); }
   xor_buf(m_data.data(), k.begin(), std::min(length(), k.length()));
   return (*this);
   }

/*
* Equality Operation for OctetStrings
*/
bool operator==(const OctetString& s1, const OctetString& s2)
   {
   return (s1.bits_of() == s2.bits_of());
   }

/*
* Unequality Operation for OctetStrings
*/
bool operator!=(const OctetString& s1, const OctetString& s2)
   {
   return !(s1 == s2);
   }

/*
* Append Operation for OctetStrings
*/
OctetString operator+(const OctetString& k1, const OctetString& k2)
   {
   secure_vector<uint8_t> out;
   out += k1.bits_of();
   out += k2.bits_of();
   return OctetString(out);
   }

/*
* XOR Operation for OctetStrings
*/
OctetString operator^(const OctetString& k1, const OctetString& k2)
   {
   secure_vector<uint8_t> out(std::max(k1.length(), k2.length()));

   copy_mem(out.data(), k1.begin(), k1.length());
   xor_buf(out.data(), k2.begin(), k2.length());
   return OctetString(out);
   }

}
/*
* Base64 Encoding and Decoding
* (C) 2010,2015,2020 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

class Base64 final
   {
   public:
      static inline std::string name() noexcept
         {
         return "base64";
         }

      static inline size_t encoding_bytes_in() noexcept
         {
         return m_encoding_bytes_in;
         }
      static inline size_t encoding_bytes_out() noexcept
         {
         return m_encoding_bytes_out;
         }

      static inline size_t decoding_bytes_in() noexcept
         {
         return m_encoding_bytes_out;
         }
      static inline size_t decoding_bytes_out() noexcept
         {
         return m_encoding_bytes_in;
         }

      static inline size_t bits_consumed() noexcept
         {
         return m_encoding_bits;
         }
      static inline size_t remaining_bits_before_padding() noexcept
         {
         return m_remaining_bits_before_padding;
         }

      static inline size_t encode_max_output(size_t input_length)
         {
         return (round_up(input_length, m_encoding_bytes_in) / m_encoding_bytes_in) * m_encoding_bytes_out;
         }
      static inline size_t decode_max_output(size_t input_length)
         {
         return (round_up(input_length, m_encoding_bytes_out) * m_encoding_bytes_in) / m_encoding_bytes_out;
         }

      static void encode(char out[8], const uint8_t in[5]) noexcept;

      static uint8_t lookup_binary_value(char input) noexcept;

      static bool check_bad_char(uint8_t bin, char input, bool ignore_ws);

      static void decode(uint8_t* out_ptr, const uint8_t decode_buf[4])
         {
         out_ptr[0] = (decode_buf[0] << 2) | (decode_buf[1] >> 4);
         out_ptr[1] = (decode_buf[1] << 4) | (decode_buf[2] >> 2);
         out_ptr[2] = (decode_buf[2] << 6) | decode_buf[3];
         }

      static inline size_t bytes_to_remove(size_t final_truncate)
         {
         return final_truncate;
         }

   private:
      static const size_t m_encoding_bits = 6;
      static const size_t m_remaining_bits_before_padding = 8;

      static const size_t m_encoding_bytes_in = 3;
      static const size_t m_encoding_bytes_out = 4;
   };

char lookup_base64_char(uint8_t x)
   {
   BOTAN_DEBUG_ASSERT(x < 64);

   const auto in_az = CT::Mask<uint8_t>::is_within_range(x, 26, 51);
   const auto in_09 = CT::Mask<uint8_t>::is_within_range(x, 52, 61);
   const auto eq_plus = CT::Mask<uint8_t>::is_equal(x, 62);
   const auto eq_slash = CT::Mask<uint8_t>::is_equal(x, 63);

   const char c_AZ = 'A' + x;
   const char c_az = 'a' + (x - 26);
   const char c_09 = '0' + (x - 2*26);
   const char c_plus = '+';
   const char c_slash = '/';

   char ret = c_AZ;
   ret = in_az.select(c_az, ret);
   ret = in_09.select(c_09, ret);
   ret = eq_plus.select(c_plus, ret);
   ret = eq_slash.select(c_slash, ret);

   return ret;
   }

//static
void Base64::encode(char out[8], const uint8_t in[5]) noexcept
   {
   const uint8_t b0 = (in[0] & 0xFC) >> 2;
   const uint8_t b1 = ((in[0] & 0x03) << 4) | (in[1] >> 4);
   const uint8_t b2 = ((in[1] & 0x0F) << 2) | (in[2] >> 6);
   const uint8_t b3 = in[2] & 0x3F;
   out[0] = lookup_base64_char(b0);
   out[1] = lookup_base64_char(b1);
   out[2] = lookup_base64_char(b2);
   out[3] = lookup_base64_char(b3);
   }

//static
uint8_t Base64::lookup_binary_value(char input) noexcept
   {
   const uint8_t c = static_cast<uint8_t>(input);

   const auto is_alpha_upper = CT::Mask<uint8_t>::is_within_range(c, uint8_t('A'), uint8_t('Z'));
   const auto is_alpha_lower = CT::Mask<uint8_t>::is_within_range(c, uint8_t('a'), uint8_t('z'));
   const auto is_decimal     = CT::Mask<uint8_t>::is_within_range(c, uint8_t('0'), uint8_t('9'));

   const auto is_plus        = CT::Mask<uint8_t>::is_equal(c, uint8_t('+'));
   const auto is_slash       = CT::Mask<uint8_t>::is_equal(c, uint8_t('/'));
   const auto is_equal       = CT::Mask<uint8_t>::is_equal(c, uint8_t('='));

   const auto is_whitespace  = CT::Mask<uint8_t>::is_any_of(c, {
         uint8_t(' '), uint8_t('\t'), uint8_t('\n'), uint8_t('\r')
      });

   const uint8_t c_upper = c - uint8_t('A');
   const uint8_t c_lower = c - uint8_t('a') + 26;
   const uint8_t c_decim = c - uint8_t('0') + 2*26;

   uint8_t ret = 0xFF; // default value

   ret = is_alpha_upper.select(c_upper, ret);
   ret = is_alpha_lower.select(c_lower, ret);
   ret = is_decimal.select(c_decim, ret);
   ret = is_plus.select(62, ret);
   ret = is_slash.select(63, ret);
   ret = is_equal.select(0x81, ret);
   ret = is_whitespace.select(0x80, ret);

   return ret;
   }

//static
bool Base64::check_bad_char(uint8_t bin, char input, bool ignore_ws)
   {
   if(bin <= 0x3F)
      {
      return true;
      }
   else if(!(bin == 0x81 || (bin == 0x80 && ignore_ws)))
      {
      std::string bad_char(1, input);
      if(bad_char == "\t")
         { bad_char = "\\t"; }
      else if(bad_char == "\n")
         { bad_char = "\\n"; }
      else if(bad_char == "\r")
         { bad_char = "\\r"; }

      throw Invalid_Argument(
         std::string("base64_decode: invalid base64 character '") +
         bad_char + "'");
      }
   return false;
   }

}

size_t base64_encode(char out[],
                     const uint8_t in[],
                     size_t input_length,
                     size_t& input_consumed,
                     bool final_inputs)
   {
   return base_encode(Base64(), out, in, input_length, input_consumed, final_inputs);
   }

std::string base64_encode(const uint8_t input[],
                          size_t input_length)
   {
   return base_encode_to_string(Base64(), input, input_length);
   }

size_t base64_decode(uint8_t out[],
                     const char in[],
                     size_t input_length,
                     size_t& input_consumed,
                     bool final_inputs,
                     bool ignore_ws)
   {
   return base_decode(Base64(), out, in, input_length, input_consumed, final_inputs, ignore_ws);
   }

size_t base64_decode(uint8_t output[],
                     const char input[],
                     size_t input_length,
                     bool ignore_ws)
   {
   return base_decode_full(Base64(), output, input, input_length, ignore_ws);
   }

size_t base64_decode(uint8_t output[],
                     const std::string& input,
                     bool ignore_ws)
   {
   return base64_decode(output, input.data(), input.length(), ignore_ws);
   }

secure_vector<uint8_t> base64_decode(const char input[],
                                     size_t input_length,
                                     bool ignore_ws)
   {
   return base_decode_to_vec<secure_vector<uint8_t>>(Base64(), input, input_length, ignore_ws);
   }

secure_vector<uint8_t> base64_decode(const std::string& input,
                                     bool ignore_ws)
   {
   return base64_decode(input.data(), input.size(), ignore_ws);
   }

size_t base64_encode_max_output(size_t input_length)
   {
   return Base64::encode_max_output(input_length);
   }

size_t base64_decode_max_output(size_t input_length)
   {
   return Base64::decode_max_output(input_length);
   }

}
/*
* BigInt Encoding/Decoding
* (C) 1999-2010,2012,2019 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

std::string BigInt::to_dec_string() const
   {
   BigInt copy = *this;
   copy.set_sign(Positive);

   uint8_t remainder;
   std::vector<uint8_t> digits;

   while(copy > 0)
      {
      ct_divide_u8(copy, 10, copy, remainder);
      digits.push_back(remainder);
      }

   std::string s;

   for(auto i = digits.rbegin(); i != digits.rend(); ++i)
      {
      s.push_back(Charset::digit2char(*i));
      }

   if(s.empty())
      s += "0";

   return s;
   }

std::string BigInt::to_hex_string() const
   {
   const std::vector<uint8_t> bits = BigInt::encode(*this);
   if(bits.empty())
      return "00";
   else
      return hex_encode(bits);
   }

/*
* Encode a BigInt
*/
void BigInt::encode(uint8_t output[], const BigInt& n, Base base)
   {
   secure_vector<uint8_t> enc = n.encode_locked(base);
   copy_mem(output, enc.data(), enc.size());
   }

namespace {

std::vector<uint8_t> str_to_vector(const std::string& s)
   {
   std::vector<uint8_t> v(s.size());
   std::memcpy(v.data(), s.data(), s.size());
   return v;
   }

secure_vector<uint8_t> str_to_lvector(const std::string& s)
   {
   secure_vector<uint8_t> v(s.size());
   std::memcpy(v.data(), s.data(), s.size());
   return v;
   }

}

/*
* Encode a BigInt
*/
std::vector<uint8_t> BigInt::encode(const BigInt& n, Base base)
   {
   if(base == Binary)
      return BigInt::encode(n);
   else if(base == Hexadecimal)
      return str_to_vector(n.to_hex_string());
   else if(base == Decimal)
      return str_to_vector(n.to_dec_string());
   else
      throw Invalid_Argument("Unknown BigInt encoding base");
   }

/*
* Encode a BigInt
*/
secure_vector<uint8_t> BigInt::encode_locked(const BigInt& n, Base base)
   {
   if(base == Binary)
      return BigInt::encode_locked(n);
   else if(base == Hexadecimal)
      return str_to_lvector(n.to_hex_string());
   else if(base == Decimal)
      return str_to_lvector(n.to_dec_string());
   else
      throw Invalid_Argument("Unknown BigInt encoding base");
   }

/*
* Encode a BigInt, with leading 0s if needed
*/
secure_vector<uint8_t> BigInt::encode_1363(const BigInt& n, size_t bytes)
   {
   if(n.bytes() > bytes)
      throw Encoding_Error("encode_1363: n is too large to encode properly");

   secure_vector<uint8_t> output(bytes);
   n.binary_encode(output.data(), output.size());
   return output;
   }

//static
void BigInt::encode_1363(uint8_t output[], size_t bytes, const BigInt& n)
   {
   if(n.bytes() > bytes)
      throw Encoding_Error("encode_1363: n is too large to encode properly");

   n.binary_encode(output, bytes);
   }

/*
* Encode two BigInt, with leading 0s if needed, and concatenate
*/
secure_vector<uint8_t> BigInt::encode_fixed_length_int_pair(const BigInt& n1, const BigInt& n2, size_t bytes)
   {
   if(n1.bytes() > bytes || n2.bytes() > bytes)
      throw Encoding_Error("encode_fixed_length_int_pair: values too large to encode properly");
   secure_vector<uint8_t> output(2 * bytes);
   n1.binary_encode(output.data()        , bytes);
   n2.binary_encode(output.data() + bytes, bytes);
   return output;
   }

/*
* Decode a BigInt
*/
BigInt BigInt::decode(const uint8_t buf[], size_t length, Base base)
   {
   BigInt r;
   if(base == Binary)
      {
      r.binary_decode(buf, length);
      }
   else if(base == Hexadecimal)
      {
      secure_vector<uint8_t> binary;

      if(length % 2)
         {
         // Handle lack of leading 0
         const char buf0_with_leading_0[2] =
            { '0', static_cast<char>(buf[0]) };

         binary = hex_decode_locked(buf0_with_leading_0, 2);

         binary += hex_decode_locked(cast_uint8_ptr_to_char(&buf[1]),
                                     length - 1,
                                     false);
         }
      else
         binary = hex_decode_locked(cast_uint8_ptr_to_char(buf),
                                    length, false);

      r.binary_decode(binary.data(), binary.size());
      }
   else if(base == Decimal)
      {
      for(size_t i = 0; i != length; ++i)
         {
         if(Charset::is_space(buf[i]))
            continue;

         if(!Charset::is_digit(buf[i]))
            throw Invalid_Argument("BigInt::decode: "
                                   "Invalid character in decimal input");

         const uint8_t x = Charset::char2digit(buf[i]);

         if(x >= 10)
            throw Invalid_Argument("BigInt: Invalid decimal string");

         r *= 10;
         r += x;
         }
      }
   else
      throw Invalid_Argument("Unknown BigInt decoding method");
   return r;
   }

}
/*
* BigInt Input/Output
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <istream>
#include <ostream>

namespace Botan {

/*
* Write the BigInt into a stream
*/
std::ostream& operator<<(std::ostream& stream, const BigInt& n)
   {
   size_t base = 10;
   if(stream.flags() & std::ios::hex)
      base = 16;
   if(stream.flags() & std::ios::oct)
      throw Invalid_Argument("Octal output of BigInt not supported");

   if(n == 0)
      stream.write("0", 1);
   else
      {
      if(n < 0)
         stream.write("-", 1);

      std::string enc;

      if(base == 10)
         enc = n.to_dec_string();
      else
         enc = n.to_hex_string();

      size_t skip = 0;
      while(skip < enc.size() && enc[skip] == '0')
         ++skip;
      stream.write(&enc[skip], enc.size() - skip);
      }
   if(!stream.good())
      throw Stream_IO_Error("BigInt output operator has failed");
   return stream;
   }

/*
* Read the BigInt from a stream
*/
std::istream& operator>>(std::istream& stream, BigInt& n)
   {
   std::string str;
   std::getline(stream, str);
   if(stream.bad() || (stream.fail() && !stream.eof()))
      throw Stream_IO_Error("BigInt input operator has failed");
   n = BigInt(str);
   return stream;
   }

}
/*
* (C) 1999-2007,2018 Jack Lloyd
*     2016 Matthias Gierlings
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

BigInt& BigInt::add(const word y[], size_t y_words, Sign y_sign)
   {
   const size_t x_sw = sig_words();

   grow_to(std::max(x_sw, y_words) + 1);

   if(sign() == y_sign)
      {
      bigint_add2(mutable_data(), size() - 1, y, y_words);
      }
   else
      {
      const int32_t relative_size = bigint_cmp(data(), x_sw, y, y_words);

      if(relative_size >= 0)
         {
         // *this >= y
         bigint_sub2(mutable_data(), x_sw, y, y_words);
         }
      else
         {
         // *this < y
         bigint_sub2_rev(mutable_data(), y, y_words);
         }

      //this->sign_fixup(relative_size, y_sign);
      if(relative_size < 0)
         set_sign(y_sign);
      else if(relative_size == 0)
         set_sign(Positive);
      }

   return (*this);
   }

BigInt& BigInt::mod_add(const BigInt& s, const BigInt& mod, secure_vector<word>& ws)
   {
   if(this->is_negative() || s.is_negative() || mod.is_negative())
      throw Invalid_Argument("BigInt::mod_add expects all arguments are positive");

   BOTAN_DEBUG_ASSERT(*this < mod);
   BOTAN_DEBUG_ASSERT(s < mod);

   /*
   t + s or t + s - p == t - (p - s)

   So first compute ws = p - s

   Then compute t + s and t - ws

   If t - ws does not borrow, then that is the correct valued
   */

   const size_t mod_sw = mod.sig_words();
   BOTAN_ARG_CHECK(mod_sw > 0, "BigInt::mod_add modulus must be positive");

   this->grow_to(mod_sw);
   s.grow_to(mod_sw);

   // First mod_sw for p - s, 2*mod_sw for bigint_addsub workspace
   if(ws.size() < 3*mod_sw)
      ws.resize(3*mod_sw);

   word borrow = bigint_sub3(&ws[0], mod.data(), mod_sw, s.data(), mod_sw);
   BOTAN_DEBUG_ASSERT(borrow == 0);

   // Compute t - ws
   borrow = bigint_sub3(&ws[mod_sw], this->data(), mod_sw, &ws[0], mod_sw);

   // Compute t + s
   bigint_add3_nc(&ws[mod_sw*2], this->data(), mod_sw, s.data(), mod_sw);

   CT::conditional_copy_mem(borrow, &ws[0], &ws[mod_sw*2], &ws[mod_sw], mod_sw);
   set_words(&ws[0], mod_sw);

   return (*this);
   }

BigInt& BigInt::mod_sub(const BigInt& s, const BigInt& mod, secure_vector<word>& ws)
   {
   if(this->is_negative() || s.is_negative() || mod.is_negative())
      throw Invalid_Argument("BigInt::mod_sub expects all arguments are positive");

   // We are assuming in this function that *this and s are no more than mod_sw words long
   BOTAN_DEBUG_ASSERT(*this < mod);
   BOTAN_DEBUG_ASSERT(s < mod);

   const size_t mod_sw = mod.sig_words();

   this->grow_to(mod_sw);
   s.grow_to(mod_sw);

   if(ws.size() < mod_sw)
      ws.resize(mod_sw);

   if(mod_sw == 4)
      bigint_mod_sub_n<4>(mutable_data(), s.data(), mod.data(), ws.data());
   else if(mod_sw == 6)
      bigint_mod_sub_n<6>(mutable_data(), s.data(), mod.data(), ws.data());
   else
      bigint_mod_sub(mutable_data(), s.data(), mod.data(), mod_sw, ws.data());

   return (*this);
   }

BigInt& BigInt::mod_mul(uint8_t y, const BigInt& mod, secure_vector<word>& ws)
   {
   BOTAN_ARG_CHECK(this->is_negative() == false, "*this must be positive");
   BOTAN_ARG_CHECK(y < 16, "y too large");

   BOTAN_DEBUG_ASSERT(*this < mod);

   *this *= static_cast<word>(y);
   this->reduce_below(mod, ws);
   return (*this);
   }

BigInt& BigInt::rev_sub(const word y[], size_t y_sw, secure_vector<word>& ws)
   {
   if(this->sign() != BigInt::Positive)
      throw Invalid_State("BigInt::sub_rev requires this is positive");

   const size_t x_sw = this->sig_words();

   ws.resize(std::max(x_sw, y_sw));
   clear_mem(ws.data(), ws.size());

   const int32_t relative_size = bigint_sub_abs(ws.data(), data(), x_sw, y, y_sw);

   this->cond_flip_sign(relative_size > 0);
   this->swap_reg(ws);

   return (*this);
   }

/*
* Multiplication Operator
*/
BigInt& BigInt::operator*=(const BigInt& y)
   {
   secure_vector<word> ws;
   return this->mul(y, ws);
   }

BigInt& BigInt::mul(const BigInt& y, secure_vector<word>& ws)
   {
   const size_t x_sw = sig_words();
   const size_t y_sw = y.sig_words();
   set_sign((sign() == y.sign()) ? Positive : Negative);

   if(x_sw == 0 || y_sw == 0)
      {
      clear();
      set_sign(Positive);
      }
   else if(x_sw == 1 && y_sw)
      {
      grow_to(y_sw + 1);
      bigint_linmul3(mutable_data(), y.data(), y_sw, word_at(0));
      }
   else if(y_sw == 1 && x_sw)
      {
      word carry = bigint_linmul2(mutable_data(), x_sw, y.word_at(0));
      set_word_at(x_sw, carry);
      }
   else
      {
      const size_t new_size = x_sw + y_sw + 1;
      ws.resize(new_size);
      secure_vector<word> z_reg(new_size);

      bigint_mul(z_reg.data(), z_reg.size(),
                 data(), size(), x_sw,
                 y.data(), y.size(), y_sw,
                 ws.data(), ws.size());

      this->swap_reg(z_reg);
      }

   return (*this);
   }

BigInt& BigInt::square(secure_vector<word>& ws)
   {
   const size_t sw = sig_words();

   secure_vector<word> z(2*sw);
   ws.resize(z.size());

   bigint_sqr(z.data(), z.size(),
              data(), size(), sw,
              ws.data(), ws.size());

   swap_reg(z);
   set_sign(BigInt::Positive);

   return (*this);
   }

BigInt& BigInt::operator*=(word y)
   {
   if(y == 0)
      {
      clear();
      set_sign(Positive);
      }

   const word carry = bigint_linmul2(mutable_data(), size(), y);
   set_word_at(size(), carry);

   return (*this);
   }

/*
* Division Operator
*/
BigInt& BigInt::operator/=(const BigInt& y)
   {
   if(y.sig_words() == 1 && is_power_of_2(y.word_at(0)))
      (*this) >>= (y.bits() - 1);
   else
      (*this) = (*this) / y;
   return (*this);
   }

/*
* Modulo Operator
*/
BigInt& BigInt::operator%=(const BigInt& mod)
   {
   return (*this = (*this) % mod);
   }

/*
* Modulo Operator
*/
word BigInt::operator%=(word mod)
   {
   if(mod == 0)
      throw BigInt::DivideByZero();

   word remainder = 0;

   if(is_power_of_2(mod))
       {
       remainder = (word_at(0) & (mod - 1));
       }
   else
      {
      const size_t sw = sig_words();
      for(size_t i = sw; i > 0; --i)
         remainder = bigint_modop(remainder, word_at(i-1), mod);
      }

   if(remainder && sign() == BigInt::Negative)
      remainder = mod - remainder;

   m_data.set_to_zero();
   m_data.set_word_at(0, remainder);
   set_sign(BigInt::Positive);
   return remainder;
   }

/*
* Left Shift Operator
*/
BigInt& BigInt::operator<<=(size_t shift)
   {
   const size_t shift_words = shift / BOTAN_MP_WORD_BITS;
   const size_t shift_bits  = shift % BOTAN_MP_WORD_BITS;
   const size_t size = sig_words();

   const size_t bits_free = top_bits_free();

   const size_t new_size = size + shift_words + (bits_free < shift_bits);

   m_data.grow_to(new_size);

   bigint_shl1(m_data.mutable_data(), new_size, size, shift_words, shift_bits);

   return (*this);
   }

/*
* Right Shift Operator
*/
BigInt& BigInt::operator>>=(size_t shift)
   {
   const size_t shift_words = shift / BOTAN_MP_WORD_BITS;
   const size_t shift_bits  = shift % BOTAN_MP_WORD_BITS;

   bigint_shr1(m_data.mutable_data(), m_data.size(), shift_words, shift_bits);

   if(is_negative() && is_zero())
      set_sign(Positive);

   return (*this);
   }

}
/*
* BigInt Binary Operators
* (C) 1999-2007,2018 Jack Lloyd
*     2016 Matthias Gierlings
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

//static
BigInt BigInt::add2(const BigInt& x, const word y[], size_t y_words, BigInt::Sign y_sign)
   {
   const size_t x_sw = x.sig_words();

   BigInt z(x.sign(), std::max(x_sw, y_words) + 1);

   if(x.sign() == y_sign)
      {
      bigint_add3(z.mutable_data(), x.data(), x_sw, y, y_words);
      }
   else
      {
      const int32_t relative_size = bigint_sub_abs(z.mutable_data(), x.data(), x_sw, y, y_words);

      //z.sign_fixup(relative_size, y_sign);
      if(relative_size < 0)
         z.set_sign(y_sign);
      else if(relative_size == 0)
         z.set_sign(BigInt::Positive);
      }

   return z;
   }

/*
* Multiplication Operator
*/
BigInt operator*(const BigInt& x, const BigInt& y)
   {
   const size_t x_sw = x.sig_words();
   const size_t y_sw = y.sig_words();

   BigInt z(BigInt::Positive, x.size() + y.size());

   if(x_sw == 1 && y_sw)
      bigint_linmul3(z.mutable_data(), y.data(), y_sw, x.word_at(0));
   else if(y_sw == 1 && x_sw)
      bigint_linmul3(z.mutable_data(), x.data(), x_sw, y.word_at(0));
   else if(x_sw && y_sw)
      {
      secure_vector<word> workspace(z.size());

      bigint_mul(z.mutable_data(), z.size(),
                 x.data(), x.size(), x_sw,
                 y.data(), y.size(), y_sw,
                 workspace.data(), workspace.size());
      }

   z.cond_flip_sign(x_sw > 0 && y_sw > 0 && x.sign() != y.sign());

   return z;
   }

/*
* Multiplication Operator
*/
BigInt operator*(const BigInt& x, word y)
   {
   const size_t x_sw = x.sig_words();

   BigInt z(BigInt::Positive, x_sw + 1);

   if(x_sw && y)
      {
      bigint_linmul3(z.mutable_data(), x.data(), x_sw, y);
      z.set_sign(x.sign());
      }

   return z;
   }

/*
* Division Operator
*/
BigInt operator/(const BigInt& x, const BigInt& y)
   {
   if(y.sig_words() == 1)
      {
      return x / y.word_at(0);
      }

   BigInt q, r;
   vartime_divide(x, y, q, r);
   return q;
   }

/*
* Division Operator
*/
BigInt operator/(const BigInt& x, word y)
   {
   if(y == 0)
      throw BigInt::DivideByZero();
   else if(y == 1)
      return x;
   else if(y == 2)
      return (x >> 1);
   else if(y <= 255)
      {
      BigInt q;
      uint8_t r;
      ct_divide_u8(x, static_cast<uint8_t>(y), q, r);
      return q;
      }

   BigInt q, r;
   vartime_divide(x, y, q, r);
   return q;
   }

/*
* Modulo Operator
*/
BigInt operator%(const BigInt& n, const BigInt& mod)
   {
   if(mod.is_zero())
      throw BigInt::DivideByZero();
   if(mod.is_negative())
      throw Invalid_Argument("BigInt::operator%: modulus must be > 0");
   if(n.is_positive() && mod.is_positive() && n < mod)
      return n;

   if(mod.sig_words() == 1)
      {
      return n % mod.word_at(0);
      }

   BigInt q, r;
   vartime_divide(n, mod, q, r);
   return r;
   }

/*
* Modulo Operator
*/
word operator%(const BigInt& n, word mod)
   {
   if(mod == 0)
      throw BigInt::DivideByZero();

   if(mod == 1)
      return 0;

   word remainder = 0;

   if(is_power_of_2(mod))
      {
      remainder = (n.word_at(0) & (mod - 1));
      }
   else
      {
      const size_t sw = n.sig_words();
      for(size_t i = sw; i > 0; --i)
         {
         remainder = bigint_modop(remainder, n.word_at(i-1), mod);
         }
      }

   if(remainder && n.sign() == BigInt::Negative)
      return mod - remainder;
   return remainder;
   }

/*
* Left Shift Operator
*/
BigInt operator<<(const BigInt& x, size_t shift)
   {
   const size_t shift_words = shift / BOTAN_MP_WORD_BITS,
                shift_bits  = shift % BOTAN_MP_WORD_BITS;

   const size_t x_sw = x.sig_words();

   BigInt y(x.sign(), x_sw + shift_words + (shift_bits ? 1 : 0));
   bigint_shl2(y.mutable_data(), x.data(), x_sw, shift_words, shift_bits);
   return y;
   }

/*
* Right Shift Operator
*/
BigInt operator>>(const BigInt& x, size_t shift)
   {
   const size_t shift_words = shift / BOTAN_MP_WORD_BITS;
   const size_t shift_bits  = shift % BOTAN_MP_WORD_BITS;
   const size_t x_sw = x.sig_words();

   BigInt y(x.sign(), x_sw - shift_words);
   bigint_shr2(y.mutable_data(), x.data(), x_sw, shift_words, shift_bits);

   if(x.is_negative() && y.is_zero())
      y.set_sign(BigInt::Positive);

   return y;
   }

}
/*
* BigInt Random Generation
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/*
* Randomize this number
*/
void BigInt::randomize(RandomNumberGenerator& rng,
                       size_t bitsize, bool set_high_bit)
   {
   set_sign(Positive);

   if(bitsize == 0)
      {
      clear();
      }
   else
      {
      secure_vector<uint8_t> array = rng.random_vec(round_up(bitsize, 8) / 8);

      // Always cut unwanted bits
      if(bitsize % 8)
         array[0] &= 0xFF >> (8 - (bitsize % 8));

      // Set the highest bit if wanted
      if (set_high_bit)
         array[0] |= 0x80 >> ((bitsize % 8) ? (8 - bitsize % 8) : 0);

      binary_decode(array);
      }
   }

/*
* Generate a random integer within given range
*/
BigInt BigInt::random_integer(RandomNumberGenerator& rng,
                              const BigInt& min, const BigInt& max)
   {
   if(min.is_negative() || max.is_negative() || max <= min)
      throw Invalid_Argument("BigInt::random_integer invalid range");

   BigInt r;

   const size_t bits = max.bits();

   do
      {
      r.randomize(rng, bits, false);
      }
   while(r < min || r >= max);

   return r;
   }

}
/*
* BigInt Base
* (C) 1999-2011,2012,2014,2019 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

BigInt::BigInt(const word words[], size_t length)
   {
   m_data.set_words(words, length);
   }

/*
* Construct a BigInt from a regular number
*/
BigInt::BigInt(uint64_t n)
   {
   if(n > 0)
      {
#if BOTAN_MP_WORD_BITS == 32
      m_data.set_word_at(0, static_cast<word>(n));
      m_data.set_word_at(1, static_cast<word>(n >> 32));
#else
      m_data.set_word_at(0, n);
#endif
      }

   }

/*
* Construct a BigInt of the specified size
*/
BigInt::BigInt(Sign s, size_t size)
   {
   m_data.set_size(size);
   m_signedness = s;
   }

/*
* Construct a BigInt from a string
*/
BigInt::BigInt(const std::string& str)
   {
   Base base = Decimal;
   size_t markers = 0;
   bool negative = false;

   if(str.length() > 0 && str[0] == '-')
      {
      markers += 1;
      negative = true;
      }

   if(str.length() > markers + 2 && str[markers    ] == '0' &&
                                    str[markers + 1] == 'x')
      {
      markers += 2;
      base = Hexadecimal;
      }

   *this = decode(cast_char_ptr_to_uint8(str.data()) + markers,
                  str.length() - markers, base);

   if(negative) set_sign(Negative);
   else         set_sign(Positive);
   }

BigInt::BigInt(const uint8_t input[], size_t length)
   {
   binary_decode(input, length);
   }

/*
* Construct a BigInt from an encoded BigInt
*/
BigInt::BigInt(const uint8_t input[], size_t length, Base base)
   {
   *this = decode(input, length, base);
   }

BigInt::BigInt(const uint8_t buf[], size_t length, size_t max_bits)
   {
   if(8 * length > max_bits)
      length = (max_bits + 7) / 8;

   binary_decode(buf, length);

   if(8 * length > max_bits)
      *this >>= (8 - (max_bits % 8));
   }

/*
* Construct a BigInt from an encoded BigInt
*/
BigInt::BigInt(RandomNumberGenerator& rng, size_t bits, bool set_high_bit)
   {
   randomize(rng, bits, set_high_bit);
   }

uint8_t BigInt::byte_at(size_t n) const
   {
   return get_byte(sizeof(word) - (n % sizeof(word)) - 1,
                   word_at(n / sizeof(word)));
   }

int32_t BigInt::cmp_word(word other) const
   {
   if(is_negative())
      return -1; // other is positive ...

   const size_t sw = this->sig_words();
   if(sw > 1)
      return 1; // must be larger since other is just one word ...

   return bigint_cmp(this->data(), sw, &other, 1);
   }

/*
* Comparison Function
*/
int32_t BigInt::cmp(const BigInt& other, bool check_signs) const
   {
   if(check_signs)
      {
      if(other.is_positive() && this->is_negative())
         return -1;

      if(other.is_negative() && this->is_positive())
         return 1;

      if(other.is_negative() && this->is_negative())
         return (-bigint_cmp(this->data(), this->size(),
                             other.data(), other.size()));
      }

   return bigint_cmp(this->data(), this->size(),
                     other.data(), other.size());
   }

bool BigInt::is_equal(const BigInt& other) const
   {
   if(this->sign() != other.sign())
      return false;

   return bigint_ct_is_eq(this->data(), this->sig_words(),
                          other.data(), other.sig_words()).is_set();
   }

bool BigInt::is_less_than(const BigInt& other) const
   {
   if(this->is_negative() && other.is_positive())
      return true;

   if(this->is_positive() && other.is_negative())
      return false;

   if(other.is_negative() && this->is_negative())
      {
      return bigint_ct_is_lt(other.data(), other.sig_words(),
                             this->data(), this->sig_words()).is_set();
      }

   return bigint_ct_is_lt(this->data(), this->sig_words(),
                          other.data(), other.sig_words()).is_set();
   }

void BigInt::encode_words(word out[], size_t size) const
   {
   const size_t words = sig_words();

   if(words > size)
      throw Encoding_Error("BigInt::encode_words value too large to encode");

   clear_mem(out, size);
   copy_mem(out, data(), words);
   }

size_t BigInt::Data::calc_sig_words() const
   {
   const size_t sz = m_reg.size();
   size_t sig = sz;

   word sub = 1;

   for(size_t i = 0; i != sz; ++i)
      {
      const word w = m_reg[sz - i - 1];
      sub &= ct_is_zero(w);
      sig -= sub;
      }

   /*
   * This depends on the data so is poisoned, but unpoison it here as
   * later conditionals are made on the size.
   */
   CT::unpoison(sig);

   return sig;
   }

/*
* Return bits {offset...offset+length}
*/
uint32_t BigInt::get_substring(size_t offset, size_t length) const
   {
   if(length == 0 || length > 32)
      throw Invalid_Argument("BigInt::get_substring invalid substring length");

   const uint32_t mask = 0xFFFFFFFF >> (32 - length);

   const size_t word_offset = offset / BOTAN_MP_WORD_BITS;
   const size_t wshift = (offset % BOTAN_MP_WORD_BITS);

   /*
   * The substring is contained within one or at most two words. The
   * offset and length are not secret, so we can perform conditional
   * operations on those values.
   */
   const word w0 = word_at(word_offset);

   if(wshift == 0 || (offset + length) / BOTAN_MP_WORD_BITS == word_offset)
      {
      return static_cast<uint32_t>(w0 >> wshift) & mask;
      }
   else
      {
      const word w1 = word_at(word_offset + 1);
      return static_cast<uint32_t>((w0 >> wshift) | (w1 << (BOTAN_MP_WORD_BITS - wshift))) & mask;
      }
   }

/*
* Convert this number to a uint32_t, if possible
*/
uint32_t BigInt::to_u32bit() const
   {
   if(is_negative())
      throw Encoding_Error("BigInt::to_u32bit: Number is negative");
   if(bits() > 32)
      throw Encoding_Error("BigInt::to_u32bit: Number is too big to convert");

   uint32_t out = 0;
   for(size_t i = 0; i != 4; ++i)
      out = (out << 8) | byte_at(3-i);
   return out;
   }

/*
* Set bit number n
*/
void BigInt::conditionally_set_bit(size_t n, bool set_it)
   {
   const size_t which = n / BOTAN_MP_WORD_BITS;
   const word mask = static_cast<word>(set_it) << (n % BOTAN_MP_WORD_BITS);
   m_data.set_word_at(which, word_at(which) | mask);
   }

/*
* Clear bit number n
*/
void BigInt::clear_bit(size_t n)
   {
   const size_t which = n / BOTAN_MP_WORD_BITS;

   if(which < size())
      {
      const word mask = ~(static_cast<word>(1) << (n % BOTAN_MP_WORD_BITS));
      m_data.set_word_at(which, word_at(which) & mask);
      }
   }

size_t BigInt::bytes() const
   {
   return round_up(bits(), 8) / 8;
   }

size_t BigInt::top_bits_free() const
   {
   const size_t words = sig_words();

   const word top_word = word_at(words - 1);
   const size_t bits_used = high_bit(top_word);
   CT::unpoison(bits_used);
   return BOTAN_MP_WORD_BITS - bits_used;
   }

size_t BigInt::bits() const
   {
   const size_t words = sig_words();

   if(words == 0)
      return 0;

   const size_t full_words = (words - 1) * BOTAN_MP_WORD_BITS;
   const size_t top_bits = BOTAN_MP_WORD_BITS - top_bits_free();

   return full_words + top_bits;
   }

/*
* Calcluate the size in a certain base
*/
size_t BigInt::encoded_size(Base base) const
   {
   static const double LOG_2_BASE_10 = 0.30102999566;

   if(base == Binary)
      return bytes();
   else if(base == Hexadecimal)
      return 2*bytes();
   else if(base == Decimal)
      return static_cast<size_t>((bits() * LOG_2_BASE_10) + 1);
   else
      throw Invalid_Argument("Unknown base for BigInt encoding");
   }

/*
* Return the negation of this number
*/
BigInt BigInt::operator-() const
   {
   BigInt x = (*this);
   x.flip_sign();
   return x;
   }

size_t BigInt::reduce_below(const BigInt& p, secure_vector<word>& ws)
   {
   if(p.is_negative() || this->is_negative())
      throw Invalid_Argument("BigInt::reduce_below both values must be positive");

   const size_t p_words = p.sig_words();

   if(size() < p_words + 1)
      grow_to(p_words + 1);

   if(ws.size() < p_words + 1)
      ws.resize(p_words + 1);

   clear_mem(ws.data(), ws.size());

   size_t reductions = 0;

   for(;;)
      {
      word borrow = bigint_sub3(ws.data(), data(), p_words + 1, p.data(), p_words);
      if(borrow)
         break;

      ++reductions;
      swap_reg(ws);
      }

   return reductions;
   }

void BigInt::ct_reduce_below(const BigInt& mod, secure_vector<word>& ws, size_t bound)
   {
   if(mod.is_negative() || this->is_negative())
      throw Invalid_Argument("BigInt::ct_reduce_below both values must be positive");

   const size_t mod_words = mod.sig_words();

   grow_to(mod_words);

   const size_t sz = size();

   ws.resize(sz);

   clear_mem(ws.data(), sz);

   for(size_t i = 0; i != bound; ++i)
      {
      word borrow = bigint_sub3(ws.data(), data(), sz, mod.data(), mod_words);

      CT::Mask<word>::is_zero(borrow).select_n(mutable_data(), ws.data(), data(), sz);
      }
   }

/*
* Return the absolute value of this number
*/
BigInt BigInt::abs() const
   {
   BigInt x = (*this);
   x.set_sign(Positive);
   return x;
   }

void BigInt::binary_encode(uint8_t buf[]) const
   {
   this->binary_encode(buf, bytes());
   }

/*
* Encode this number into bytes
*/
void BigInt::binary_encode(uint8_t output[], size_t len) const
   {
   const size_t full_words = len / sizeof(word);
   const size_t extra_bytes = len % sizeof(word);

   for(size_t i = 0; i != full_words; ++i)
      {
      const word w = word_at(i);
      store_be(w, output + (len - (i+1)*sizeof(word)));
      }

   if(extra_bytes > 0)
      {
      const word w = word_at(full_words);

      for(size_t i = 0; i != extra_bytes; ++i)
         {
         output[extra_bytes - i - 1] = get_byte(sizeof(word) - i - 1, w);
         }
      }
   }

/*
* Set this number to the value in buf
*/
void BigInt::binary_decode(const uint8_t buf[], size_t length)
   {
   clear();

   const size_t full_words = length / sizeof(word);
   const size_t extra_bytes = length % sizeof(word);

   secure_vector<word> reg((round_up(full_words + (extra_bytes > 0 ? 1 : 0), 8)));

   for(size_t i = 0; i != full_words; ++i)
      {
      reg[i] = load_be<word>(buf + length - sizeof(word)*(i+1), 0);
      }

   if(extra_bytes > 0)
      {
      for(size_t i = 0; i != extra_bytes; ++i)
         reg[full_words] = (reg[full_words] << 8) | buf[i];
      }

   m_data.swap(reg);
   }

void BigInt::ct_cond_add(bool predicate, const BigInt& value)
   {
   if(this->is_negative() || value.is_negative())
      throw Invalid_Argument("BigInt::ct_cond_add requires both values to be positive");
   this->grow_to(1 + value.sig_words());

   bigint_cnd_add(static_cast<word>(predicate),
                  this->mutable_data(), this->size(),
                  value.data(), value.sig_words());
   }

void BigInt::ct_cond_swap(bool predicate, BigInt& other)
   {
   const size_t max_words = std::max(size(), other.size());
   grow_to(max_words);
   other.grow_to(max_words);

   bigint_cnd_swap(predicate, this->mutable_data(), other.mutable_data(), max_words);
   }

void BigInt::cond_flip_sign(bool predicate)
   {
   // This code is assuming Negative == 0, Positive == 1

   const auto mask = CT::Mask<uint8_t>::expand(predicate);

   const uint8_t current_sign = static_cast<uint8_t>(sign());

   const uint8_t new_sign = mask.select(current_sign ^ 1, current_sign);

   set_sign(static_cast<Sign>(new_sign));
   }

void BigInt::ct_cond_assign(bool predicate, const BigInt& other)
   {
   const size_t t_words = size();
   const size_t o_words = other.size();

   if(o_words < t_words)
      grow_to(o_words);

   const size_t r_words = std::max(t_words, o_words);

   const auto mask = CT::Mask<word>::expand(predicate);

   for(size_t i = 0; i != r_words; ++i)
      {
      const word o_word = other.word_at(i);
      const word t_word = this->word_at(i);
      this->set_word_at(i, mask.select(o_word, t_word));
      }

   const bool different_sign = sign() != other.sign();
   cond_flip_sign(predicate && different_sign);
   }

#if defined(BOTAN_HAS_VALGRIND)
void BigInt::const_time_poison() const
   {
   CT::poison(m_data.const_data(), m_data.size());
   }

void BigInt::const_time_unpoison() const
   {
   CT::unpoison(m_data.const_data(), m_data.size());
   }
#endif

void BigInt::const_time_lookup(secure_vector<word>& output,
                               const std::vector<BigInt>& vec,
                               size_t idx)
   {
   const size_t words = output.size();

   clear_mem(output.data(), output.size());

   CT::poison(&idx, sizeof(idx));

   for(size_t i = 0; i != vec.size(); ++i)
      {
      BOTAN_ASSERT(vec[i].size() >= words,
                   "Word size as expected in const_time_lookup");

      const auto mask = CT::Mask<word>::is_equal(i, idx);

      for(size_t w = 0; w != words; ++w)
         {
         const word viw = vec[i].word_at(w);
         output[w] = mask.if_set_return(viw);
         }
      }

   CT::unpoison(idx);
   CT::unpoison(output.data(), output.size());
   }

}
/*
* Division Algorithm
* (C) 1999-2007,2012,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

/*
* Handle signed operands, if necessary
*/
void sign_fixup(const BigInt& x, const BigInt& y, BigInt& q, BigInt& r)
   {
   q.cond_flip_sign(x.sign() != y.sign());

   if(x.is_negative() && r.is_nonzero())
      {
      q -= 1;
      r = y.abs() - r;
      }
   }

inline bool division_check(word q, word y2, word y1,
                           word x3, word x2, word x1)
   {
   /*
   Compute (y3,y2,y1) = (y2,y1) * q
   and return true if (y3,y2,y1) > (x3,x2,x1)
   */

   word y3 = 0;
   y1 = word_madd2(q, y1, &y3);
   y2 = word_madd2(q, y2, &y3);

   const word x[3] = { x1, x2, x3 };
   const word y[3] = { y1, y2, y3 };

   return bigint_ct_is_lt(x, 3, y, 3).is_set();
   }

}

void ct_divide(const BigInt& x, const BigInt& y, BigInt& q_out, BigInt& r_out)
   {
   const size_t x_words = x.sig_words();
   const size_t y_words = y.sig_words();

   const size_t x_bits = x.bits();

   BigInt q(BigInt::Positive, x_words);
   BigInt r(BigInt::Positive, y_words);
   BigInt t(BigInt::Positive, y_words); // a temporary

   for(size_t i = 0; i != x_bits; ++i)
      {
      const size_t b = x_bits - 1 - i;
      const bool x_b = x.get_bit(b);

      r *= 2;
      r.conditionally_set_bit(0, x_b);

      const bool r_gte_y = bigint_sub3(t.mutable_data(), r.data(), r.size(), y.data(), y_words) == 0;

      q.conditionally_set_bit(b, r_gte_y);
      r.ct_cond_swap(r_gte_y, t);
      }

   sign_fixup(x, y, q, r);
   r_out = r;
   q_out = q;
   }

void ct_divide_u8(const BigInt& x, uint8_t y, BigInt& q_out, uint8_t& r_out)
   {
   const size_t x_words = x.sig_words();
   const size_t x_bits = x.bits();

   BigInt q(BigInt::Positive, x_words);
   uint32_t r = 0;

   for(size_t i = 0; i != x_bits; ++i)
      {
      const size_t b = x_bits - 1 - i;
      const bool x_b = x.get_bit(b);

      r *= 2;
      r += x_b;

      const auto r_gte_y = CT::Mask<uint32_t>::is_gte(r, y);

      q.conditionally_set_bit(b, r_gte_y.is_set());
      r = r_gte_y.select(r - y, r);
      }

   if(x.is_negative())
      {
      q.flip_sign();
      if(r != 0)
         {
         --q;
         r = y - r;
         }
      }

   r_out = static_cast<uint8_t>(r);
   q_out = q;
   }

BigInt ct_modulo(const BigInt& x, const BigInt& y)
   {
   if(y.is_negative() || y.is_zero())
      throw Invalid_Argument("ct_modulo requires y > 0");

   const size_t y_words = y.sig_words();

   const size_t x_bits = x.bits();

   BigInt r(BigInt::Positive, y_words);
   BigInt t(BigInt::Positive, y_words);

   for(size_t i = 0; i != x_bits; ++i)
      {
      const size_t b = x_bits - 1 - i;
      const bool x_b = x.get_bit(b);

      r *= 2;
      r.conditionally_set_bit(0, x_b);

      const bool r_gte_y = bigint_sub3(t.mutable_data(), r.data(), r.size(), y.data(), y_words) == 0;

      r.ct_cond_swap(r_gte_y, t);
      }

   if(x.is_negative())
      {
      if(r.is_nonzero())
         {
         r = y - r;
         }
      }

   return r;
   }

/*
* Solve x = q * y + r
*
* See Handbook of Applied Cryptography section 14.2.5
*/
void vartime_divide(const BigInt& x, const BigInt& y_arg, BigInt& q_out, BigInt& r_out)
   {
   if(y_arg.is_zero())
      throw BigInt::DivideByZero();

   const size_t y_words = y_arg.sig_words();

   BOTAN_ASSERT_NOMSG(y_words > 0);

   BigInt y = y_arg;

   BigInt r = x;
   BigInt q = 0;
   secure_vector<word> ws;

   r.set_sign(BigInt::Positive);
   y.set_sign(BigInt::Positive);

   // Calculate shifts needed to normalize y with high bit set
   const size_t shifts = y.top_bits_free();

   y <<= shifts;
   r <<= shifts;

   // we know y has not changed size, since we only shifted up to set high bit
   const size_t t = y_words - 1;
   const size_t n = std::max(y_words, r.sig_words()) - 1; // r may have changed size however

   BOTAN_ASSERT_NOMSG(n >= t);

   q.grow_to(n - t + 1);

   word* q_words = q.mutable_data();

   BigInt shifted_y = y << (BOTAN_MP_WORD_BITS * (n-t));

   // Set q_{n-t} to number of times r > shifted_y
   q_words[n-t] = r.reduce_below(shifted_y, ws);

   const word y_t0  = y.word_at(t);
   const word y_t1  = y.word_at(t-1);
   BOTAN_DEBUG_ASSERT((y_t0 >> (BOTAN_MP_WORD_BITS-1)) == 1);

   for(size_t j = n; j != t; --j)
      {
      const word x_j0  = r.word_at(j);
      const word x_j1 = r.word_at(j-1);
      const word x_j2 = r.word_at(j-2);

      word qjt = bigint_divop(x_j0, x_j1, y_t0);

      qjt = CT::Mask<word>::is_equal(x_j0, y_t0).select(MP_WORD_MAX, qjt);

      // Per HAC 14.23, this operation is required at most twice
      qjt -= division_check(qjt, y_t0, y_t1, x_j0, x_j1, x_j2);
      qjt -= division_check(qjt, y_t0, y_t1, x_j0, x_j1, x_j2);
      BOTAN_DEBUG_ASSERT(division_check(qjt, y_t0, y_t1, x_j0, x_j1, x_j2) == false);

      shifted_y >>= BOTAN_MP_WORD_BITS;
      // Now shifted_y == y << (BOTAN_MP_WORD_BITS * (j-t-1))

      // TODO this sequence could be better
      r -= qjt * shifted_y;
      qjt -= r.is_negative();
      r += static_cast<word>(r.is_negative()) * shifted_y;

      q_words[j-t-1] = qjt;
      }

   r >>= shifts;

   sign_fixup(x, y_arg, q, r);

   r_out = r;
   q_out = q;
   }

}
/*
* Block Ciphers
* (C) 2015 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_AES)
#endif

#if defined(BOTAN_HAS_ARIA)
#endif

#if defined(BOTAN_HAS_BLOWFISH)
#endif

#if defined(BOTAN_HAS_CAMELLIA)
#endif

#if defined(BOTAN_HAS_CAST_128)
#endif

#if defined(BOTAN_HAS_CAST_256)
#endif

#if defined(BOTAN_HAS_CASCADE)
#endif

#if defined(BOTAN_HAS_DES)
#endif

#if defined(BOTAN_HAS_GOST_28147_89)
#endif

#if defined(BOTAN_HAS_IDEA)
#endif

#if defined(BOTAN_HAS_KASUMI)
#endif

#if defined(BOTAN_HAS_LION)
#endif

#if defined(BOTAN_HAS_MISTY1)
#endif

#if defined(BOTAN_HAS_NOEKEON)
#endif

#if defined(BOTAN_HAS_SEED)
#endif

#if defined(BOTAN_HAS_SERPENT)
#endif

#if defined(BOTAN_HAS_SHACAL2)
#endif

#if defined(BOTAN_HAS_SM4)
#endif

#if defined(BOTAN_HAS_TWOFISH)
#endif

#if defined(BOTAN_HAS_THREEFISH_512)
#endif

#if defined(BOTAN_HAS_XTEA)
#endif

#if defined(BOTAN_HAS_OPENSSL)
#endif

#if defined(BOTAN_HAS_COMMONCRYPTO)
#endif

namespace Botan {


/*
* RC5 Encryption
*/
void RC5::encrypt_n(const byte in[], byte out[], size_t blocks) const
   {
   for(size_t i = 0; i != blocks; ++i)
      {
      u32bit A = load_le<u32bit>(in, 0);
      u32bit B = load_le<u32bit>(in, 1);

      A += m_S[0]; B += m_S[1];
      for(size_t j = 0; j != m_rounds; j += 4)
         {
         A = rotate_left(A ^ B, B % 32) + m_S[2*j+2];
         B = rotate_left(B ^ A, A % 32) + m_S[2*j+3];

         A = rotate_left(A ^ B, B % 32) + m_S[2*j+4];
         B = rotate_left(B ^ A, A % 32) + m_S[2*j+5];

         A = rotate_left(A ^ B, B % 32) + m_S[2*j+6];
         B = rotate_left(B ^ A, A % 32) + m_S[2*j+7];

         A = rotate_left(A ^ B, B % 32) + m_S[2*j+8];
         B = rotate_left(B ^ A, A % 32) + m_S[2*j+9];
         }

      store_le(out, A, B);

      in += BLOCK_SIZE;
      out += BLOCK_SIZE;
      }
   }

/*
* RC5 Decryption
*/
void RC5::decrypt_n(const byte in[], byte out[], size_t blocks) const
   {
   for(size_t i = 0; i != blocks; ++i)
      {
      u32bit A = load_le<u32bit>(in, 0);
      u32bit B = load_le<u32bit>(in, 1);

      for(size_t j = m_rounds; j != 0; j -= 4)
         {
         B = rotate_right(B - m_S[2*j+1], A % 32) ^ A;
         A = rotate_right(A - m_S[2*j  ], B % 32) ^ B;

         B = rotate_right(B - m_S[2*j-1], A % 32) ^ A;
         A = rotate_right(A - m_S[2*j-2], B % 32) ^ B;

         B = rotate_right(B - m_S[2*j-3], A % 32) ^ A;
         A = rotate_right(A - m_S[2*j-4], B % 32) ^ B;

         B = rotate_right(B - m_S[2*j-5], A % 32) ^ A;
         A = rotate_right(A - m_S[2*j-6], B % 32) ^ B;
         }
      B -= m_S[1]; A -= m_S[0];

      store_le(out, A, B);

      in += BLOCK_SIZE;
      out += BLOCK_SIZE;
      }
   }

/*
* RC5 Key Schedule
*/
void RC5::key_schedule(const byte key[], size_t length)
   {
   m_S.resize(2*m_rounds + 2);

   const size_t WORD_KEYLENGTH = (((length - 1) / 4) + 1);
   const size_t MIX_ROUNDS     = 3 * std::max(WORD_KEYLENGTH, m_S.size());

   m_S[0] = 0xB7E15163;
   for(size_t i = 1; i != m_S.size(); ++i)
      m_S[i] = m_S[i-1] + 0x9E3779B9;

   secure_vector<u32bit> K(8);

   for(s32bit i = length-1; i >= 0; --i)
      K[i/4] = (K[i/4] << 8) + key[i];

   u32bit A = 0, B = 0;

   for(size_t i = 0; i != MIX_ROUNDS; ++i)
      {
      A = rotate_left(m_S[i % m_S.size()] + A + B, 3);
      B = rotate_left(K[i % WORD_KEYLENGTH] + A + B, (A + B) % 32);
      m_S[i % m_S.size()] = A;
      K[i % WORD_KEYLENGTH] = B;
      }
   }

void RC5::clear()
   {
   zap(m_S);
   }

/*
* Return the name of this type
*/
std::string RC5::name() const
   {
   return "RC5(" + std::to_string(m_rounds) + ")";
   }

/*
* RC5 Constructor
*/
RC5::RC5(size_t r) : m_rounds(r)
   {
   if(m_rounds < 8 || m_rounds > 32 || (m_rounds % 4 != 0))
      throw Invalid_Argument("RC5: Invalid number of rounds " +
                             std::to_string(m_rounds));
   }

}


namespace Botan {

std::unique_ptr<BlockCipher>
BlockCipher::create(const std::string& algo,
                    const std::string& provider)
   {
   if(algo == "RC5")
      {
      return std::unique_ptr<BlockCipher>(new RC5(16));
      }

#if defined(BOTAN_HAS_COMMONCRYPTO)
   if(provider.empty() || provider == "commoncrypto")
      {
      if(auto bc = make_commoncrypto_block_cipher(algo))
         return bc;

      if(!provider.empty())
         return nullptr;
      }
#endif

#if defined(BOTAN_HAS_OPENSSL)
   if(provider.empty() || provider == "openssl")
      {
      if(auto bc = make_openssl_block_cipher(algo))
         return bc;

      if(!provider.empty())
         return nullptr;
      }
#endif

   // TODO: CryptoAPI
   // TODO: /dev/crypto

   // Only base providers from here on out
   if(provider.empty() == false && provider != "base")
      return nullptr;

#if defined(BOTAN_HAS_AES)
   if(algo == "AES-128")
      {
      return std::unique_ptr<BlockCipher>(new AES_128);
      }

   if(algo == "AES-192")
      {
      return std::unique_ptr<BlockCipher>(new AES_192);
      }

   if(algo == "AES-256")
      {
      return std::unique_ptr<BlockCipher>(new AES_256);
      }
#endif

#if defined(BOTAN_HAS_ARIA)
   if(algo == "ARIA-128")
      {
      return std::unique_ptr<BlockCipher>(new ARIA_128);
      }

   if(algo == "ARIA-192")
      {
      return std::unique_ptr<BlockCipher>(new ARIA_192);
      }

   if(algo == "ARIA-256")
      {
      return std::unique_ptr<BlockCipher>(new ARIA_256);
      }
#endif

#if defined(BOTAN_HAS_SERPENT)
   if(algo == "Serpent")
      {
      return std::unique_ptr<BlockCipher>(new Serpent);
      }
#endif

#if defined(BOTAN_HAS_SHACAL2)
   if(algo == "SHACAL2")
      {
      return std::unique_ptr<BlockCipher>(new SHACAL2);
      }
#endif

#if defined(BOTAN_HAS_TWOFISH)
   if(algo == "Twofish")
      {
      return std::unique_ptr<BlockCipher>(new Twofish);
      }
#endif

#if defined(BOTAN_HAS_THREEFISH_512)
   if(algo == "Threefish-512")
      {
      return std::unique_ptr<BlockCipher>(new Threefish_512);
      }
#endif

#if defined(BOTAN_HAS_BLOWFISH)
   if(algo == "Blowfish")
      {
      return std::unique_ptr<BlockCipher>(new Blowfish);
      }
#endif

#if defined(BOTAN_HAS_CAMELLIA)
   if(algo == "Camellia-128")
      {
      return std::unique_ptr<BlockCipher>(new Camellia_128);
      }

   if(algo == "Camellia-192")
      {
      return std::unique_ptr<BlockCipher>(new Camellia_192);
      }

   if(algo == "Camellia-256")
      {
      return std::unique_ptr<BlockCipher>(new Camellia_256);
      }
#endif

#if defined(BOTAN_HAS_DES)
   if(algo == "DES")
      {
      return std::unique_ptr<BlockCipher>(new DES);
      }

   if(algo == "DESX")
      {
      return std::unique_ptr<BlockCipher>(new DESX);
      }

   if(algo == "TripleDES" || algo == "3DES" || algo == "DES-EDE")
      {
      return std::unique_ptr<BlockCipher>(new TripleDES);
      }
#endif

#if defined(BOTAN_HAS_NOEKEON)
   if(algo == "Noekeon")
      {
      return std::unique_ptr<BlockCipher>(new Noekeon);
      }
#endif

#if defined(BOTAN_HAS_CAST_128)
   if(algo == "CAST-128" || algo == "CAST5")
      {
      return std::unique_ptr<BlockCipher>(new CAST_128);
      }
#endif

#if defined(BOTAN_HAS_CAST_256)
   if(algo == "CAST-256")
      {
      return std::unique_ptr<BlockCipher>(new CAST_256);
      }
#endif

#if defined(BOTAN_HAS_IDEA)
   if(algo == "IDEA")
      {
      return std::unique_ptr<BlockCipher>(new IDEA);
      }
#endif

#if defined(BOTAN_HAS_KASUMI)
   if(algo == "KASUMI")
      {
      return std::unique_ptr<BlockCipher>(new KASUMI);
      }
#endif

#if defined(BOTAN_HAS_MISTY1)
   if(algo == "MISTY1")
      {
      return std::unique_ptr<BlockCipher>(new MISTY1);
      }
#endif

#if defined(BOTAN_HAS_SEED)
   if(algo == "SEED")
      {
      return std::unique_ptr<BlockCipher>(new SEED);
      }
#endif

#if defined(BOTAN_HAS_SM4)
   if(algo == "SM4")
      {
      return std::unique_ptr<BlockCipher>(new SM4);
      }
#endif

#if defined(BOTAN_HAS_XTEA)
   if(algo == "XTEA")
      {
      return std::unique_ptr<BlockCipher>(new XTEA);
      }
#endif

   const SCAN_Name req(algo);

#if defined(BOTAN_HAS_GOST_28147_89)
   if(req.algo_name() == "GOST-28147-89")
      {
      return std::unique_ptr<BlockCipher>(new GOST_28147_89(req.arg(0, "R3411_94_TestParam")));
      }
#endif

#if defined(BOTAN_HAS_CASCADE)
   if(req.algo_name() == "Cascade" && req.arg_count() == 2)
      {
      std::unique_ptr<BlockCipher> c1(BlockCipher::create(req.arg(0)));
      std::unique_ptr<BlockCipher> c2(BlockCipher::create(req.arg(1)));

      if(c1 && c2)
         return std::unique_ptr<BlockCipher>(new Cascade_Cipher(c1.release(), c2.release()));
      }
#endif

#if defined(BOTAN_HAS_LION)
   if(req.algo_name() == "Lion" && req.arg_count_between(2, 3))
      {
      std::unique_ptr<HashFunction> hash(HashFunction::create(req.arg(0)));
      std::unique_ptr<StreamCipher> stream(StreamCipher::create(req.arg(1)));

      if(hash && stream)
         {
         const size_t block_size = req.arg_as_integer(2, 1024);
         return std::unique_ptr<BlockCipher>(new Lion(hash.release(), stream.release(), block_size));
         }
      }
#endif

   BOTAN_UNUSED(req);
   BOTAN_UNUSED(provider);

   return nullptr;
   }

//static
std::unique_ptr<BlockCipher>
BlockCipher::create_or_throw(const std::string& algo,
                             const std::string& provider)
   {
   if(auto bc = BlockCipher::create(algo, provider))
      {
      return bc;
      }
   throw Lookup_Error("Block cipher", algo, provider);
   }

std::vector<std::string> BlockCipher::providers(const std::string& algo)
   {
   return probe_providers_of<BlockCipher>(algo, { "base", "openssl", "commoncrypto" });
   }

}
/*
* CBC Mode
* (C) 1999-2007,2013,2017 Jack Lloyd
* (C) 2016 Daniel Neus, Rohde & Schwarz Cybersecurity
* (C) 2018 Ribose Inc
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

CBC_Mode::CBC_Mode(BlockCipher* cipher, BlockCipherModePaddingMethod* padding) :
   m_cipher(cipher),
   m_padding(padding),
   m_block_size(cipher->block_size())
   {
   if(m_padding && !m_padding->valid_blocksize(m_block_size))
      throw Invalid_Argument("Padding " + m_padding->name() +
                             " cannot be used with " +
                             cipher->name() + "/CBC");
   }

void CBC_Mode::clear()
   {
   m_cipher->clear();
   reset();
   }

void CBC_Mode::reset()
   {
   m_state.clear();
   }

std::string CBC_Mode::name() const
   {
   if(m_padding)
      return cipher().name() + "/CBC/" + padding().name();
   else
      return cipher().name() + "/CBC/CTS";
   }

size_t CBC_Mode::update_granularity() const
   {
   return cipher().parallel_bytes();
   }

Key_Length_Specification CBC_Mode::key_spec() const
   {
   return cipher().key_spec();
   }

size_t CBC_Mode::default_nonce_length() const
   {
   return block_size();
   }

bool CBC_Mode::valid_nonce_length(size_t n) const
   {
   return (n == 0 || n == block_size());
   }

void CBC_Mode::key_schedule(const uint8_t key[], size_t length)
   {
   m_cipher->set_key(key, length);
   m_state.clear();
   }

void CBC_Mode::start_msg(const uint8_t nonce[], size_t nonce_len)
   {
   if(!valid_nonce_length(nonce_len))
      throw Invalid_IV_Length(name(), nonce_len);

   /*
   * A nonce of zero length means carry the last ciphertext value over
   * as the new IV, as unfortunately some protocols require this. If
   * this is the first message then we use an IV of all zeros.
   */
   if(nonce_len)
      m_state.assign(nonce, nonce + nonce_len);
   else if(m_state.empty())
      m_state.resize(m_cipher->block_size());
   // else leave the state alone
   }

size_t CBC_Encryption::minimum_final_size() const
   {
   return 0;
   }

size_t CBC_Encryption::output_length(size_t input_length) const
   {
   if(input_length == 0)
      return block_size();
   else
      return round_up(input_length, block_size());
   }

size_t CBC_Encryption::process(uint8_t buf[], size_t sz)
   {
   BOTAN_STATE_CHECK(state().empty() == false);
   const size_t BS = block_size();

   BOTAN_ASSERT(sz % BS == 0, "CBC input is full blocks");
   const size_t blocks = sz / BS;

   if(blocks > 0)
      {
      xor_buf(&buf[0], state_ptr(), BS);
      cipher().encrypt(&buf[0]);

      for(size_t i = 1; i != blocks; ++i)
         {
         xor_buf(&buf[BS*i], &buf[BS*(i-1)], BS);
         cipher().encrypt(&buf[BS*i]);
         }

      state().assign(&buf[BS*(blocks-1)], &buf[BS*blocks]);
      }

   return sz;
   }

void CBC_Encryption::finish(secure_vector<uint8_t>& buffer, size_t offset)
   {
   BOTAN_STATE_CHECK(state().empty() == false);
   BOTAN_ASSERT(buffer.size() >= offset, "Offset is sane");

   const size_t BS = block_size();

   const size_t bytes_in_final_block = (buffer.size()-offset) % BS;

   padding().add_padding(buffer, bytes_in_final_block, BS);

   BOTAN_ASSERT_EQUAL(buffer.size() % BS, offset % BS, "Padded to block boundary");

   update(buffer, offset);
   }

bool CTS_Encryption::valid_nonce_length(size_t n) const
   {
   return (n == block_size());
   }

size_t CTS_Encryption::minimum_final_size() const
   {
   return block_size() + 1;
   }

size_t CTS_Encryption::output_length(size_t input_length) const
   {
   return input_length; // no ciphertext expansion in CTS
   }

void CTS_Encryption::finish(secure_vector<uint8_t>& buffer, size_t offset)
   {
   BOTAN_STATE_CHECK(state().empty() == false);
   BOTAN_ASSERT(buffer.size() >= offset, "Offset is sane");
   uint8_t* buf = buffer.data() + offset;
   const size_t sz = buffer.size() - offset;

   const size_t BS = block_size();

   if(sz < BS + 1)
      throw Encoding_Error(name() + ": insufficient data to encrypt");

   if(sz % BS == 0)
      {
      update(buffer, offset);

      // swap last two blocks
      for(size_t i = 0; i != BS; ++i)
         std::swap(buffer[buffer.size()-BS+i], buffer[buffer.size()-2*BS+i]);
      }
   else
      {
      const size_t full_blocks = ((sz / BS) - 1) * BS;
      const size_t final_bytes = sz - full_blocks;
      BOTAN_ASSERT(final_bytes > BS && final_bytes < 2*BS, "Left over size in expected range");

      secure_vector<uint8_t> last(buf + full_blocks, buf + full_blocks + final_bytes);
      buffer.resize(full_blocks + offset);
      update(buffer, offset);

      xor_buf(last.data(), state_ptr(), BS);
      cipher().encrypt(last.data());

      for(size_t i = 0; i != final_bytes - BS; ++i)
         {
         last[i] ^= last[i + BS];
         last[i + BS] ^= last[i];
         }

      cipher().encrypt(last.data());

      buffer += last;
      }
   }

size_t CBC_Decryption::output_length(size_t input_length) const
   {
   return input_length; // precise for CTS, worst case otherwise
   }

size_t CBC_Decryption::minimum_final_size() const
   {
   return block_size();
   }

size_t CBC_Decryption::process(uint8_t buf[], size_t sz)
   {
   BOTAN_STATE_CHECK(state().empty() == false);

   const size_t BS = block_size();

   BOTAN_ASSERT(sz % BS == 0, "Input is full blocks");
   size_t blocks = sz / BS;

   while(blocks)
      {
      const size_t to_proc = std::min(BS * blocks, m_tempbuf.size());

      cipher().decrypt_n(buf, m_tempbuf.data(), to_proc / BS);

      xor_buf(m_tempbuf.data(), state_ptr(), BS);
      xor_buf(&m_tempbuf[BS], buf, to_proc - BS);
      copy_mem(state_ptr(), buf + (to_proc - BS), BS);

      copy_mem(buf, m_tempbuf.data(), to_proc);

      buf += to_proc;
      blocks -= to_proc / BS;
      }

   return sz;
   }

void CBC_Decryption::finish(secure_vector<uint8_t>& buffer, size_t offset)
   {
   BOTAN_STATE_CHECK(state().empty() == false);
   BOTAN_ASSERT(buffer.size() >= offset, "Offset is sane");
   const size_t sz = buffer.size() - offset;

   const size_t BS = block_size();

   if(sz == 0 || sz % BS)
      throw Decoding_Error(name() + ": Ciphertext not a multiple of block size");

   update(buffer, offset);

   const size_t pad_bytes = BS - padding().unpad(&buffer[buffer.size()-BS], BS);
   buffer.resize(buffer.size() - pad_bytes); // remove padding
   if(pad_bytes == 0 && padding().name() != "NoPadding")
      {
      throw Decoding_Error("Invalid CBC padding");
      }
   }

void CBC_Decryption::reset()
   {
   CBC_Mode::reset();
   zeroise(m_tempbuf);
   }

bool CTS_Decryption::valid_nonce_length(size_t n) const
   {
   return (n == block_size());
   }

size_t CTS_Decryption::minimum_final_size() const
   {
   return block_size() + 1;
   }

void CTS_Decryption::finish(secure_vector<uint8_t>& buffer, size_t offset)
   {
   BOTAN_STATE_CHECK(state().empty() == false);
   BOTAN_ASSERT(buffer.size() >= offset, "Offset is sane");
   const size_t sz = buffer.size() - offset;
   uint8_t* buf = buffer.data() + offset;

   const size_t BS = block_size();

   if(sz < BS + 1)
      throw Encoding_Error(name() + ": insufficient data to decrypt");

   if(sz % BS == 0)
      {
      // swap last two blocks

      for(size_t i = 0; i != BS; ++i)
         std::swap(buffer[buffer.size()-BS+i], buffer[buffer.size()-2*BS+i]);

      update(buffer, offset);
      }
   else
      {
      const size_t full_blocks = ((sz / BS) - 1) * BS;
      const size_t final_bytes = sz - full_blocks;
      BOTAN_ASSERT(final_bytes > BS && final_bytes < 2*BS, "Left over size in expected range");

      secure_vector<uint8_t> last(buf + full_blocks, buf + full_blocks + final_bytes);
      buffer.resize(full_blocks + offset);
      update(buffer, offset);

      cipher().decrypt(last.data());

      xor_buf(last.data(), &last[BS], final_bytes - BS);

      for(size_t i = 0; i != final_bytes - BS; ++i)
         std::swap(last[i], last[i + BS]);

      cipher().decrypt(last.data());
      xor_buf(last.data(), state_ptr(), BS);

      buffer += last;
      }
   }

}
/*
* Runtime CPU detection
* (C) 2009,2010,2013,2017 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

bool CPUID::has_simd_32()
   {
#if defined(BOTAN_TARGET_SUPPORTS_SSE2)
   return CPUID::has_sse2();
#elif defined(BOTAN_TARGET_SUPPORTS_ALTIVEC)
   return CPUID::has_altivec();
#elif defined(BOTAN_TARGET_SUPPORTS_NEON)
   return CPUID::has_neon();
#else
   return true;
#endif
   }

//static
std::string CPUID::to_string()
   {
   std::vector<std::string> flags;

#define CPUID_PRINT(flag) do { if(has_##flag()) { flags.push_back(#flag); } } while(0)

#if defined(BOTAN_TARGET_CPU_IS_X86_FAMILY)
   CPUID_PRINT(sse2);
   CPUID_PRINT(ssse3);
   CPUID_PRINT(sse41);
   CPUID_PRINT(sse42);
   CPUID_PRINT(avx2);
   CPUID_PRINT(avx512f);
   CPUID_PRINT(avx512dq);
   CPUID_PRINT(avx512bw);
   CPUID_PRINT(avx512_icelake);

   CPUID_PRINT(rdtsc);
   CPUID_PRINT(bmi1);
   CPUID_PRINT(bmi2);
   CPUID_PRINT(adx);

   CPUID_PRINT(aes_ni);
   CPUID_PRINT(clmul);
   CPUID_PRINT(rdrand);
   CPUID_PRINT(rdseed);
   CPUID_PRINT(intel_sha);
   CPUID_PRINT(avx512_aes);
   CPUID_PRINT(avx512_clmul);
#endif

#if defined(BOTAN_TARGET_CPU_IS_PPC_FAMILY)
   CPUID_PRINT(altivec);
   CPUID_PRINT(power_crypto);
   CPUID_PRINT(darn_rng);
#endif

#if defined(BOTAN_TARGET_CPU_IS_ARM_FAMILY)
   CPUID_PRINT(neon);
   CPUID_PRINT(arm_sve);

   CPUID_PRINT(arm_sha1);
   CPUID_PRINT(arm_sha2);
   CPUID_PRINT(arm_aes);
   CPUID_PRINT(arm_pmull);
   CPUID_PRINT(arm_sha2_512);
   CPUID_PRINT(arm_sha3);
   CPUID_PRINT(arm_sm3);
   CPUID_PRINT(arm_sm4);
#endif

#undef CPUID_PRINT

   return string_join(flags, ' ');
   }

//static
void CPUID::print(std::ostream& o)
   {
   o << "CPUID flags: " << CPUID::to_string() << "\n";
   }

//static
void CPUID::initialize()
   {
   state() = CPUID_Data();
   }

CPUID::CPUID_Data::CPUID_Data()
   {
   m_cache_line_size = 0;
   m_processor_features = 0;

#if defined(BOTAN_TARGET_CPU_IS_PPC_FAMILY) || \
    defined(BOTAN_TARGET_CPU_IS_ARM_FAMILY) || \
    defined(BOTAN_TARGET_CPU_IS_X86_FAMILY)

   m_processor_features = detect_cpu_features(&m_cache_line_size);

#endif

   m_processor_features |= CPUID::CPUID_INITIALIZED_BIT;

   if(m_cache_line_size == 0)
      m_cache_line_size = BOTAN_TARGET_CPU_DEFAULT_CACHE_LINE_SIZE;

   m_endian_status = runtime_check_endian();
   }

//static
CPUID::Endian_Status CPUID::CPUID_Data::runtime_check_endian()
   {
   // Check runtime endian
   const uint32_t endian32 = 0x01234567;
   const uint8_t* e8 = reinterpret_cast<const uint8_t*>(&endian32);

   CPUID::Endian_Status endian = CPUID::Endian_Status::Unknown;

   if(e8[0] == 0x01 && e8[1] == 0x23 && e8[2] == 0x45 && e8[3] == 0x67)
      {
      endian = CPUID::Endian_Status::Big;
      }
   else if(e8[0] == 0x67 && e8[1] == 0x45 && e8[2] == 0x23 && e8[3] == 0x01)
      {
      endian = CPUID::Endian_Status::Little;
      }
   else
      {
      throw Internal_Error("Unexpected endian at runtime, neither big nor little");
      }

   // If we were compiled with a known endian, verify it matches at runtime
#if defined(BOTAN_TARGET_CPU_IS_LITTLE_ENDIAN)
   BOTAN_ASSERT(endian == CPUID::Endian_Status::Little, "Build and runtime endian match");
#elif defined(BOTAN_TARGET_CPU_IS_BIG_ENDIAN)
   BOTAN_ASSERT(endian == CPUID::Endian_Status::Big, "Build and runtime endian match");
#endif

   return endian;
   }

std::vector<Botan::CPUID::CPUID_bits>
CPUID::bit_from_string(const std::string& tok)
   {
#if defined(BOTAN_TARGET_CPU_IS_X86_FAMILY)
   if(tok == "sse2" || tok == "simd")
      return {Botan::CPUID::CPUID_SSE2_BIT};
   if(tok == "ssse3")
      return {Botan::CPUID::CPUID_SSSE3_BIT};
   if(tok == "sse41")
      return {Botan::CPUID::CPUID_SSE41_BIT};
   if(tok == "sse42")
      return {Botan::CPUID::CPUID_SSE42_BIT};
   // aes_ni is the string printed on the console when running "botan cpuid"
   if(tok == "aesni" || tok == "aes_ni")
      return {Botan::CPUID::CPUID_AESNI_BIT};
   if(tok == "clmul")
      return {Botan::CPUID::CPUID_CLMUL_BIT};
   if(tok == "avx2")
      return {Botan::CPUID::CPUID_AVX2_BIT};
   if(tok == "avx512f")
      return {Botan::CPUID::CPUID_AVX512F_BIT};
   if(tok == "avx512_icelake")
      return {Botan::CPUID::CPUID_AVX512_ICL_BIT};
   // there were two if statements testing "sha" and "intel_sha" separately; combined
   if(tok == "sha" || tok=="intel_sha")
      return {Botan::CPUID::CPUID_SHA_BIT};
   if(tok == "rdtsc")
      return {Botan::CPUID::CPUID_RDTSC_BIT};
   if(tok == "bmi1")
      return {Botan::CPUID::CPUID_BMI1_BIT};
   if(tok == "bmi2")
      return {Botan::CPUID::CPUID_BMI2_BIT};
   if(tok == "adx")
      return {Botan::CPUID::CPUID_ADX_BIT};
   if(tok == "rdrand")
      return {Botan::CPUID::CPUID_RDRAND_BIT};
   if(tok == "rdseed")
      return {Botan::CPUID::CPUID_RDSEED_BIT};
   if(tok == "avx512_aes")
      return {Botan::CPUID::CPUID_AVX512_AES_BIT};
   if(tok == "avx512_clmul")
      return {Botan::CPUID::CPUID_AVX512_CLMUL_BIT};

#elif defined(BOTAN_TARGET_CPU_IS_PPC_FAMILY)
   if(tok == "altivec" || tok == "simd")
      return {Botan::CPUID::CPUID_ALTIVEC_BIT};
   if(tok == "power_crypto")
      return {Botan::CPUID::CPUID_POWER_CRYPTO_BIT};
   if(tok == "darn_rng")
      return {Botan::CPUID::CPUID_DARN_BIT};

#elif defined(BOTAN_TARGET_CPU_IS_ARM_FAMILY)
   if(tok == "neon" || tok == "simd")
      return {Botan::CPUID::CPUID_ARM_NEON_BIT};
   if(tok == "arm_sve")
      return {Botan::CPUID::CPUID_ARM_SVE_BIT};
   if(tok == "armv8sha1" || tok == "arm_sha1")
      return {Botan::CPUID::CPUID_ARM_SHA1_BIT};
   if(tok == "armv8sha2" || tok == "arm_sha2")
      return {Botan::CPUID::CPUID_ARM_SHA2_BIT};
   if(tok == "armv8aes" || tok == "arm_aes")
      return {Botan::CPUID::CPUID_ARM_AES_BIT};
   if(tok == "armv8pmull" || tok == "arm_pmull")
      return {Botan::CPUID::CPUID_ARM_PMULL_BIT};
   if(tok == "armv8sha3" || tok == "arm_sha3")
      return {Botan::CPUID::CPUID_ARM_SHA3_BIT};
   if(tok == "armv8sha2_512" || tok == "arm_sha2_512")
      return {Botan::CPUID::CPUID_ARM_SHA2_512_BIT};
   if(tok == "armv8sm3" || tok == "arm_sm3")
      return {Botan::CPUID::CPUID_ARM_SM3_BIT};
   if(tok == "armv8sm4" || tok == "arm_sm4")
      return {Botan::CPUID::CPUID_ARM_SM4_BIT};

#else
   BOTAN_UNUSED(tok);
#endif

   return {};
   }

}
/*
* Runtime CPU detection for ARM
* (C) 2009,2010,2013,2017 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_TARGET_CPU_IS_ARM_FAMILY)

#if defined(BOTAN_TARGET_OS_IS_IOS)
  #include <sys/types.h>
  #include <sys/sysctl.h>

#else

#if defined(BOTAN_TARGET_OS_HAS_GETAUXVAL) || defined(BOTAN_TARGET_OS_HAS_ELF_AUX_INFO)
  #include <sys/auxv.h>
#endif

#endif

#endif

namespace Botan {

#if defined(BOTAN_TARGET_CPU_IS_ARM_FAMILY)

#if defined(BOTAN_TARGET_OS_IS_IOS)

namespace {

uint64_t flags_by_ios_machine_type(const std::string& machine)
   {
   /*
   * This relies on a map of known machine names to features. This
   * will quickly grow out of date as new products are introduced, but
   * is apparently the best we can do for iOS.
   */

   struct version_info {
      std::string name;
      size_t min_version_neon;
      size_t min_version_armv8;
      };

   static const version_info min_versions[] = {
      { "iPhone", 2, 6 },
      { "iPad", 1, 4 },
      { "iPod", 4, 7 },
      { "AppleTV", 2, 5 },
   };

   if(machine.size() < 3)
      return 0;

   auto comma = machine.find(',');

   // Simulator, or something we don't know about
   if(comma == std::string::npos)
      return 0;

   std::string product = machine.substr(0, comma);

   size_t version = 0;
   size_t place = 1;
   while(product.size() > 1 && ::isdigit(product.back()))
      {
      const size_t digit = product.back() - '0';
      version += digit * place;
      place *= 10;
      product.pop_back();
      }

   if(version == 0)
      return 0;

   for(const version_info& info : min_versions)
      {
      if(info.name != product)
         continue;

      if(version >= info.min_version_armv8)
         {
         return CPUID::CPUID_ARM_AES_BIT |
                CPUID::CPUID_ARM_PMULL_BIT |
                CPUID::CPUID_ARM_SHA1_BIT |
                CPUID::CPUID_ARM_SHA2_BIT |
                CPUID::CPUID_ARM_NEON_BIT;
         }

      if(version >= info.min_version_neon)
         return CPUID::CPUID_ARM_NEON_BIT;
      }

   // Some other product we don't know about
   return 0;
   }

}

#endif

uint64_t CPUID::CPUID_Data::detect_cpu_features(size_t* cache_line_size)
   {
   BOTAN_UNUSED(cache_line_size);

   uint64_t detected_features = 0;

#if defined(BOTAN_TARGET_OS_HAS_GETAUXVAL) || defined(BOTAN_TARGET_OS_HAS_ELF_AUX_INFO)
   /*
   * On systems with getauxval these bits should normally be defined
   * in bits/auxv.h but some buggy? glibc installs seem to miss them.
   * These following values are all fixed, for the Linux ELF format,
   * so we just hardcode them in ARM_hwcap_bit enum.
   */

   enum ARM_hwcap_bit {
#if defined(BOTAN_TARGET_ARCH_IS_ARM32)
      NEON_bit  = (1 << 12),
      AES_bit   = (1 << 0),
      PMULL_bit = (1 << 1),
      SHA1_bit  = (1 << 2),
      SHA2_bit  = (1 << 3),

      ARCH_hwcap_neon   = 16, // AT_HWCAP
      ARCH_hwcap_crypto = 26, // AT_HWCAP2
#elif defined(BOTAN_TARGET_ARCH_IS_ARM64)
      NEON_bit  = (1 << 1),
      AES_bit   = (1 << 3),
      PMULL_bit = (1 << 4),
      SHA1_bit  = (1 << 5),
      SHA2_bit  = (1 << 6),
      SHA3_bit  = (1 << 17),
      SM3_bit  = (1 << 18),
      SM4_bit  = (1 << 19),
      SHA2_512_bit  = (1 << 21),
      SVE_bit = (1 << 22),

      ARCH_hwcap_neon   = 16, // AT_HWCAP
      ARCH_hwcap_crypto = 16, // AT_HWCAP
#endif
   };

#if defined(AT_DCACHEBSIZE)
   // Exists only on Linux
   const unsigned long dcache_line = OS::get_auxval(AT_DCACHEBSIZE);

   // plausibility check
   if(dcache_line == 32 || dcache_line == 64 || dcache_line == 128)
      *cache_line_size = static_cast<size_t>(dcache_line);
#endif

   const unsigned long hwcap_neon = OS::get_auxval(ARM_hwcap_bit::ARCH_hwcap_neon);
   if(hwcap_neon & ARM_hwcap_bit::NEON_bit)
      detected_features |= CPUID::CPUID_ARM_NEON_BIT;

   /*
   On aarch64 this ends up calling getauxval twice with AT_HWCAP
   It doesn't seem worth optimizing this out, since getauxval is
   just reading a field in the ELF header.
   */
   const unsigned long hwcap_crypto = OS::get_auxval(ARM_hwcap_bit::ARCH_hwcap_crypto);
   if(hwcap_crypto & ARM_hwcap_bit::AES_bit)
      detected_features |= CPUID::CPUID_ARM_AES_BIT;
   if(hwcap_crypto & ARM_hwcap_bit::PMULL_bit)
      detected_features |= CPUID::CPUID_ARM_PMULL_BIT;
   if(hwcap_crypto & ARM_hwcap_bit::SHA1_bit)
      detected_features |= CPUID::CPUID_ARM_SHA1_BIT;
   if(hwcap_crypto & ARM_hwcap_bit::SHA2_bit)
      detected_features |= CPUID::CPUID_ARM_SHA2_BIT;

#if defined(BOTAN_TARGET_ARCH_IS_ARM64)
   if(hwcap_crypto & ARM_hwcap_bit::SHA3_bit)
      detected_features |= CPUID::CPUID_ARM_SHA3_BIT;
   if(hwcap_crypto & ARM_hwcap_bit::SM3_bit)
      detected_features |= CPUID::CPUID_ARM_SM3_BIT;
   if(hwcap_crypto & ARM_hwcap_bit::SM4_bit)
      detected_features |= CPUID::CPUID_ARM_SM4_BIT;
   if(hwcap_crypto & ARM_hwcap_bit::SHA2_512_bit)
      detected_features |= CPUID::CPUID_ARM_SHA2_512_BIT;
   if(hwcap_crypto & ARM_hwcap_bit::SVE_bit)
      detected_features |= CPUID::CPUID_ARM_SVE_BIT;
#endif

#elif defined(BOTAN_TARGET_OS_IS_IOS)

   char machine[64] = { 0 };
   size_t size = sizeof(machine) - 1;
   ::sysctlbyname("hw.machine", machine, &size, nullptr, 0);

   detected_features = flags_by_ios_machine_type(machine);
   // No way to detect cache line size on iOS?

#elif defined(BOTAN_USE_GCC_INLINE_ASM) && defined(BOTAN_TARGET_ARCH_IS_ARM64)

   /*
   No getauxval API available, fall back on probe functions. We only
   bother with Aarch64 here to simplify the code and because going to
   extreme contortions to support detect NEON on devices that probably
   don't support it doesn't seem worthwhile.

   NEON registers v0-v7 are caller saved in Aarch64
   */

   auto neon_probe  = []() noexcept -> int { asm("and v0.16b, v0.16b, v0.16b"); return 1; };
   auto aes_probe   = []() noexcept -> int { asm(".word 0x4e284800"); return 1; };
   auto pmull_probe = []() noexcept -> int { asm(".word 0x0ee0e000"); return 1; };
   auto sha1_probe  = []() noexcept -> int { asm(".word 0x5e280800"); return 1; };
   auto sha2_probe  = []() noexcept -> int { asm(".word 0x5e282800"); return 1; };

   // Only bother running the crypto detection if we found NEON

   if(OS::run_cpu_instruction_probe(neon_probe) == 1)
      {
      detected_features |= CPUID::CPUID_ARM_NEON_BIT;

      if(OS::run_cpu_instruction_probe(aes_probe) == 1)
         detected_features |= CPUID::CPUID_ARM_AES_BIT;
      if(OS::run_cpu_instruction_probe(pmull_probe) == 1)
         detected_features |= CPUID::CPUID_ARM_PMULL_BIT;
      if(OS::run_cpu_instruction_probe(sha1_probe) == 1)
         detected_features |= CPUID::CPUID_ARM_SHA1_BIT;
      if(OS::run_cpu_instruction_probe(sha2_probe) == 1)
         detected_features |= CPUID::CPUID_ARM_SHA2_BIT;
      }

#endif

   return detected_features;
   }

#endif

}
/*
* Runtime CPU detection for POWER/PowerPC
* (C) 2009,2010,2013,2017 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_TARGET_CPU_IS_PPC_FAMILY)

/*
* On macOS and OpenBSD ppc, use sysctl to detect AltiVec
*/
#if defined(BOTAN_TARGET_OS_IS_MACOS)
  #include <sys/sysctl.h>
#elif defined(BOTAN_TARGET_OS_IS_OPENBSD)
  #include <sys/param.h>
  #include <sys/sysctl.h>
  #include <machine/cpu.h>
#endif

#endif

namespace Botan {

#if defined(BOTAN_TARGET_CPU_IS_PPC_FAMILY)

/*
* PowerPC specific block: check for AltiVec using either
* sysctl or by reading processor version number register.
*/
uint64_t CPUID::CPUID_Data::detect_cpu_features(size_t* cache_line_size)
   {
   BOTAN_UNUSED(cache_line_size);

#if defined(BOTAN_TARGET_OS_IS_MACOS) || defined(BOTAN_TARGET_OS_IS_OPENBSD)
   // On macOS and OpenBSD, use sysctl

   int sels[2] = {
#if defined(BOTAN_TARGET_OS_IS_OPENBSD)
      CTL_MACHDEP, CPU_ALTIVEC
#else
      CTL_HW, HW_VECTORUNIT
#endif
   };

   int vector_type = 0;
   size_t length = sizeof(vector_type);
   int error = ::sysctl(sels, 2, &vector_type, &length, NULL, 0);

   if(error == 0 && vector_type > 0)
      return CPUID::CPUID_ALTIVEC_BIT;

#elif (defined(BOTAN_TARGET_OS_HAS_GETAUXVAL) || defined(BOTAN_TARGET_HAS_ELF_AUX_INFO)) && defined(BOTAN_TARGET_ARCH_IS_PPC64)

   enum PPC_hwcap_bit {
      ALTIVEC_bit  = (1 << 28),
      CRYPTO_bit   = (1 << 25),
      DARN_bit     = (1 << 21),

      ARCH_hwcap_altivec = 16, // AT_HWCAP
      ARCH_hwcap_crypto  = 26, // AT_HWCAP2
   };

   uint64_t detected_features = 0;

   const unsigned long hwcap_altivec = OS::get_auxval(PPC_hwcap_bit::ARCH_hwcap_altivec);
   if(hwcap_altivec & PPC_hwcap_bit::ALTIVEC_bit)
      detected_features |= CPUID::CPUID_ALTIVEC_BIT;

   const unsigned long hwcap_crypto = OS::get_auxval(PPC_hwcap_bit::ARCH_hwcap_crypto);
   if(hwcap_crypto & PPC_hwcap_bit::CRYPTO_bit)
     detected_features |= CPUID::CPUID_POWER_CRYPTO_BIT;
   if(hwcap_crypto & PPC_hwcap_bit::DARN_bit)
     detected_features |= CPUID::CPUID_DARN_BIT;

   return detected_features;

#else

   /*
   On PowerPC, MSR 287 is PVR, the Processor Version Number
   Normally it is only accessible to ring 0, but Linux and NetBSD
   (others, too, maybe?) will trap and emulate it for us.
   */

   int pvr = OS::run_cpu_instruction_probe([]() noexcept -> int {
      uint32_t pvr = 0;
      asm volatile("mfspr %0, 287" : "=r" (pvr));
      // Top 16 bits suffice to identify the model
      return static_cast<int>(pvr >> 16);
      });

   if(pvr > 0)
      {
      const uint16_t ALTIVEC_PVR[] = {
         0x003E, // IBM POWER6
         0x003F, // IBM POWER7
         0x004A, // IBM POWER7p
         0x004B, // IBM POWER8E
         0x004C, // IBM POWER8 NVL
         0x004D, // IBM POWER8
         0x004E, // IBM POWER9
         0x000C, // G4-7400
         0x0039, // G5 970
         0x003C, // G5 970FX
         0x0044, // G5 970MP
         0x0070, // Cell PPU
         0, // end
      };

      for(size_t i = 0; ALTIVEC_PVR[i]; ++i)
         {
         if(pvr == ALTIVEC_PVR[i])
            return CPUID::CPUID_ALTIVEC_BIT;
         }

      return 0;
      }

   // TODO try direct instruction probing

#endif

   return 0;
   }

#endif

}
/*
* Runtime CPU detection for x86
* (C) 2009,2010,2013,2017 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_TARGET_CPU_IS_X86_FAMILY)

#if defined(BOTAN_BUILD_COMPILER_IS_MSVC)
  #include <intrin.h>
#elif defined(BOTAN_BUILD_COMPILER_IS_INTEL)
  #include <ia32intrin.h>
#elif defined(BOTAN_BUILD_COMPILER_IS_GCC) || defined(BOTAN_BUILD_COMPILER_IS_CLANG)
  #include <cpuid.h>
#endif

#endif

namespace Botan {

#if defined(BOTAN_TARGET_CPU_IS_X86_FAMILY)

uint64_t CPUID::CPUID_Data::detect_cpu_features(size_t* cache_line_size)
   {
#if defined(BOTAN_BUILD_COMPILER_IS_MSVC)
  #define X86_CPUID(type, out) do { __cpuid((int*)out, type); } while(0)
  #define X86_CPUID_SUBLEVEL(type, level, out) do { __cpuidex((int*)out, type, level); } while(0)

#elif defined(BOTAN_BUILD_COMPILER_IS_INTEL)
  #define X86_CPUID(type, out) do { __cpuid(out, type); } while(0)
  #define X86_CPUID_SUBLEVEL(type, level, out) do { __cpuidex((int*)out, type, level); } while(0)

#elif defined(BOTAN_TARGET_ARCH_IS_X86_64) && defined(BOTAN_USE_GCC_INLINE_ASM)
  #define X86_CPUID(type, out)                                                    \
     asm("cpuid\n\t" : "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3]) \
         : "0" (type))

  #define X86_CPUID_SUBLEVEL(type, level, out)                                    \
     asm("cpuid\n\t" : "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3]) \
         : "0" (type), "2" (level))

#elif defined(BOTAN_BUILD_COMPILER_IS_GCC) || defined(BOTAN_BUILD_COMPILER_IS_CLANG)
  #define X86_CPUID(type, out) do { __get_cpuid(type, out, out+1, out+2, out+3); } while(0)

  #define X86_CPUID_SUBLEVEL(type, level, out) \
     do { __cpuid_count(type, level, out[0], out[1], out[2], out[3]); } while(0)
#else
  #warning "No way of calling x86 cpuid instruction for this compiler"
  #define X86_CPUID(type, out) do { clear_mem(out, 4); } while(0)
  #define X86_CPUID_SUBLEVEL(type, level, out) do { clear_mem(out, 4); } while(0)
#endif

   uint64_t features_detected = 0;
   uint32_t cpuid[4] = { 0 };

   // CPUID 0: vendor identification, max sublevel
   X86_CPUID(0, cpuid);

   const uint32_t max_supported_sublevel = cpuid[0];

   const uint32_t INTEL_CPUID[3] = { 0x756E6547, 0x6C65746E, 0x49656E69 };
   const uint32_t AMD_CPUID[3] = { 0x68747541, 0x444D4163, 0x69746E65 };
   const bool is_intel = same_mem(cpuid + 1, INTEL_CPUID, 3);
   const bool is_amd = same_mem(cpuid + 1, AMD_CPUID, 3);

   if(max_supported_sublevel >= 1)
      {
      // CPUID 1: feature bits
      X86_CPUID(1, cpuid);
      const uint64_t flags0 = (static_cast<uint64_t>(cpuid[2]) << 32) | cpuid[3];

      enum x86_CPUID_1_bits : uint64_t {
         RDTSC = (1ULL << 4),
         SSE2 = (1ULL << 26),
         CLMUL = (1ULL << 33),
         SSSE3 = (1ULL << 41),
         SSE41 = (1ULL << 51),
         SSE42 = (1ULL << 52),
         AESNI = (1ULL << 57),
         RDRAND = (1ULL << 62)
      };

      if(flags0 & x86_CPUID_1_bits::RDTSC)
         features_detected |= CPUID::CPUID_RDTSC_BIT;
      if(flags0 & x86_CPUID_1_bits::SSE2)
         features_detected |= CPUID::CPUID_SSE2_BIT;
      if(flags0 & x86_CPUID_1_bits::CLMUL)
         features_detected |= CPUID::CPUID_CLMUL_BIT;
      if(flags0 & x86_CPUID_1_bits::SSSE3)
         features_detected |= CPUID::CPUID_SSSE3_BIT;
      if(flags0 & x86_CPUID_1_bits::SSE41)
         features_detected |= CPUID::CPUID_SSE41_BIT;
      if(flags0 & x86_CPUID_1_bits::SSE42)
         features_detected |= CPUID::CPUID_SSE42_BIT;
      if(flags0 & x86_CPUID_1_bits::AESNI)
         features_detected |= CPUID::CPUID_AESNI_BIT;
      if(flags0 & x86_CPUID_1_bits::RDRAND)
         features_detected |= CPUID::CPUID_RDRAND_BIT;
      }

   if(is_intel)
      {
      // Intel cache line size is in cpuid(1) output
      *cache_line_size = 8 * get_byte(2, cpuid[1]);
      }
   else if(is_amd)
      {
      // AMD puts it in vendor zone
      X86_CPUID(0x80000005, cpuid);
      *cache_line_size = get_byte(3, cpuid[2]);
      }

   if(max_supported_sublevel >= 7)
      {
      clear_mem(cpuid, 4);
      X86_CPUID_SUBLEVEL(7, 0, cpuid);

      enum x86_CPUID_7_bits : uint64_t {
         BMI1 = (1ULL << 3),
         AVX2 = (1ULL << 5),
         BMI2 = (1ULL << 8),
         AVX512_F = (1ULL << 16),
         AVX512_DQ = (1ULL << 17),
         RDSEED = (1ULL << 18),
         ADX = (1ULL << 19),
         AVX512_IFMA = (1ULL << 21),
         SHA = (1ULL << 29),
         AVX512_BW = (1ULL << 30),
         AVX512_VL = (1ULL << 31),
         AVX512_VBMI = (1ULL << 33),
         AVX512_VBMI2 = (1ULL << 38),
         AVX512_VAES = (1ULL << 41),
         AVX512_VCLMUL = (1ULL << 42),
         AVX512_VBITALG = (1ULL << 44),
      };

      const uint64_t flags7 = (static_cast<uint64_t>(cpuid[2]) << 32) | cpuid[1];

      if(flags7 & x86_CPUID_7_bits::AVX2)
         features_detected |= CPUID::CPUID_AVX2_BIT;
      if(flags7 & x86_CPUID_7_bits::BMI1)
         {
         features_detected |= CPUID::CPUID_BMI1_BIT;
         /*
         We only set the BMI2 bit if BMI1 is also supported, so BMI2
         code can safely use both extensions. No known processor
         implements BMI2 but not BMI1.
         */
         if(flags7 & x86_CPUID_7_bits::BMI2)
            features_detected |= CPUID::CPUID_BMI2_BIT;
         }

      if(flags7 & x86_CPUID_7_bits::AVX512_F)
         {
         features_detected |= CPUID::CPUID_AVX512F_BIT;

         if(flags7 & x86_CPUID_7_bits::AVX512_DQ)
            features_detected |= CPUID::CPUID_AVX512DQ_BIT;
         if(flags7 & x86_CPUID_7_bits::AVX512_BW)
            features_detected |= CPUID::CPUID_AVX512BW_BIT;

         const uint64_t ICELAKE_FLAGS =
            x86_CPUID_7_bits::AVX512_F |
            x86_CPUID_7_bits::AVX512_DQ |
            x86_CPUID_7_bits::AVX512_IFMA |
            x86_CPUID_7_bits::AVX512_BW |
            x86_CPUID_7_bits::AVX512_VL |
            x86_CPUID_7_bits::AVX512_VBMI |
            x86_CPUID_7_bits::AVX512_VBMI2 |
            x86_CPUID_7_bits::AVX512_VBITALG;

         if((flags7 & ICELAKE_FLAGS) == ICELAKE_FLAGS)
            features_detected |= CPUID::CPUID_AVX512_ICL_BIT;

         if(flags7 & x86_CPUID_7_bits::AVX512_VAES)
            features_detected |= CPUID::CPUID_AVX512_AES_BIT;
         if(flags7 & x86_CPUID_7_bits::AVX512_VCLMUL)
            features_detected |= CPUID::CPUID_AVX512_CLMUL_BIT;
         }

      if(flags7 & x86_CPUID_7_bits::RDSEED)
         features_detected |= CPUID::CPUID_RDSEED_BIT;
      if(flags7 & x86_CPUID_7_bits::ADX)
         features_detected |= CPUID::CPUID_ADX_BIT;
      if(flags7 & x86_CPUID_7_bits::SHA)
         features_detected |= CPUID::CPUID_SHA_BIT;
      }

#undef X86_CPUID
#undef X86_CPUID_SUBLEVEL

   /*
   * If we don't have access to CPUID, we can still safely assume that
   * any x86-64 processor has SSE2 and RDTSC
   */
#if defined(BOTAN_TARGET_ARCH_IS_X86_64)
   if(features_detected == 0)
      {
      features_detected |= CPUID::CPUID_SSE2_BIT;
      features_detected |= CPUID::CPUID_RDTSC_BIT;
      }
#endif

   return features_detected;
   }

#endif

}
/*
* DL Scheme
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

size_t DL_Scheme_PublicKey::key_length() const
   {
   return m_group.p_bits();
   }

size_t DL_Scheme_PublicKey::estimated_strength() const
   {
   return m_group.estimated_strength();
   }

AlgorithmIdentifier DL_Scheme_PublicKey::algorithm_identifier() const
   {
   return AlgorithmIdentifier(get_oid(),
                              m_group.DER_encode(group_format()));
   }

std::vector<uint8_t> DL_Scheme_PublicKey::public_key_bits() const
   {
   std::vector<uint8_t> output;
   DER_Encoder(output).encode(m_y);
   return output;
   }

DL_Scheme_PublicKey::DL_Scheme_PublicKey(const DL_Group& group, const BigInt& y) :
   m_y(y),
   m_group(group)
   {
   }

DL_Scheme_PublicKey::DL_Scheme_PublicKey(const AlgorithmIdentifier& alg_id,
                                         const std::vector<uint8_t>& key_bits,
                                         DL_Group::Format format) :
   m_group(alg_id.get_parameters(), format)
   {
   BER_Decoder(key_bits).decode(m_y);
   }

secure_vector<uint8_t> DL_Scheme_PrivateKey::private_key_bits() const
   {
   return DER_Encoder().encode(m_x).get_contents();
   }

DL_Scheme_PrivateKey::DL_Scheme_PrivateKey(const AlgorithmIdentifier& alg_id,
                                           const secure_vector<uint8_t>& key_bits,
                                           DL_Group::Format format)
   {
   m_group.BER_decode(alg_id.get_parameters(), format);

   BER_Decoder(key_bits).decode(m_x);
   }

/*
* Check Public DL Parameters
*/
bool DL_Scheme_PublicKey::check_key(RandomNumberGenerator& rng,
                                    bool strong) const
   {
   return m_group.verify_group(rng, strong) && m_group.verify_public_element(m_y);
   }

/*
* Check DL Scheme Private Parameters
*/
bool DL_Scheme_PrivateKey::check_key(RandomNumberGenerator& rng,
                                     bool strong) const
   {
   return m_group.verify_group(rng, strong) && m_group.verify_element_pair(m_y, m_x);
   }

}
/*
* Discrete Logarithm Parameters
* (C) 1999-2008,2015,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

class DL_Group_Data final
   {
   public:
      DL_Group_Data(const BigInt& p, const BigInt& q, const BigInt& g, DL_Group_Source source) :
         m_p(p), m_q(q), m_g(g),
         m_mod_p(p),
         m_mod_q(q),
         m_monty_params(std::make_shared<Montgomery_Params>(m_p, m_mod_p)),
         m_monty(monty_precompute(m_monty_params, m_g, /*window bits=*/4)),
         m_p_bits(p.bits()),
         m_q_bits(q.bits()),
         m_estimated_strength(dl_work_factor(m_p_bits)),
         m_exponent_bits(dl_exponent_size(m_p_bits)),
         m_source(source)
         {
         }

      ~DL_Group_Data() = default;

      DL_Group_Data(const DL_Group_Data& other) = delete;
      DL_Group_Data& operator=(const DL_Group_Data& other) = delete;

      const BigInt& p() const { return m_p; }
      const BigInt& q() const { return m_q; }
      const BigInt& g() const { return m_g; }

      BigInt mod_p(const BigInt& x) const { return m_mod_p.reduce(x); }

      BigInt multiply_mod_p(const BigInt& x, const BigInt& y) const
         {
         return m_mod_p.multiply(x, y);
         }

      BigInt mod_q(const BigInt& x) const { return m_mod_q.reduce(x); }

      BigInt multiply_mod_q(const BigInt& x, const BigInt& y) const
         {
         return m_mod_q.multiply(x, y);
         }

      BigInt square_mod_q(const BigInt& x) const
         {
         return m_mod_q.square(x);
         }

      std::shared_ptr<const Montgomery_Params> monty_params_p() const
         { return m_monty_params; }

      size_t p_bits() const { return m_p_bits; }
      size_t q_bits() const { return m_q_bits; }
      size_t p_bytes() const { return (m_p_bits + 7) / 8; }
      size_t q_bytes() const { return (m_q_bits + 7) / 8; }

      size_t estimated_strength() const { return m_estimated_strength; }

      size_t exponent_bits() const { return m_exponent_bits; }

      BigInt power_g_p(const BigInt& k, size_t max_k_bits) const
         {
         return monty_execute(*m_monty, k, max_k_bits);
         }

      bool q_is_set() const { return m_q_bits > 0; }

      void assert_q_is_set(const std::string& function) const
         {
         if(q_is_set() == false)
            throw Invalid_State("DL_Group::" + function + " q is not set for this group");
         }

      DL_Group_Source source() const { return m_source; }

   private:
      BigInt m_p;
      BigInt m_q;
      BigInt m_g;
      Modular_Reducer m_mod_p;
      Modular_Reducer m_mod_q;
      std::shared_ptr<const Montgomery_Params> m_monty_params;
      std::shared_ptr<const Montgomery_Exponentation_State> m_monty;
      size_t m_p_bits;
      size_t m_q_bits;
      size_t m_estimated_strength;
      size_t m_exponent_bits;
      DL_Group_Source m_source;
   };

//static
std::shared_ptr<DL_Group_Data> DL_Group::BER_decode_DL_group(const uint8_t data[], size_t data_len,
                                                             DL_Group::Format format,
                                                             DL_Group_Source source)
   {
   BigInt p, q, g;

   BER_Decoder decoder(data, data_len);
   BER_Decoder ber = decoder.start_cons(SEQUENCE);

   if(format == DL_Group::ANSI_X9_57)
      {
      ber.decode(p)
         .decode(q)
         .decode(g)
         .verify_end();
      }
   else if(format == DL_Group::ANSI_X9_42)
      {
      ber.decode(p)
         .decode(g)
         .decode(q)
         .discard_remaining();
      }
   else if(format == DL_Group::PKCS_3)
      {
      // q is left as zero
      ber.decode(p)
         .decode(g)
         .discard_remaining();
      }
   else
      throw Invalid_Argument("Unknown DL_Group encoding " + std::to_string(format));

   return std::make_shared<DL_Group_Data>(p, q, g, source);
   }

//static
std::shared_ptr<DL_Group_Data>
DL_Group::load_DL_group_info(const char* p_str,
                             const char* q_str,
                             const char* g_str)
   {
   const BigInt p(p_str);
   const BigInt q(q_str);
   const BigInt g(g_str);

   return std::make_shared<DL_Group_Data>(p, q, g, DL_Group_Source::Builtin);
   }

//static
std::shared_ptr<DL_Group_Data>
DL_Group::load_DL_group_info(const char* p_str,
                             const char* g_str)
   {
   const BigInt p(p_str);
   const BigInt q = (p - 1) / 2;
   const BigInt g(g_str);

   return std::make_shared<DL_Group_Data>(p, q, g, DL_Group_Source::Builtin);
   }

namespace {

DL_Group::Format pem_label_to_dl_format(const std::string& label)
   {
   if(label == "DH PARAMETERS")
      return DL_Group::PKCS_3;
   else if(label == "DSA PARAMETERS")
      return DL_Group::ANSI_X9_57;
   else if(label == "X942 DH PARAMETERS" || label == "X9.42 DH PARAMETERS")
      return DL_Group::ANSI_X9_42;
   else
      throw Decoding_Error("DL_Group: Invalid PEM label " + label);
   }

}

/*
* DL_Group Constructor
*/
DL_Group::DL_Group(const std::string& str)
   {
   // Either a name or a PEM block, try name first
   m_data = DL_group_info(str);

   if(m_data == nullptr)
      {
      try
         {
         std::string label;
         const std::vector<uint8_t> ber = unlock(PEM_Code::decode(str, label));
         Format format = pem_label_to_dl_format(label);

         m_data = BER_decode_DL_group(ber.data(), ber.size(), format, DL_Group_Source::ExternalSource);
         }
      catch(...) {}
      }

   if(m_data == nullptr)
      throw Invalid_Argument("DL_Group: Unknown group " + str);
   }

namespace {

/*
* Create generator of the q-sized subgroup (DSA style generator)
*/
BigInt make_dsa_generator(const BigInt& p, const BigInt& q)
   {
   BigInt e, r;
   vartime_divide(p - 1, q, e, r);

   if(e == 0 || r > 0)
      throw Invalid_Argument("make_dsa_generator q does not divide p-1");

   for(size_t i = 0; i != PRIME_TABLE_SIZE; ++i)
      {
      // TODO precompute!
      BigInt g = power_mod(PRIMES[i], e, p);
      if(g > 1)
         return g;
      }

   throw Internal_Error("DL_Group: Couldn't create a suitable generator");
   }

}

/*
* DL_Group Constructor
*/
DL_Group::DL_Group(RandomNumberGenerator& rng,
                   PrimeType type, size_t pbits, size_t qbits)
   {
   if(pbits < 1024)
      throw Invalid_Argument("DL_Group: prime size " + std::to_string(pbits) + " is too small");

   if(type == Strong)
      {
      if(qbits != 0 && qbits != pbits - 1)
         throw Invalid_Argument("Cannot create strong-prime DL_Group with specified q bits");

      const BigInt p = random_safe_prime(rng, pbits);
      const BigInt q = (p - 1) / 2;

      /*
      Always choose a generator that is quadratic reside mod p,
      this forces g to be a generator of the subgroup of size q.
      */
      BigInt g = 2;
      if(jacobi(g, p) != 1)
         {
         // prime table does not contain 2
         for(size_t i = 0; i < PRIME_TABLE_SIZE; ++i)
            {
            g = PRIMES[i];
            if(jacobi(g, p) == 1)
               break;
            }
         }

      m_data = std::make_shared<DL_Group_Data>(p, q, g, DL_Group_Source::RandomlyGenerated);
      }
   else if(type == Prime_Subgroup)
      {
      if(qbits == 0)
         qbits = dl_exponent_size(pbits);

      const BigInt q = random_prime(rng, qbits);
      Modular_Reducer mod_2q(2*q);
      BigInt X;
      BigInt p;
      while(p.bits() != pbits || !is_prime(p, rng, 128, true))
         {
         X.randomize(rng, pbits);
         p = X - mod_2q.reduce(X) + 1;
         }

      const BigInt g = make_dsa_generator(p, q);
      m_data = std::make_shared<DL_Group_Data>(p, q, g, DL_Group_Source::RandomlyGenerated);
      }
   else if(type == DSA_Kosherizer)
      {
      if(qbits == 0)
         qbits = ((pbits <= 1024) ? 160 : 256);

      BigInt p, q;
      generate_dsa_primes(rng, p, q, pbits, qbits);
      const BigInt g = make_dsa_generator(p, q);
      m_data = std::make_shared<DL_Group_Data>(p, q, g, DL_Group_Source::RandomlyGenerated);
      }
   else
      {
      throw Invalid_Argument("DL_Group unknown PrimeType");
      }
   }

/*
* DL_Group Constructor
*/
DL_Group::DL_Group(RandomNumberGenerator& rng,
                   const std::vector<uint8_t>& seed,
                   size_t pbits, size_t qbits)
   {
   BigInt p, q;

   if(!generate_dsa_primes(rng, p, q, pbits, qbits, seed))
      throw Invalid_Argument("DL_Group: The seed given does not generate a DSA group");

   BigInt g = make_dsa_generator(p, q);

   m_data = std::make_shared<DL_Group_Data>(p, q, g, DL_Group_Source::RandomlyGenerated);
   }

/*
* DL_Group Constructor
*/
DL_Group::DL_Group(const BigInt& p, const BigInt& g)
   {
   m_data = std::make_shared<DL_Group_Data>(p, 0, g, DL_Group_Source::ExternalSource);
   }

/*
* DL_Group Constructor
*/
DL_Group::DL_Group(const BigInt& p, const BigInt& q, const BigInt& g)
   {
   m_data = std::make_shared<DL_Group_Data>(p, q, g, DL_Group_Source::ExternalSource);
   }

const DL_Group_Data& DL_Group::data() const
   {
   if(m_data)
      return *m_data;

   throw Invalid_State("DL_Group uninitialized");
   }

bool DL_Group::verify_public_element(const BigInt& y) const
   {
   const BigInt& p = get_p();
   const BigInt& q = get_q();

   if(y <= 1 || y >= p)
      return false;

   if(q.is_zero() == false)
      {
      if(power_mod(y, q, p) != 1)
         return false;
      }

   return true;
   }

bool DL_Group::verify_element_pair(const BigInt& y, const BigInt& x) const
   {
   const BigInt& p = get_p();

   if(y <= 1 || y >= p || x <= 1 || x >= p)
      return false;

   if(y != power_g_p(x))
      return false;

   return true;
   }

/*
* Verify the parameters
*/
bool DL_Group::verify_group(RandomNumberGenerator& rng,
                            bool strong) const
   {
   const bool from_builtin = (source() == DL_Group_Source::Builtin);

   if(!strong && from_builtin)
      return true;

   const BigInt& p = get_p();
   const BigInt& q = get_q();
   const BigInt& g = get_g();

   if(g < 2 || p < 3 || q < 0)
      return false;

   const size_t test_prob = 128;
   const bool is_randomly_generated = (source() != DL_Group_Source::ExternalSource);

   if(q != 0)
      {
      if((p - 1) % q != 0)
         {
         return false;
         }
      if(this->power_g_p(q) != 1)
         {
         return false;
         }
      if(!is_prime(q, rng, test_prob, is_randomly_generated))
         {
         return false;
         }
      }

   if(!is_prime(p, rng, test_prob, is_randomly_generated))
      {
      return false;
      }

   return true;
   }

/*
* Return the prime
*/
const BigInt& DL_Group::get_p() const
   {
   return data().p();
   }

/*
* Return the generator
*/
const BigInt& DL_Group::get_g() const
   {
   return data().g();
   }

/*
* Return the subgroup
*/
const BigInt& DL_Group::get_q() const
   {
   return data().q();
   }

std::shared_ptr<const Montgomery_Params> DL_Group::monty_params_p() const
   {
   return data().monty_params_p();
   }

size_t DL_Group::p_bits() const
   {
   return data().p_bits();
   }

size_t DL_Group::p_bytes() const
   {
   return data().p_bytes();
   }

size_t DL_Group::q_bits() const
   {
   data().assert_q_is_set("q_bits");
   return data().q_bits();
   }

size_t DL_Group::q_bytes() const
   {
   data().assert_q_is_set("q_bytes");
   return data().q_bytes();
   }

size_t DL_Group::estimated_strength() const
   {
   return data().estimated_strength();
   }

size_t DL_Group::exponent_bits() const
   {
   return data().exponent_bits();
   }

BigInt DL_Group::inverse_mod_p(const BigInt& x) const
   {
   // precompute??
   return inverse_mod(x, get_p());
   }

BigInt DL_Group::mod_p(const BigInt& x) const
   {
   return data().mod_p(x);
   }

BigInt DL_Group::multiply_mod_p(const BigInt& x, const BigInt& y) const
   {
   return data().multiply_mod_p(x, y);
   }

BigInt DL_Group::inverse_mod_q(const BigInt& x) const
   {
   data().assert_q_is_set("inverse_mod_q");
   // precompute??
   return inverse_mod(x, get_q());
   }

BigInt DL_Group::mod_q(const BigInt& x) const
   {
   data().assert_q_is_set("mod_q");
   return data().mod_q(x);
   }

BigInt DL_Group::multiply_mod_q(const BigInt& x, const BigInt& y) const
   {
   data().assert_q_is_set("multiply_mod_q");
   return data().multiply_mod_q(x, y);
   }

BigInt DL_Group::multiply_mod_q(const BigInt& x, const BigInt& y, const BigInt& z) const
   {
   data().assert_q_is_set("multiply_mod_q");
   return data().multiply_mod_q(data().multiply_mod_q(x, y), z);
   }

BigInt DL_Group::square_mod_q(const BigInt& x) const
   {
   data().assert_q_is_set("square_mod_q");
   return data().square_mod_q(x);
   }

BigInt DL_Group::multi_exponentiate(const BigInt& x, const BigInt& y, const BigInt& z) const
   {
   return monty_multi_exp(data().monty_params_p(), get_g(), x, y, z);
   }

BigInt DL_Group::power_g_p(const BigInt& x) const
   {
   return data().power_g_p(x, x.bits());
   }

BigInt DL_Group::power_g_p(const BigInt& x, size_t max_x_bits) const
   {
   return data().power_g_p(x, max_x_bits);
   }

DL_Group_Source DL_Group::source() const
   {
   return data().source();
   }

/*
* DER encode the parameters
*/
std::vector<uint8_t> DL_Group::DER_encode(Format format) const
   {
   if(get_q().is_zero() && (format == ANSI_X9_57 || format == ANSI_X9_42))
      throw Encoding_Error("Cannot encode DL_Group in ANSI formats when q param is missing");

   std::vector<uint8_t> output;
   DER_Encoder der(output);

   if(format == ANSI_X9_57)
      {
      der.start_cons(SEQUENCE)
            .encode(get_p())
            .encode(get_q())
            .encode(get_g())
         .end_cons();
      }
   else if(format == ANSI_X9_42)
      {
      der.start_cons(SEQUENCE)
            .encode(get_p())
            .encode(get_g())
            .encode(get_q())
         .end_cons();
      }
   else if(format == PKCS_3)
      {
      der.start_cons(SEQUENCE)
            .encode(get_p())
            .encode(get_g())
         .end_cons();
      }
   else
      throw Invalid_Argument("Unknown DL_Group encoding " + std::to_string(format));

   return output;
   }

/*
* PEM encode the parameters
*/
std::string DL_Group::PEM_encode(Format format) const
   {
   const std::vector<uint8_t> encoding = DER_encode(format);

   if(format == PKCS_3)
      return PEM_Code::encode(encoding, "DH PARAMETERS");
   else if(format == ANSI_X9_57)
      return PEM_Code::encode(encoding, "DSA PARAMETERS");
   else if(format == ANSI_X9_42)
      return PEM_Code::encode(encoding, "X9.42 DH PARAMETERS");
   else
      throw Invalid_Argument("Unknown DL_Group encoding " + std::to_string(format));
   }

DL_Group::DL_Group(const uint8_t ber[], size_t ber_len, Format format)
   {
   m_data = BER_decode_DL_group(ber, ber_len, format, DL_Group_Source::ExternalSource);
   }

void DL_Group::BER_decode(const std::vector<uint8_t>& ber, Format format)
   {
   m_data = BER_decode_DL_group(ber.data(), ber.size(), format, DL_Group_Source::ExternalSource);
   }

//static
DL_Group DL_Group::DL_Group_from_PEM(const std::string& pem)
   {
   std::string label;
   const std::vector<uint8_t> ber = unlock(PEM_Code::decode(pem, label));
   Format format = pem_label_to_dl_format(label);
   return DL_Group(ber, format);
   }

/*
* Decode PEM encoded parameters
*/
void DL_Group::PEM_decode(const std::string& pem)
   {
   std::string label;
   const std::vector<uint8_t> ber = unlock(PEM_Code::decode(pem, label));
   Format format = pem_label_to_dl_format(label);

   m_data = BER_decode_DL_group(ber.data(), ber.size(), format, DL_Group_Source::ExternalSource);
   }

//static
std::string DL_Group::PEM_for_named_group(const std::string& name)
   {
   DL_Group group(name);
   DL_Group::Format format = group.get_q().is_zero() ? DL_Group::PKCS_3 : DL_Group::ANSI_X9_42;
   return group.PEM_encode(format);
   }

}
/*
* List of discrete log groups
* (C) 2013 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

//static
std::shared_ptr<DL_Group_Data> DL_Group::DL_group_info(const std::string& name)
   {
   /* TLS FFDHE groups */

   if(name == "ffdhe/ietf/2048")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFADF85458A2BB4A9AAFDC5620273D3CF1D8B9C583CE2D3695A9E13641146433FBCC939DCE249B3EF97D2FE363630C75D8F681B202AEC4617AD3DF1ED5D5FD65612433F51F5F066ED0856365553DED1AF3B557135E7F57C935984F0C70E0E68B77E2A689DAF3EFE8721DF158A136ADE73530ACCA4F483A797ABC0AB182B324FB61D108A94BB2C8E3FBB96ADAB760D7F4681D4F42A3DE394DF4AE56EDE76372BB190B07A7C8EE0A6D709E02FCE1CDF7E2ECC03404CD28342F619172FE9CE98583FF8E4F1232EEF28183C3FE3B1B4C6FAD733BB5FCBC2EC22005C58EF1837D1683B2C6F34A26C1B2EFFA886B423861285C97FFFFFFFFFFFFFFFF",
                                "0x2");
      }

   if(name == "ffdhe/ietf/3072")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFADF85458A2BB4A9AAFDC5620273D3CF1D8B9C583CE2D3695A9E13641146433FBCC939DCE249B3EF97D2FE363630C75D8F681B202AEC4617AD3DF1ED5D5FD65612433F51F5F066ED0856365553DED1AF3B557135E7F57C935984F0C70E0E68B77E2A689DAF3EFE8721DF158A136ADE73530ACCA4F483A797ABC0AB182B324FB61D108A94BB2C8E3FBB96ADAB760D7F4681D4F42A3DE394DF4AE56EDE76372BB190B07A7C8EE0A6D709E02FCE1CDF7E2ECC03404CD28342F619172FE9CE98583FF8E4F1232EEF28183C3FE3B1B4C6FAD733BB5FCBC2EC22005C58EF1837D1683B2C6F34A26C1B2EFFA886B4238611FCFDCDE355B3B6519035BBC34F4DEF99C023861B46FC9D6E6C9077AD91D2691F7F7EE598CB0FAC186D91CAEFE130985139270B4130C93BC437944F4FD4452E2D74DD364F2E21E71F54BFF5CAE82AB9C9DF69EE86D2BC522363A0DABC521979B0DEADA1DBF9A42D5C4484E0ABCD06BFA53DDEF3C1B20EE3FD59D7C25E41D2B66C62E37FFFFFFFFFFFFFFFF",
                                "0x2");
      }

   if(name == "ffdhe/ietf/4096")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFADF85458A2BB4A9AAFDC5620273D3CF1D8B9C583CE2D3695A9E13641146433FBCC939DCE249B3EF97D2FE363630C75D8F681B202AEC4617AD3DF1ED5D5FD65612433F51F5F066ED0856365553DED1AF3B557135E7F57C935984F0C70E0E68B77E2A689DAF3EFE8721DF158A136ADE73530ACCA4F483A797ABC0AB182B324FB61D108A94BB2C8E3FBB96ADAB760D7F4681D4F42A3DE394DF4AE56EDE76372BB190B07A7C8EE0A6D709E02FCE1CDF7E2ECC03404CD28342F619172FE9CE98583FF8E4F1232EEF28183C3FE3B1B4C6FAD733BB5FCBC2EC22005C58EF1837D1683B2C6F34A26C1B2EFFA886B4238611FCFDCDE355B3B6519035BBC34F4DEF99C023861B46FC9D6E6C9077AD91D2691F7F7EE598CB0FAC186D91CAEFE130985139270B4130C93BC437944F4FD4452E2D74DD364F2E21E71F54BFF5CAE82AB9C9DF69EE86D2BC522363A0DABC521979B0DEADA1DBF9A42D5C4484E0ABCD06BFA53DDEF3C1B20EE3FD59D7C25E41D2B669E1EF16E6F52C3164DF4FB7930E9E4E58857B6AC7D5F42D69F6D187763CF1D5503400487F55BA57E31CC7A7135C886EFB4318AED6A1E012D9E6832A907600A918130C46DC778F971AD0038092999A333CB8B7A1A1DB93D7140003C2A4ECEA9F98D0ACC0A8291CDCEC97DCF8EC9B55A7F88A46B4DB5A851F44182E1C68A007E5E655F6AFFFFFFFFFFFFFFFF",
                                "0x2");
      }

   if(name == "ffdhe/ietf/6144")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFADF85458A2BB4A9AAFDC5620273D3CF1D8B9C583CE2D3695A9E13641146433FBCC939DCE249B3EF97D2FE363630C75D8F681B202AEC4617AD3DF1ED5D5FD65612433F51F5F066ED0856365553DED1AF3B557135E7F57C935984F0C70E0E68B77E2A689DAF3EFE8721DF158A136ADE73530ACCA4F483A797ABC0AB182B324FB61D108A94BB2C8E3FBB96ADAB760D7F4681D4F42A3DE394DF4AE56EDE76372BB190B07A7C8EE0A6D709E02FCE1CDF7E2ECC03404CD28342F619172FE9CE98583FF8E4F1232EEF28183C3FE3B1B4C6FAD733BB5FCBC2EC22005C58EF1837D1683B2C6F34A26C1B2EFFA886B4238611FCFDCDE355B3B6519035BBC34F4DEF99C023861B46FC9D6E6C9077AD91D2691F7F7EE598CB0FAC186D91CAEFE130985139270B4130C93BC437944F4FD4452E2D74DD364F2E21E71F54BFF5CAE82AB9C9DF69EE86D2BC522363A0DABC521979B0DEADA1DBF9A42D5C4484E0ABCD06BFA53DDEF3C1B20EE3FD59D7C25E41D2B669E1EF16E6F52C3164DF4FB7930E9E4E58857B6AC7D5F42D69F6D187763CF1D5503400487F55BA57E31CC7A7135C886EFB4318AED6A1E012D9E6832A907600A918130C46DC778F971AD0038092999A333CB8B7A1A1DB93D7140003C2A4ECEA9F98D0ACC0A8291CDCEC97DCF8EC9B55A7F88A46B4DB5A851F44182E1C68A007E5E0DD9020BFD64B645036C7A4E677D2C38532A3A23BA4442CAF53EA63BB454329B7624C8917BDD64B1C0FD4CB38E8C334C701C3ACDAD0657FCCFEC719B1F5C3E4E46041F388147FB4CFDB477A52471F7A9A96910B855322EDB6340D8A00EF092350511E30ABEC1FFF9E3A26E7FB29F8C183023C3587E38DA0077D9B4763E4E4B94B2BBC194C6651E77CAF992EEAAC0232A281BF6B3A739C1226116820AE8DB5847A67CBEF9C9091B462D538CD72B03746AE77F5E62292C311562A846505DC82DB854338AE49F5235C95B91178CCF2DD5CACEF403EC9D1810C6272B045B3B71F9DC6B80D63FDD4A8E9ADB1E6962A69526D43161C1A41D570D7938DAD4A40E329CD0E40E65FFFFFFFFFFFFFFFF",
                                "0x2");
      }

   if(name == "ffdhe/ietf/8192")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFADF85458A2BB4A9AAFDC5620273D3CF1D8B9C583CE2D3695A9E13641146433FBCC939DCE249B3EF97D2FE363630C75D8F681B202AEC4617AD3DF1ED5D5FD65612433F51F5F066ED0856365553DED1AF3B557135E7F57C935984F0C70E0E68B77E2A689DAF3EFE8721DF158A136ADE73530ACCA4F483A797ABC0AB182B324FB61D108A94BB2C8E3FBB96ADAB760D7F4681D4F42A3DE394DF4AE56EDE76372BB190B07A7C8EE0A6D709E02FCE1CDF7E2ECC03404CD28342F619172FE9CE98583FF8E4F1232EEF28183C3FE3B1B4C6FAD733BB5FCBC2EC22005C58EF1837D1683B2C6F34A26C1B2EFFA886B4238611FCFDCDE355B3B6519035BBC34F4DEF99C023861B46FC9D6E6C9077AD91D2691F7F7EE598CB0FAC186D91CAEFE130985139270B4130C93BC437944F4FD4452E2D74DD364F2E21E71F54BFF5CAE82AB9C9DF69EE86D2BC522363A0DABC521979B0DEADA1DBF9A42D5C4484E0ABCD06BFA53DDEF3C1B20EE3FD59D7C25E41D2B669E1EF16E6F52C3164DF4FB7930E9E4E58857B6AC7D5F42D69F6D187763CF1D5503400487F55BA57E31CC7A7135C886EFB4318AED6A1E012D9E6832A907600A918130C46DC778F971AD0038092999A333CB8B7A1A1DB93D7140003C2A4ECEA9F98D0ACC0A8291CDCEC97DCF8EC9B55A7F88A46B4DB5A851F44182E1C68A007E5E0DD9020BFD64B645036C7A4E677D2C38532A3A23BA4442CAF53EA63BB454329B7624C8917BDD64B1C0FD4CB38E8C334C701C3ACDAD0657FCCFEC719B1F5C3E4E46041F388147FB4CFDB477A52471F7A9A96910B855322EDB6340D8A00EF092350511E30ABEC1FFF9E3A26E7FB29F8C183023C3587E38DA0077D9B4763E4E4B94B2BBC194C6651E77CAF992EEAAC0232A281BF6B3A739C1226116820AE8DB5847A67CBEF9C9091B462D538CD72B03746AE77F5E62292C311562A846505DC82DB854338AE49F5235C95B91178CCF2DD5CACEF403EC9D1810C6272B045B3B71F9DC6B80D63FDD4A8E9ADB1E6962A69526D43161C1A41D570D7938DAD4A40E329CCFF46AAA36AD004CF600C8381E425A31D951AE64FDB23FCEC9509D43687FEB69EDD1CC5E0B8CC3BDF64B10EF86B63142A3AB8829555B2F747C932665CB2C0F1CC01BD70229388839D2AF05E454504AC78B7582822846C0BA35C35F5C59160CC046FD8251541FC68C9C86B022BB7099876A460E7451A8A93109703FEE1C217E6C3826E52C51AA691E0E423CFC99E9E31650C1217B624816CDAD9A95F9D5B8019488D9C0A0A1FE3075A577E23183F81D4A3F2FA4571EFC8CE0BA8A4FE8B6855DFE72B0A66EDED2FBABFBE58A30FAFABE1C5D71A87E2F741EF8C1FE86FEA6BBFDE530677F0D97D11D49F7A8443D0822E506A9F4614E011E2A94838FF88CD68C8BB7C5C6424CFFFFFFFFFFFFFFFF",
                                "0x2");
      }

   /* IETF IPsec groups */

   if(name == "modp/ietf/1024")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE65381FFFFFFFFFFFFFFFF",
                                "0x2");
      }

   if(name == "modp/ietf/1536")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA237327FFFFFFFFFFFFFFFF",
                                "0x2");
      }

   if(name == "modp/ietf/2048")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF",
                                "0x2");
      }

   if(name == "modp/ietf/3072")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A93AD2CAFFFFFFFFFFFFFFFF",
                                "0x2");
      }

   if(name == "modp/ietf/4096")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C934063199FFFFFFFFFFFFFFFF",
                                "0x2");
      }

   if(name == "modp/ietf/6144")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C93402849236C3FAB4D27C7026C1D4DCB2602646DEC9751E763DBA37BDF8FF9406AD9E530EE5DB382F413001AEB06A53ED9027D831179727B0865A8918DA3EDBEBCF9B14ED44CE6CBACED4BB1BDB7F1447E6CC254B332051512BD7AF426FB8F401378CD2BF5983CA01C64B92ECF032EA15D1721D03F482D7CE6E74FEF6D55E702F46980C82B5A84031900B1C9E59E7C97FBEC7E8F323A97A7E36CC88BE0F1D45B7FF585AC54BD407B22B4154AACC8F6D7EBF48E1D814CC5ED20F8037E0A79715EEF29BE32806A1D58BB7C5DA76F550AA3D8A1FBFF0EB19CCB1A313D55CDA56C9EC2EF29632387FE8D76E3C0468043E8F663F4860EE12BF2D5B0B7474D6E694F91E6DCC4024FFFFFFFFFFFFFFFF",
                                "0x2");
      }

   if(name == "modp/ietf/8192")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C93402849236C3FAB4D27C7026C1D4DCB2602646DEC9751E763DBA37BDF8FF9406AD9E530EE5DB382F413001AEB06A53ED9027D831179727B0865A8918DA3EDBEBCF9B14ED44CE6CBACED4BB1BDB7F1447E6CC254B332051512BD7AF426FB8F401378CD2BF5983CA01C64B92ECF032EA15D1721D03F482D7CE6E74FEF6D55E702F46980C82B5A84031900B1C9E59E7C97FBEC7E8F323A97A7E36CC88BE0F1D45B7FF585AC54BD407B22B4154AACC8F6D7EBF48E1D814CC5ED20F8037E0A79715EEF29BE32806A1D58BB7C5DA76F550AA3D8A1FBFF0EB19CCB1A313D55CDA56C9EC2EF29632387FE8D76E3C0468043E8F663F4860EE12BF2D5B0B7474D6E694F91E6DBE115974A3926F12FEE5E438777CB6A932DF8CD8BEC4D073B931BA3BC832B68D9DD300741FA7BF8AFC47ED2576F6936BA424663AAB639C5AE4F5683423B4742BF1C978238F16CBE39D652DE3FDB8BEFC848AD922222E04A4037C0713EB57A81A23F0C73473FC646CEA306B4BCBC8862F8385DDFA9D4B7FA2C087E879683303ED5BDD3A062B3CF5B3A278A66D2A13F83F44F82DDF310EE074AB6A364597E899A0255DC164F31CC50846851DF9AB48195DED7EA1B1D510BD7EE74D73FAF36BC31ECFA268359046F4EB879F924009438B481C6CD7889A002ED5EE382BC9190DA6FC026E479558E4475677E9AA9E3050E2765694DFC81F56E880B96E7160C980DD98EDD3DFFFFFFFFFFFFFFFFF",
                                "0x2");
      }

   /* SRP groups

   SRP groups have a p st (p-1)/2 is prime, but g is not a generator
   of subgroup of size q, so set q == 0 to bypass generator check

   Missing q doesn't matter for SRP, and nothing but SRP should be
   using these parameters.
   */

   if(name == "modp/srp/1024")
      {
      return load_DL_group_info("0xEEAF0AB9ADB38DD69C33F80AFA8FC5E86072618775FF3C0B9EA2314C9C256576D674DF7496EA81D3383B4813D692C6E0E0D5D8E250B98BE48E495C1D6089DAD15DC7D7B46154D6B6CE8EF4AD69B15D4982559B297BCF1885C529F566660E57EC68EDBC3C05726CC02FD4CBF4976EAA9AFD5138FE8376435B9FC61D2FC0EB06E3",
                                "0",
                                "0x2");
      }

   if(name == "modp/srp/1536")
      {
      return load_DL_group_info("0x9DEF3CAFB939277AB1F12A8617A47BBBDBA51DF499AC4C80BEEEA9614B19CC4D5F4F5F556E27CBDE51C6A94BE4607A291558903BA0D0F84380B655BB9A22E8DCDF028A7CEC67F0D08134B1C8B97989149B609E0BE3BAB63D47548381DBC5B1FC764E3F4B53DD9DA1158BFD3E2B9C8CF56EDF019539349627DB2FD53D24B7C48665772E437D6C7F8CE442734AF7CCB7AE837C264AE3A9BEB87F8A2FE9B8B5292E5A021FFF5E91479E8CE7A28C2442C6F315180F93499A234DCF76E3FED135F9BB",
                                "0",
                                "0x2");
      }

   if(name == "modp/srp/2048")
      {
      return load_DL_group_info("0xAC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF73",
                                "0",
                                "0x2");
      }

   if(name == "modp/srp/3072")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A93AD2CAFFFFFFFFFFFFFFFF",
                                "0",
                                "0x5");
      }

   if(name == "modp/srp/4096")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C934063199FFFFFFFFFFFFFFFF",
                                "0",
                                "0x5");
      }

   if(name == "modp/srp/6144")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C93402849236C3FAB4D27C7026C1D4DCB2602646DEC9751E763DBA37BDF8FF9406AD9E530EE5DB382F413001AEB06A53ED9027D831179727B0865A8918DA3EDBEBCF9B14ED44CE6CBACED4BB1BDB7F1447E6CC254B332051512BD7AF426FB8F401378CD2BF5983CA01C64B92ECF032EA15D1721D03F482D7CE6E74FEF6D55E702F46980C82B5A84031900B1C9E59E7C97FBEC7E8F323A97A7E36CC88BE0F1D45B7FF585AC54BD407B22B4154AACC8F6D7EBF48E1D814CC5ED20F8037E0A79715EEF29BE32806A1D58BB7C5DA76F550AA3D8A1FBFF0EB19CCB1A313D55CDA56C9EC2EF29632387FE8D76E3C0468043E8F663F4860EE12BF2D5B0B7474D6E694F91E6DCC4024FFFFFFFFFFFFFFFF",
                                "0",
                                "0x5");
      }

   if(name == "modp/srp/8192")
      {
      return load_DL_group_info("0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C93402849236C3FAB4D27C7026C1D4DCB2602646DEC9751E763DBA37BDF8FF9406AD9E530EE5DB382F413001AEB06A53ED9027D831179727B0865A8918DA3EDBEBCF9B14ED44CE6CBACED4BB1BDB7F1447E6CC254B332051512BD7AF426FB8F401378CD2BF5983CA01C64B92ECF032EA15D1721D03F482D7CE6E74FEF6D55E702F46980C82B5A84031900B1C9E59E7C97FBEC7E8F323A97A7E36CC88BE0F1D45B7FF585AC54BD407B22B4154AACC8F6D7EBF48E1D814CC5ED20F8037E0A79715EEF29BE32806A1D58BB7C5DA76F550AA3D8A1FBFF0EB19CCB1A313D55CDA56C9EC2EF29632387FE8D76E3C0468043E8F663F4860EE12BF2D5B0B7474D6E694F91E6DBE115974A3926F12FEE5E438777CB6A932DF8CD8BEC4D073B931BA3BC832B68D9DD300741FA7BF8AFC47ED2576F6936BA424663AAB639C5AE4F5683423B4742BF1C978238F16CBE39D652DE3FDB8BEFC848AD922222E04A4037C0713EB57A81A23F0C73473FC646CEA306B4BCBC8862F8385DDFA9D4B7FA2C087E879683303ED5BDD3A062B3CF5B3A278A66D2A13F83F44F82DDF310EE074AB6A364597E899A0255DC164F31CC50846851DF9AB48195DED7EA1B1D510BD7EE74D73FAF36BC31ECFA268359046F4EB879F924009438B481C6CD7889A002ED5EE382BC9190DA6FC026E479558E4475677E9AA9E3050E2765694DFC81F56E880B96E7160C980DD98EDD3DFFFFFFFFFFFFFFFFF",
                                "0",
                                "0x13");
      }

   /* DSA groups */

   if(name == "dsa/jce/1024")
      {
      return load_DL_group_info("0xFD7F53811D75122952DF4A9C2EECE4E7F611B7523CEF4400C31E3F80B6512669455D402251FB593D8D58FABFC5F5BA30F6CB9B556CD7813B801D346FF26660B76B9950A5A49F9FE8047B1022C24FBBA9D7FEB7C61BF83B57E7C6A8A6150F04FB83F6D3C51EC3023554135A169132F675F3AE2B61D72AEFF22203199DD14801C7",
                                "0x9760508F15230BCCB292B982A2EB840BF0581CF5",
                                "0x469603512E30278CD3947595DB22EEC9826A6322ADC97344F41D740C325724C8F9EFBAA7D4D803FF8C609DCD100EBC5BDFCFAD7C6A425FAEA786EA2050EBE98351EA1FDA1FDF24D6947AA6B9AA23766953802F4D7D4A8ECBA06D19768A2491FFB16D0EF9C43A99B5F71672FF6F0A24B444D0736D04D38A1A1322DAF6CDD88C9D");
      }

   if(name == "dsa/botan/2048")
      {
      return load_DL_group_info("0x91C48A4FDFBCF7C02AE95E7DA126122B5DD2864F559B87E8E74A286D52F59BD1DE68DFD645D0E00C60C080031891980374EEB594A532BFD67B9A09EAC4B8663A07910E68F39465FB7040D25DF13932EBAC4347A530ECBA61C854F9B880D3C0C3660080587C45566DADE26BD5A394BE093B4C0F24B5AFFEF8EC6C5B3E57FB89025A9BC16769932131E16D3C94EFCAB18D0DF061203CC53E6103BC72D5594BFD40CA65380F44A9A851DCB075495FC033A8A58071A1BD78FE052F66555648EB4B719D2AFE8B4880F8DAD6F15818BA178F89274C870BE9B96EB08C46C40040CC2EFE1DFB1B1868DD319DE3C34A32A63AB6EB1224209A419680CC7902D1728D4DF9E1",
                                "0x8CD7D450F86F0AD94EEE4CE469A8756D1EBD1058241943EAFFB0B354585E924D",
                                "0xD9F5E0761B4DBD1833D6AB1A961A0996C5F22303F72D84C140F67C431D94AB5715BEA81A0C98D39CE4BCF78D6B9EBC895D34FE89D94091D5848615EF15F5E86F11D96F6C969E203DDFA58356420A49CB444B595B901A933CFE0767B594F18A07B7F91DECDBA446B88990F78F2FF91F2FE7CD43FD2E46D18EADA1F7BB6602C617F6EF3A4B284F2FD9BA10A36042DE8FA87A2CA36597FEC81157A1485E44041DF02830111CB880BBE6ED494814886F965CDC3135F5CCF1383728BF65B806F9692C0B10D6C4C09C75A6CA3B4013CB16AB2C105F6BE23AEA9000EAB2178985F972C98057E1C86E44E7218688EA4AE0F3636DCCA745C9DCD4E6AFFB67CCBC13D6131");
      }

   if(name == "dsa/botan/3072")
      {
      return load_DL_group_info("0xE4B50880759663585E142460CA2D9DFF132F8AE4C840DDA3A2666889124FE5638B84E8A29B7AF3FA1209BE6BFC4B5072ED3B2B7387BAF3F857F478A80228EF3600B76B3DCFB61D20D34465B2506D2CAF87DF6E7DC0CE91BD2D167A46F6ADCC31C531E4F9C7ABBDB92ADDF35B0A806C66292A5F5E17E964DD099903733AC428AB35D80EA6F685BFBA8BE4068E5418AE5ECAD9E8FF073DE2B63E4E7EAD35C8A9B70B5BD47CFB88D373B66F37931939B0AB71BD5595809086DA0155337D185A0E4FB36A519B1B6202B8591E6002449CF1CD3A66384F6D2073B1CD73BECA93BAF1E1A6117D0238F222AE1ED7FED185A890E7F67FAB8FEB9753CC134A5183DFE87AE2595F7B5C2D9FBB42249FDD59513E1D3396B3EB2FD86684F285A8448FE757A029881C40760B94EF919BDF9740C38389599EC51A6E9BB519A8E068491E9CE0A2FCFE3CB60D66CF0DFAD20A8EC684048684A61444575BD1724D7352B44A760077B3BD6BD385CE5B0A7250CC0BF768DA82923806EB9CFBB138843731B618208C759B",
                                "0xB3EBD364EC69EF8CF3BAF643B75734B16339B2E49E5CDE1B59C1E9FB40EE0C5B",
                                "0x2BED21EEF83964A230AE89BBA71D9F7C39C52FC8229B4E3BC7E5944D329DA10F010EAC9E7BAF6C009FC4EB2960723E2B56DF4663E4C3AC800E9258DE2F7649D206782893F865EFCA498D2EEF30074EA5E8A7AB262712A4D94A2F3B0B9A92EE400FB38A3CC59A5DC7E436D5C004B22E35028381B51C93407EB32D4AE0FD42CB45E12D0ECEE8A26238EDE2082A7B1522113C66CEF8D745C6CF3CB945F84D2F4DE16D44A71DE198270E13F03553C88B8D323AD0B948A1BF2103A949979B6ED16FB5F3C953D95B7C8E88CA67DCF5A636FB9CA39D924215F7A884ED6C7EE3C96D8D9715427974B7C4351282E13D3773F7D28B452F10892A13C7587328DEA4827B6B369B2A8DC172ADC583F51F2A6598C5483E5BC467B02F91D059C402D18E2C2680F776AA06F49280A2C72C17CC42D5B6E740C5C4B1AB3C51C2ED092BE2A2D8B053AE5773D1425ED2B08F06E2DD50592DF1A478C15591CDFD11564FF88FF38B721D42392FDA473212DCFD8D2D88A976A00AFFE6FFFB430A359E64CA2B351CA2412394");
      }

   return std::shared_ptr<DL_Group_Data>();
   }

}
/*
* DSA
* (C) 1999-2010,2014,2016 Jack Lloyd
* (C) 2016 René Korthaus
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_RFC6979_GENERATOR)
#endif

namespace Botan {

/*
* DSA_PublicKey Constructor
*/
DSA_PublicKey::DSA_PublicKey(const DL_Group& grp, const BigInt& y1)
   {
   m_group = grp;
   m_y = y1;
   }

/*
* Create a DSA private key
*/
DSA_PrivateKey::DSA_PrivateKey(RandomNumberGenerator& rng,
                               const DL_Group& grp,
                               const BigInt& x_arg)
   {
   m_group = grp;

   if(x_arg == 0)
      m_x = BigInt::random_integer(rng, 2, group_q());
   else
      m_x = x_arg;

   m_y = m_group.power_g_p(m_x, m_group.q_bits());
   }

DSA_PrivateKey::DSA_PrivateKey(const AlgorithmIdentifier& alg_id,
                               const secure_vector<uint8_t>& key_bits) :
   DL_Scheme_PrivateKey(alg_id, key_bits, DL_Group::ANSI_X9_57)
   {
   m_y = m_group.power_g_p(m_x, m_group.q_bits());
   }

/*
* Check Private DSA Parameters
*/
bool DSA_PrivateKey::check_key(RandomNumberGenerator& rng, bool strong) const
   {
   if(!DL_Scheme_PrivateKey::check_key(rng, strong) || m_x >= group_q())
      return false;

   if(!strong)
      return true;

   return KeyPair::signature_consistency_check(rng, *this, "EMSA1(SHA-256)");
   }

namespace {

/**
* Object that can create a DSA signature
*/
class DSA_Signature_Operation final : public PK_Ops::Signature_with_EMSA
   {
   public:
      DSA_Signature_Operation(const DSA_PrivateKey& dsa,
                              const std::string& emsa,
                              RandomNumberGenerator& rng) :
         PK_Ops::Signature_with_EMSA(emsa),
         m_group(dsa.get_group()),
         m_x(dsa.get_x())
         {
#if defined(BOTAN_HAS_RFC6979_GENERATOR)
         m_rfc6979_hash = hash_for_emsa(emsa);
#endif

         m_b = BigInt::random_integer(rng, 2, dsa.group_q());
         m_b_inv = m_group.inverse_mod_q(m_b);
         }

      size_t signature_length() const override { return 2*m_group.q_bytes(); }
      size_t max_input_bits() const override { return m_group.q_bits(); }

      secure_vector<uint8_t> raw_sign(const uint8_t msg[], size_t msg_len,
                                   RandomNumberGenerator& rng) override;
   private:
      const DL_Group m_group;
      const BigInt& m_x;
#if defined(BOTAN_HAS_RFC6979_GENERATOR)
      std::string m_rfc6979_hash;
#endif

      BigInt m_b, m_b_inv;
   };

secure_vector<uint8_t>
DSA_Signature_Operation::raw_sign(const uint8_t msg[], size_t msg_len,
                                  RandomNumberGenerator& rng)
   {
   const BigInt& q = m_group.get_q();

   BigInt m(msg, msg_len, m_group.q_bits());

   while(m >= q)
      m -= q;

#if defined(BOTAN_HAS_RFC6979_GENERATOR)
   BOTAN_UNUSED(rng);
   const BigInt k = generate_rfc6979_nonce(m_x, q, m, m_rfc6979_hash);
#else
   const BigInt k = BigInt::random_integer(rng, 1, q);
#endif

   const BigInt k_inv = m_group.inverse_mod_q(k);

   /*
   * It may not be strictly necessary for the reduction (g^k mod p) mod q to be
   * const time, since r is published as part of the signature, and deriving
   * anything useful about k from g^k mod p would seem to require computing a
   * discrete logarithm.
   *
   * However it only increases the cost of signatures by about 7-10%, and DSA is
   * only for legacy use anyway so we don't care about the performance so much.
   */
   const BigInt r = ct_modulo(m_group.power_g_p(k, m_group.q_bits()), m_group.get_q());

   /*
   * Blind the input message and compute x*r+m as (x*r*b + m*b)/b
   */
   m_b = m_group.square_mod_q(m_b);
   m_b_inv = m_group.square_mod_q(m_b_inv);

   m = m_group.multiply_mod_q(m_b, m);
   const BigInt xr = m_group.multiply_mod_q(m_b, m_x, r);

   const BigInt s = m_group.multiply_mod_q(m_b_inv, k_inv, m_group.mod_q(xr+m));

   // With overwhelming probability, a bug rather than actual zero r/s
   if(r.is_zero() || s.is_zero())
      throw Internal_Error("Computed zero r/s during DSA signature");

   return BigInt::encode_fixed_length_int_pair(r, s, q.bytes());
   }

/**
* Object that can verify a DSA signature
*/
class DSA_Verification_Operation final : public PK_Ops::Verification_with_EMSA
   {
   public:
      DSA_Verification_Operation(const DSA_PublicKey& dsa,
                                 const std::string& emsa) :
         PK_Ops::Verification_with_EMSA(emsa),
         m_group(dsa.get_group()),
         m_y(dsa.get_y())
         {
         }

      size_t max_input_bits() const override { return m_group.q_bits(); }

      bool with_recovery() const override { return false; }

      bool verify(const uint8_t msg[], size_t msg_len,
                  const uint8_t sig[], size_t sig_len) override;
   private:
      const DL_Group m_group;
      const BigInt& m_y;
   };

bool DSA_Verification_Operation::verify(const uint8_t msg[], size_t msg_len,
                                        const uint8_t sig[], size_t sig_len)
   {
   const BigInt& q = m_group.get_q();
   const size_t q_bytes = q.bytes();

   if(sig_len != 2*q_bytes || msg_len > q_bytes)
      return false;

   BigInt r(sig, q_bytes);
   BigInt s(sig + q_bytes, q_bytes);
   BigInt i(msg, msg_len, q.bits());

   if(r <= 0 || r >= q || s <= 0 || s >= q)
      return false;

   s = inverse_mod(s, q);

   const BigInt sr = m_group.multiply_mod_q(s, r);
   const BigInt si = m_group.multiply_mod_q(s, i);

   s = m_group.multi_exponentiate(si, m_y, sr);

   // s is too big for Barrett, and verification doesn't need to be const-time
   return (s % m_group.get_q() == r);
   }

}

std::unique_ptr<PK_Ops::Verification>
DSA_PublicKey::create_verification_op(const std::string& params,
                                      const std::string& provider) const
   {
   if(provider == "base" || provider.empty())
      return std::unique_ptr<PK_Ops::Verification>(new DSA_Verification_Operation(*this, params));
   throw Provider_Not_Found(algo_name(), provider);
   }

std::unique_ptr<PK_Ops::Signature>
DSA_PrivateKey::create_signature_op(RandomNumberGenerator& rng,
                                    const std::string& params,
                                    const std::string& provider) const
   {
   if(provider == "base" || provider.empty())
      return std::unique_ptr<PK_Ops::Signature>(new DSA_Signature_Operation(*this, params, rng));
   throw Provider_Not_Found(algo_name(), provider);
   }

}
/*
* EMSA1
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

secure_vector<uint8_t> emsa1_encoding(const secure_vector<uint8_t>& msg,
                                  size_t output_bits)
   {
   if(8*msg.size() <= output_bits)
      return msg;

   size_t shift = 8*msg.size() - output_bits;

   size_t byte_shift = shift / 8, bit_shift = shift % 8;
   secure_vector<uint8_t> digest(msg.size() - byte_shift);

   for(size_t j = 0; j != msg.size() - byte_shift; ++j)
      digest[j] = msg[j];

   if(bit_shift)
      {
      uint8_t carry = 0;
      for(size_t j = 0; j != digest.size(); ++j)
         {
         uint8_t temp = digest[j];
         digest[j] = (temp >> bit_shift) | carry;
         carry = (temp << (8 - bit_shift));
         }
      }
   return digest;
   }

}

std::string EMSA1::name() const
   {
   return "EMSA1(" + m_hash->name() + ")";
   }

EMSA* EMSA1::clone()
   {
   return new EMSA1(m_hash->clone());
   }

void EMSA1::update(const uint8_t input[], size_t length)
   {
   m_hash->update(input, length);
   }

secure_vector<uint8_t> EMSA1::raw_data()
   {
   return m_hash->final();
   }

secure_vector<uint8_t> EMSA1::encoding_of(const secure_vector<uint8_t>& msg,
                                       size_t output_bits,
                                       RandomNumberGenerator&)
   {
   if(msg.size() != hash_output_length())
      throw Encoding_Error("EMSA1::encoding_of: Invalid size for input");
   return emsa1_encoding(msg, output_bits);
   }

bool EMSA1::verify(const secure_vector<uint8_t>& input,
                   const secure_vector<uint8_t>& raw,
                   size_t key_bits)
   {
   if(raw.size() != m_hash->output_length())
      return false;

   // Call emsa1_encoding to handle any required bit shifting
   const secure_vector<uint8_t> our_coding = emsa1_encoding(raw, key_bits);

   if(our_coding.size() < input.size())
      return false;

   const size_t offset = our_coding.size() - input.size(); // must be >= 0 per check above

   // If our encoding is longer, all the bytes in it must be zero
   for(size_t i = 0; i != offset; ++i)
      if(our_coding[i] != 0)
            return false;

   return constant_time_compare(input.data(), &our_coding[offset], input.size());
   }

AlgorithmIdentifier EMSA1::config_for_x509(const Private_Key& key,
                                           const std::string& cert_hash_name) const
   {
   if(cert_hash_name != m_hash->name())
      throw Invalid_Argument("Hash function from opts and hash_fn argument"
         " need to be identical");
   // check that the signature algorithm and the padding scheme fit
   if(!sig_algo_and_pad_ok(key.algo_name(), "EMSA1"))
      {
      throw Invalid_Argument("Encoding scheme with canonical name EMSA1"
         " not supported for signature algorithm " + key.algo_name());
      }

   const OID oid = OID::from_string(key.algo_name() + "/" + name());

   const std::string algo_name = key.algo_name();
   std::vector<uint8_t> parameters;
   if(algo_name == "DSA" ||
      algo_name == "ECDSA" ||
      algo_name == "ECGDSA" ||
      algo_name == "ECKCDSA" ||
      algo_name == "GOST-34.10" ||
      algo_name == "GOST-34.10-2012-256" ||
      algo_name == "GOST-34.10-2012-512")
      {
      // for DSA, ECDSA, GOST parameters "SHALL" be empty
      }
   else
      {
      parameters = key.algorithm_identifier().get_parameters();
      }

   return AlgorithmIdentifier(oid, parameters);
   }

}
/*
* Entropy Source Polling
* (C) 2008-2010,2015 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_SYSTEM_RNG)
#endif

#if defined(BOTAN_HAS_PROCESSOR_RNG)
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_RDRAND)
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_RDSEED)
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_DARN)
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_DEV_RANDOM)
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_WIN32)
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_PROC_WALKER)
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_GETENTROPY)
#endif

namespace Botan {

namespace {

#if defined(BOTAN_HAS_SYSTEM_RNG)

class System_RNG_EntropySource final : public Entropy_Source
   {
   public:
      size_t poll(RandomNumberGenerator& rng) override
         {
         const size_t poll_bits = BOTAN_RNG_RESEED_POLL_BITS;
         rng.reseed_from_rng(system_rng(), poll_bits);
         return poll_bits;
         }

      std::string name() const override { return "system_rng"; }
   };

#endif

#if defined(BOTAN_HAS_PROCESSOR_RNG)

class Processor_RNG_EntropySource final : public Entropy_Source
   {
   public:
      size_t poll(RandomNumberGenerator& rng) override
         {
         /*
         * Intel's documentation for RDRAND at
         * https://software.intel.com/en-us/articles/intel-digital-random-number-generator-drng-software-implementation-guide
         * claims that software can guarantee a reseed event by polling enough data:
         * "There is an upper bound of 511 samples per seed in the implementation
         * where samples are 128 bits in size and can provide two 64-bit random
         * numbers each."
         *
         * By requesting 65536 bits we are asking for 512 samples and thus are assured
         * that at some point in producing the output, at least one reseed of the
         * internal state will occur.
         *
         * The reseeding conditions of the POWER and ARM processor RNGs are not known
         * but probably work in a somewhat similar manner. The exact amount requested
         * may be tweaked if and when such conditions become publically known.
         */
         const size_t poll_bits = 65536;
         rng.reseed_from_rng(m_hwrng, poll_bits);
         // Avoid trusting a black box, don't count this as contributing entropy:
         return 0;
         }

      std::string name() const override { return m_hwrng.name(); }
   private:
      Processor_RNG m_hwrng;
   };

#endif

}

std::unique_ptr<Entropy_Source> Entropy_Source::create(const std::string& name)
   {
#if defined(BOTAN_HAS_SYSTEM_RNG)
   if(name == "system_rng" || name == "win32_cryptoapi")
      {
      return std::unique_ptr<Entropy_Source>(new System_RNG_EntropySource);
      }
#endif

#if defined(BOTAN_HAS_PROCESSOR_RNG)
   if(name == "hwrng" || name == "rdrand" || name == "p9_darn")
      {
      if(Processor_RNG::available())
         {
         return std::unique_ptr<Entropy_Source>(new Processor_RNG_EntropySource);
         }
      }
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_RDSEED)
   if(name == "rdseed")
      {
      return std::unique_ptr<Entropy_Source>(new Intel_Rdseed);
      }
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_GETENTROPY)
   if(name == "getentropy")
      {
      return std::unique_ptr<Entropy_Source>(new Getentropy);
      }
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_DEV_RANDOM)
   if(name == "dev_random")
      {
      return std::unique_ptr<Entropy_Source>(new Device_EntropySource(BOTAN_SYSTEM_RNG_POLL_DEVICES));
      }
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_PROC_WALKER)
   if(name == "proc_walk" && OS::running_in_privileged_state() == false)
      {
      const std::string root_dir = BOTAN_ENTROPY_PROC_FS_PATH;
      if(!root_dir.empty())
         return std::unique_ptr<Entropy_Source>(new ProcWalking_EntropySource(root_dir));
      }
#endif

#if defined(BOTAN_HAS_ENTROPY_SRC_WIN32)
   if(name == "system_stats")
      {
      return std::unique_ptr<Entropy_Source>(new Win32_EntropySource);
      }
#endif

   BOTAN_UNUSED(name);
   return std::unique_ptr<Entropy_Source>();
   }

void Entropy_Sources::add_source(std::unique_ptr<Entropy_Source> src)
   {
   if(src.get())
      {
      m_srcs.push_back(std::move(src));
      }
   }

std::vector<std::string> Entropy_Sources::enabled_sources() const
   {
   std::vector<std::string> sources;
   for(size_t i = 0; i != m_srcs.size(); ++i)
      {
      sources.push_back(m_srcs[i]->name());
      }
   return sources;
   }

size_t Entropy_Sources::poll(RandomNumberGenerator& rng,
                             size_t poll_bits,
                             std::chrono::milliseconds timeout)
   {
   typedef std::chrono::system_clock clock;

   auto deadline = clock::now() + timeout;

   size_t bits_collected = 0;

   for(size_t i = 0; i != m_srcs.size(); ++i)
      {
      bits_collected += m_srcs[i]->poll(rng);

      if (bits_collected >= poll_bits || clock::now() > deadline)
         break;
      }

   return bits_collected;
   }

size_t Entropy_Sources::poll_just(RandomNumberGenerator& rng, const std::string& the_src)
   {
   for(size_t i = 0; i != m_srcs.size(); ++i)
      {
      if(m_srcs[i]->name() == the_src)
         {
         return m_srcs[i]->poll(rng);
         }
      }

   return 0;
   }

Entropy_Sources::Entropy_Sources(const std::vector<std::string>& sources)
   {
   for(auto&& src_name : sources)
      {
      add_source(Entropy_Source::create(src_name));
      }
   }

Entropy_Sources& Entropy_Sources::global_sources()
   {
   static Entropy_Sources global_entropy_sources(BOTAN_ENTROPY_DEFAULT_SOURCES);

   return global_entropy_sources;
   }

}

/*
* Hash Functions
* (C) 2015 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_ADLER32)
#endif

#if defined(BOTAN_HAS_CRC24)
#endif

#if defined(BOTAN_HAS_CRC32)
#endif

#if defined(BOTAN_HAS_GOST_34_11)
#endif

#if defined(BOTAN_HAS_KECCAK)
#endif

#if defined(BOTAN_HAS_MD4)
#endif

#if defined(BOTAN_HAS_MD5)
#endif

#if defined(BOTAN_HAS_RIPEMD_160)
#endif

#if defined(BOTAN_HAS_SHA1)
#endif

#if defined(BOTAN_HAS_SHA2_32)
#endif

#if defined(BOTAN_HAS_SHA2_64)
#endif

#if defined(BOTAN_HAS_SHA3)
#endif

#if defined(BOTAN_HAS_SHAKE)
#endif

#if defined(BOTAN_HAS_SKEIN_512)
#endif

#if defined(BOTAN_HAS_STREEBOG)
#endif

#if defined(BOTAN_HAS_SM3)
#endif

#if defined(BOTAN_HAS_TIGER)
#endif

#if defined(BOTAN_HAS_WHIRLPOOL)
#endif

#if defined(BOTAN_HAS_PARALLEL_HASH)
#endif

#if defined(BOTAN_HAS_COMB4P)
#endif

#if defined(BOTAN_HAS_BLAKE2B)
#endif

#if defined(BOTAN_HAS_OPENSSL)
#endif

#if defined(BOTAN_HAS_COMMONCRYPTO)
#endif

namespace Botan {

std::unique_ptr<HashFunction> HashFunction::create(const std::string& algo_spec,
                                                   const std::string& provider)
   {

#if defined(BOTAN_HAS_COMMONCRYPTO)
   if(provider.empty() || provider == "commoncrypto")
      {
      if(auto hash = make_commoncrypto_hash(algo_spec))
         return hash;

      if(!provider.empty())
         return nullptr;
      }
#endif

#if defined(BOTAN_HAS_OPENSSL)
   if(provider.empty() || provider == "openssl")
      {
      if(auto hash = make_openssl_hash(algo_spec))
         return hash;

      if(!provider.empty())
         return nullptr;
      }
#endif

   if(provider.empty() == false && provider != "base")
      return nullptr; // unknown provider

#if defined(BOTAN_HAS_SHA1)
   if(algo_spec == "SHA-160" ||
      algo_spec == "SHA-1" ||
      algo_spec == "SHA1")
      {
      return std::unique_ptr<HashFunction>(new SHA_160);
      }
#endif

#if defined(BOTAN_HAS_SHA2_32)
   if(algo_spec == "SHA-224")
      {
      return std::unique_ptr<HashFunction>(new SHA_224);
      }

   if(algo_spec == "SHA-256")
      {
      return std::unique_ptr<HashFunction>(new SHA_256);
      }
#endif

#if defined(BOTAN_HAS_SHA2_64)
   if(algo_spec == "SHA-384")
      {
      return std::unique_ptr<HashFunction>(new SHA_384);
      }

   if(algo_spec == "SHA-512")
      {
      return std::unique_ptr<HashFunction>(new SHA_512);
      }

   if(algo_spec == "SHA-512-256")
      {
      return std::unique_ptr<HashFunction>(new SHA_512_256);
      }
#endif

#if defined(BOTAN_HAS_RIPEMD_160)
   if(algo_spec == "RIPEMD-160")
      {
      return std::unique_ptr<HashFunction>(new RIPEMD_160);
      }
#endif

#if defined(BOTAN_HAS_WHIRLPOOL)
   if(algo_spec == "Whirlpool")
      {
      return std::unique_ptr<HashFunction>(new Whirlpool);
      }
#endif

#if defined(BOTAN_HAS_MD5)
   if(algo_spec == "MD5")
      {
      return std::unique_ptr<HashFunction>(new MD5);
      }
#endif

#if defined(BOTAN_HAS_MD4)
   if(algo_spec == "MD4")
      {
      return std::unique_ptr<HashFunction>(new MD4);
      }
#endif

#if defined(BOTAN_HAS_GOST_34_11)
   if(algo_spec == "GOST-R-34.11-94" || algo_spec == "GOST-34.11")
      {
      return std::unique_ptr<HashFunction>(new GOST_34_11);
      }
#endif

#if defined(BOTAN_HAS_ADLER32)
   if(algo_spec == "Adler32")
      {
      return std::unique_ptr<HashFunction>(new Adler32);
      }
#endif

#if defined(BOTAN_HAS_CRC24)
   if(algo_spec == "CRC24")
      {
      return std::unique_ptr<HashFunction>(new CRC24);
      }
#endif

#if defined(BOTAN_HAS_CRC32)
   if(algo_spec == "CRC32")
      {
      return std::unique_ptr<HashFunction>(new CRC32);
      }
#endif

   const SCAN_Name req(algo_spec);

#if defined(BOTAN_HAS_TIGER)
   if(req.algo_name() == "Tiger")
      {
      return std::unique_ptr<HashFunction>(
         new Tiger(req.arg_as_integer(0, 24),
                   req.arg_as_integer(1, 3)));
      }
#endif

#if defined(BOTAN_HAS_SKEIN_512)
   if(req.algo_name() == "Skein-512")
      {
      return std::unique_ptr<HashFunction>(
         new Skein_512(req.arg_as_integer(0, 512), req.arg(1, "")));
      }
#endif

#if defined(BOTAN_HAS_BLAKE2B)
   if(req.algo_name() == "Blake2b" || req.algo_name() == "BLAKE2b")
      {
      return std::unique_ptr<HashFunction>(
         new Blake2b(req.arg_as_integer(0, 512)));
   }
#endif

#if defined(BOTAN_HAS_KECCAK)
   if(req.algo_name() == "Keccak-1600")
      {
      return std::unique_ptr<HashFunction>(
         new Keccak_1600(req.arg_as_integer(0, 512)));
      }
#endif

#if defined(BOTAN_HAS_SHA3)
   if(req.algo_name() == "SHA-3")
      {
      return std::unique_ptr<HashFunction>(
         new SHA_3(req.arg_as_integer(0, 512)));
      }
#endif

#if defined(BOTAN_HAS_SHAKE)
   if(req.algo_name() == "SHAKE-128")
      {
      return std::unique_ptr<HashFunction>(new SHAKE_128(req.arg_as_integer(0, 128)));
      }
   if(req.algo_name() == "SHAKE-256")
      {
      return std::unique_ptr<HashFunction>(new SHAKE_256(req.arg_as_integer(0, 256)));
      }
#endif

#if defined(BOTAN_HAS_STREEBOG)
   if(algo_spec == "Streebog-256")
      {
      return std::unique_ptr<HashFunction>(new Streebog_256);
      }
   if(algo_spec == "Streebog-512")
      {
      return std::unique_ptr<HashFunction>(new Streebog_512);
      }
#endif

#if defined(BOTAN_HAS_SM3)
   if(algo_spec == "SM3")
      {
      return std::unique_ptr<HashFunction>(new SM3);
      }
#endif

#if defined(BOTAN_HAS_WHIRLPOOL)
   if(req.algo_name() == "Whirlpool")
      {
      return std::unique_ptr<HashFunction>(new Whirlpool);
      }
#endif

#if defined(BOTAN_HAS_PARALLEL_HASH)
   if(req.algo_name() == "Parallel")
      {
      std::vector<std::unique_ptr<HashFunction>> hashes;

      for(size_t i = 0; i != req.arg_count(); ++i)
         {
         auto h = HashFunction::create(req.arg(i));
         if(!h)
            {
            return nullptr;
            }
         hashes.push_back(std::move(h));
         }

      return std::unique_ptr<HashFunction>(new Parallel(hashes));
      }
#endif

#if defined(BOTAN_HAS_COMB4P)
   if(req.algo_name() == "Comb4P" && req.arg_count() == 2)
      {
      std::unique_ptr<HashFunction> h1(HashFunction::create(req.arg(0)));
      std::unique_ptr<HashFunction> h2(HashFunction::create(req.arg(1)));

      if(h1 && h2)
         return std::unique_ptr<HashFunction>(new Comb4P(h1.release(), h2.release()));
      }
#endif


   return nullptr;
   }

//static
std::unique_ptr<HashFunction>
HashFunction::create_or_throw(const std::string& algo,
                              const std::string& provider)
   {
   if(auto hash = HashFunction::create(algo, provider))
      {
      return hash;
      }
   throw Lookup_Error("Hash", algo, provider);
   }

std::vector<std::string> HashFunction::providers(const std::string& algo_spec)
   {
   return probe_providers_of<HashFunction>(algo_spec, {"base", "openssl", "commoncrypto"});
   }

}

/*
* Hex Encoding and Decoding
* (C) 2010,2020 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

char hex_encode_nibble(uint8_t n, bool uppercase)
   {
   BOTAN_DEBUG_ASSERT(n <= 15);

   const auto in_09 = CT::Mask<uint8_t>::is_lt(n, 10);

   const char c_09 = n + '0';
   const char c_af = n + (uppercase ? 'A' : 'a') - 10;

   return in_09.select(c_09, c_af);
   }

}

void hex_encode(char output[],
                const uint8_t input[],
                size_t input_length,
                bool uppercase)
   {
   for(size_t i = 0; i != input_length; ++i)
      {
      const uint8_t n0 = (input[i] >> 4) & 0xF;
      const uint8_t n1 = (input[i]     ) & 0xF;

      output[2*i  ] = hex_encode_nibble(n0, uppercase);
      output[2*i+1] = hex_encode_nibble(n1, uppercase);
      }
   }

std::string hex_encode(const uint8_t input[],
                       size_t input_length,
                       bool uppercase)
   {
   std::string output(2 * input_length, 0);

   if(input_length)
      hex_encode(&output.front(), input, input_length, uppercase);

   return output;
   }

namespace {

uint8_t hex_char_to_bin(char input)
   {
   const uint8_t c = static_cast<uint8_t>(input);

   const auto is_alpha_upper = CT::Mask<uint8_t>::is_within_range(c, uint8_t('A'), uint8_t('F'));
   const auto is_alpha_lower = CT::Mask<uint8_t>::is_within_range(c, uint8_t('a'), uint8_t('f'));
   const auto is_decimal     = CT::Mask<uint8_t>::is_within_range(c, uint8_t('0'), uint8_t('9'));

   const auto is_whitespace  = CT::Mask<uint8_t>::is_any_of(c, {
         uint8_t(' '), uint8_t('\t'), uint8_t('\n'), uint8_t('\r')
      });

   const uint8_t c_upper = c - uint8_t('A') + 10;
   const uint8_t c_lower = c - uint8_t('a') + 10;
   const uint8_t c_decim = c - uint8_t('0');

   uint8_t ret = 0xFF; // default value

   ret = is_alpha_upper.select(c_upper, ret);
   ret = is_alpha_lower.select(c_lower, ret);
   ret = is_decimal.select(c_decim, ret);
   ret = is_whitespace.select(0x80, ret);

   return ret;
   }

}


size_t hex_decode(uint8_t output[],
                  const char input[],
                  size_t input_length,
                  size_t& input_consumed,
                  bool ignore_ws)
   {
   uint8_t* out_ptr = output;
   bool top_nibble = true;

   clear_mem(output, input_length / 2);

   for(size_t i = 0; i != input_length; ++i)
      {
      const uint8_t bin = hex_char_to_bin(input[i]);

      if(bin >= 0x10)
         {
         if(bin == 0x80 && ignore_ws)
            continue;

         std::string bad_char(1, input[i]);
         if(bad_char == "\t")
           bad_char = "\\t";
         else if(bad_char == "\n")
           bad_char = "\\n";

         throw Invalid_Argument(
           std::string("hex_decode: invalid hex character '") +
           bad_char + "'");
         }

      if(top_nibble)
         *out_ptr |= bin << 4;
      else
         *out_ptr |= bin;

      top_nibble = !top_nibble;
      if(top_nibble)
         ++out_ptr;
      }

   input_consumed = input_length;
   size_t written = (out_ptr - output);

   /*
   * We only got half of a uint8_t at the end; zap the half-written
   * output and mark it as unread
   */
   if(!top_nibble)
      {
      *out_ptr = 0;
      input_consumed -= 1;
      }

   return written;
   }

size_t hex_decode(uint8_t output[],
                  const char input[],
                  size_t input_length,
                  bool ignore_ws)
   {
   size_t consumed = 0;
   size_t written = hex_decode(output, input, input_length,
                               consumed, ignore_ws);

   if(consumed != input_length)
      throw Invalid_Argument("hex_decode: input did not have full bytes");

   return written;
   }

size_t hex_decode(uint8_t output[],
                  const std::string& input,
                  bool ignore_ws)
   {
   return hex_decode(output, input.data(), input.length(), ignore_ws);
   }

secure_vector<uint8_t> hex_decode_locked(const char input[],
                                      size_t input_length,
                                      bool ignore_ws)
   {
   secure_vector<uint8_t> bin(1 + input_length / 2);

   size_t written = hex_decode(bin.data(),
                               input,
                               input_length,
                               ignore_ws);

   bin.resize(written);
   return bin;
   }

secure_vector<uint8_t> hex_decode_locked(const std::string& input,
                                      bool ignore_ws)
   {
   return hex_decode_locked(input.data(), input.size(), ignore_ws);
   }

std::vector<uint8_t> hex_decode(const char input[],
                             size_t input_length,
                             bool ignore_ws)
   {
   std::vector<uint8_t> bin(1 + input_length / 2);

   size_t written = hex_decode(bin.data(),
                               input,
                               input_length,
                               ignore_ws);

   bin.resize(written);
   return bin;
   }

std::vector<uint8_t> hex_decode(const std::string& input,
                             bool ignore_ws)
   {
   return hex_decode(input.data(), input.size(), ignore_ws);
   }

}
/*
* KDF Retrieval
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_HKDF)
#endif

#if defined(BOTAN_HAS_KDF1)
#endif

#if defined(BOTAN_HAS_KDF2)
#endif

#if defined(BOTAN_HAS_KDF1_18033)
#endif

#if defined(BOTAN_HAS_TLS_V10_PRF) || defined(BOTAN_HAS_TLS_V12_PRF)
#endif

#if defined(BOTAN_HAS_X942_PRF)
#endif

#if defined(BOTAN_HAS_SP800_108)
#endif

#if defined(BOTAN_HAS_SP800_56A)
#endif

#if defined(BOTAN_HAS_SP800_56C)
#endif

namespace Botan {

namespace {

template<typename KDF_Type>
std::unique_ptr<KDF>
kdf_create_mac_or_hash(const std::string& nm)
   {
   if(auto mac = MessageAuthenticationCode::create(nm))
      return std::unique_ptr<KDF>(new KDF_Type(mac.release()));

   if(auto mac = MessageAuthenticationCode::create("HMAC(" + nm + ")"))
      return std::unique_ptr<KDF>(new KDF_Type(mac.release()));

   return nullptr;
   }

}

std::unique_ptr<KDF> KDF::create(const std::string& algo_spec,
                                 const std::string& provider)
   {
   const SCAN_Name req(algo_spec);

#if defined(BOTAN_HAS_HKDF)
   if(req.algo_name() == "HKDF" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         return kdf_create_mac_or_hash<HKDF>(req.arg(0));
         }
      }

   if(req.algo_name() == "HKDF-Extract" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         return kdf_create_mac_or_hash<HKDF_Extract>(req.arg(0));
         }
      }

   if(req.algo_name() == "HKDF-Expand" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         return kdf_create_mac_or_hash<HKDF_Expand>(req.arg(0));
         }
      }
#endif

#if defined(BOTAN_HAS_KDF2)
   if(req.algo_name() == "KDF2" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         if(auto hash = HashFunction::create(req.arg(0)))
            return std::unique_ptr<KDF>(new KDF2(hash.release()));
         }
      }
#endif

#if defined(BOTAN_HAS_KDF1_18033)
   if(req.algo_name() == "KDF1-18033" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         if(auto hash = HashFunction::create(req.arg(0)))
            return std::unique_ptr<KDF>(new KDF1_18033(hash.release()));
         }
      }
#endif

#if defined(BOTAN_HAS_KDF1)
   if(req.algo_name() == "KDF1" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         if(auto hash = HashFunction::create(req.arg(0)))
            return std::unique_ptr<KDF>(new KDF1(hash.release()));
         }
      }
#endif

#if defined(BOTAN_HAS_TLS_V10_PRF)
   if(req.algo_name() == "TLS-PRF" && req.arg_count() == 0)
      {
      if(provider.empty() || provider == "base")
         {
         auto hmac_md5 = MessageAuthenticationCode::create("HMAC(MD5)");
         auto hmac_sha1 = MessageAuthenticationCode::create("HMAC(SHA-1)");

         if(hmac_md5 && hmac_sha1)
            return std::unique_ptr<KDF>(new TLS_PRF(std::move(hmac_md5), std::move(hmac_sha1)));
         }
      }
#endif

#if defined(BOTAN_HAS_TLS_V12_PRF)
   if(req.algo_name() == "TLS-12-PRF" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         return kdf_create_mac_or_hash<TLS_12_PRF>(req.arg(0));
         }
      }
#endif

#if defined(BOTAN_HAS_X942_PRF)
   if(req.algo_name() == "X9.42-PRF" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         return std::unique_ptr<KDF>(new X942_PRF(req.arg(0)));
         }
      }
#endif

#if defined(BOTAN_HAS_SP800_108)
   if(req.algo_name() == "SP800-108-Counter" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         return kdf_create_mac_or_hash<SP800_108_Counter>(req.arg(0));
         }
      }

   if(req.algo_name() == "SP800-108-Feedback" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         return kdf_create_mac_or_hash<SP800_108_Feedback>(req.arg(0));
         }
      }

   if(req.algo_name() == "SP800-108-Pipeline" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         return kdf_create_mac_or_hash<SP800_108_Pipeline>(req.arg(0));
         }
      }
#endif

#if defined(BOTAN_HAS_SP800_56A)
   if(req.algo_name() == "SP800-56A" && req.arg_count() == 1)
      {
      if(auto hash = HashFunction::create(req.arg(0)))
         return std::unique_ptr<KDF>(new SP800_56A_Hash(hash.release()));
      if(auto mac = MessageAuthenticationCode::create(req.arg(0)))
         return std::unique_ptr<KDF>(new SP800_56A_HMAC(mac.release()));
      }
#endif

#if defined(BOTAN_HAS_SP800_56C)
   if(req.algo_name() == "SP800-56C" && req.arg_count() == 1)
      {
      std::unique_ptr<KDF> exp(kdf_create_mac_or_hash<SP800_108_Feedback>(req.arg(0)));
      if(exp)
         {
         if(auto mac = MessageAuthenticationCode::create(req.arg(0)))
            return std::unique_ptr<KDF>(new SP800_56C(mac.release(), exp.release()));

         if(auto mac = MessageAuthenticationCode::create("HMAC(" + req.arg(0) + ")"))
            return std::unique_ptr<KDF>(new SP800_56C(mac.release(), exp.release()));
         }
      }
#endif

   BOTAN_UNUSED(req);
   BOTAN_UNUSED(provider);

   return nullptr;
   }

//static
std::unique_ptr<KDF>
KDF::create_or_throw(const std::string& algo,
                             const std::string& provider)
   {
   if(auto kdf = KDF::create(algo, provider))
      {
      return kdf;
      }
   throw Lookup_Error("KDF", algo, provider);
   }

std::vector<std::string> KDF::providers(const std::string& algo_spec)
   {
   return probe_providers_of<KDF>(algo_spec, { "base" });
   }

KDF* get_kdf(const std::string& algo_spec)
   {
   SCAN_Name request(algo_spec);

   if(request.algo_name() == "Raw")
      return nullptr; // No KDF

   //return KDF::create_or_throw(algo_spec).release();
   auto kdf = KDF::create(algo_spec);
   if(!kdf)
      throw Algorithm_Not_Found(algo_spec);
   return kdf.release();
   }

}
/*
* Keypair Checks
* (C) 1999-2010 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace KeyPair {

/*
* Check an encryption key pair for consistency
*/
bool encryption_consistency_check(RandomNumberGenerator& rng,
                                  const Private_Key& private_key,
                                  const Public_Key& public_key,
                                  const std::string& padding)
   {
   PK_Encryptor_EME encryptor(public_key, rng, padding);
   PK_Decryptor_EME decryptor(private_key, rng, padding);

   /*
   Weird corner case, if the key is too small to encrypt anything at
   all. This can happen with very small RSA keys with PSS
   */
   if(encryptor.maximum_input_size() == 0)
      return true;

   std::vector<uint8_t> plaintext;
   rng.random_vec(plaintext, encryptor.maximum_input_size() - 1);

   std::vector<uint8_t> ciphertext = encryptor.encrypt(plaintext, rng);
   if(ciphertext == plaintext)
      return false;

   std::vector<uint8_t> decrypted = unlock(decryptor.decrypt(ciphertext));

   return (plaintext == decrypted);
   }

/*
* Check a signature key pair for consistency
*/
bool signature_consistency_check(RandomNumberGenerator& rng,
                                 const Private_Key& private_key,
                                 const Public_Key& public_key,
                                 const std::string& padding)
   {
   PK_Signer signer(private_key, rng, padding);
   PK_Verifier verifier(public_key, padding);

   std::vector<uint8_t> message(32);
   rng.randomize(message.data(), message.size());

   std::vector<uint8_t> signature;

   try
      {
      signature = signer.sign_message(message, rng);
      }
   catch(Encoding_Error&)
      {
      return false;
      }

   if(!verifier.verify_message(message, signature))
      return false;

   // Now try to check a corrupt signature, ensure it does not succeed
   ++signature[0];

   if(verifier.verify_message(message, signature))
      return false;

   return true;
   }

}

}
/*
* Message Authentication Code base class
* (C) 1999-2008 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_CBC_MAC)
#endif

#if defined(BOTAN_HAS_CMAC)
#endif

#if defined(BOTAN_HAS_GMAC)
#endif

#if defined(BOTAN_HAS_HMAC)
#endif

#if defined(BOTAN_HAS_POLY1305)
#endif

#if defined(BOTAN_HAS_SIPHASH)
#endif

#if defined(BOTAN_HAS_ANSI_X919_MAC)
#endif

namespace Botan {

std::unique_ptr<MessageAuthenticationCode>
MessageAuthenticationCode::create(const std::string& algo_spec,
                                  const std::string& provider)
   {
   const SCAN_Name req(algo_spec);

#if defined(BOTAN_HAS_GMAC)
   if(req.algo_name() == "GMAC" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         if(auto bc = BlockCipher::create(req.arg(0)))
            return std::unique_ptr<MessageAuthenticationCode>(new GMAC(bc.release()));
         }
      }
#endif

#if defined(BOTAN_HAS_HMAC)
   if(req.algo_name() == "HMAC" && req.arg_count() == 1)
      {
      // TODO OpenSSL
      if(provider.empty() || provider == "base")
         {
         if(auto h = HashFunction::create(req.arg(0)))
            return std::unique_ptr<MessageAuthenticationCode>(new HMAC(h.release()));
         }
      }
#endif

#if defined(BOTAN_HAS_POLY1305)
   if(req.algo_name() == "Poly1305" && req.arg_count() == 0)
      {
      if(provider.empty() || provider == "base")
         return std::unique_ptr<MessageAuthenticationCode>(new Poly1305);
      }
#endif

#if defined(BOTAN_HAS_SIPHASH)
   if(req.algo_name() == "SipHash")
      {
      if(provider.empty() || provider == "base")
         {
         return std::unique_ptr<MessageAuthenticationCode>(
            new SipHash(req.arg_as_integer(0, 2), req.arg_as_integer(1, 4)));
         }
      }
#endif

#if defined(BOTAN_HAS_CMAC)
   if((req.algo_name() == "CMAC" || req.algo_name() == "OMAC") && req.arg_count() == 1)
      {
      // TODO: OpenSSL CMAC
      if(provider.empty() || provider == "base")
         {
         if(auto bc = BlockCipher::create(req.arg(0)))
            return std::unique_ptr<MessageAuthenticationCode>(new CMAC(bc.release()));
         }
      }
#endif


#if defined(BOTAN_HAS_CBC_MAC)
   if(req.algo_name() == "CBC-MAC" && req.arg_count() == 1)
      {
      if(provider.empty() || provider == "base")
         {
         if(auto bc = BlockCipher::create(req.arg(0)))
            return std::unique_ptr<MessageAuthenticationCode>(new CBC_MAC(bc.release()));
         }
      }
#endif

#if defined(BOTAN_HAS_ANSI_X919_MAC)
   if(req.algo_name() == "X9.19-MAC")
      {
      if(provider.empty() || provider == "base")
         {
         return std::unique_ptr<MessageAuthenticationCode>(new ANSI_X919_MAC);
         }
      }
#endif

   BOTAN_UNUSED(req);
   BOTAN_UNUSED(provider);

   return nullptr;
   }

std::vector<std::string>
MessageAuthenticationCode::providers(const std::string& algo_spec)
   {
   return probe_providers_of<MessageAuthenticationCode>(algo_spec, {"base", "openssl"});
   }

//static
std::unique_ptr<MessageAuthenticationCode>
MessageAuthenticationCode::create_or_throw(const std::string& algo,
                                           const std::string& provider)
   {
   if(auto mac = MessageAuthenticationCode::create(algo, provider))
      {
      return mac;
      }
   throw Lookup_Error("MAC", algo, provider);
   }

void MessageAuthenticationCode::start_msg(const uint8_t nonce[], size_t nonce_len)
   {
   BOTAN_UNUSED(nonce);
   if(nonce_len > 0)
      throw Invalid_IV_Length(name(), nonce_len);
   }

/*
* Default (deterministic) MAC verification operation
*/
bool MessageAuthenticationCode::verify_mac(const uint8_t mac[], size_t length)
   {
   secure_vector<uint8_t> our_mac = final();

   if(our_mac.size() != length)
      return false;

   return constant_time_compare(our_mac.data(), mac, length);
   }

}
/*
* MD5
* (C) 1999-2008 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

std::unique_ptr<HashFunction> MD5::copy_state() const
   {
   return std::unique_ptr<HashFunction>(new MD5(*this));
   }

namespace {

/*
* MD5 FF Function
*/
template<size_t S>
inline void FF(uint32_t& A, uint32_t B, uint32_t C, uint32_t D, uint32_t M)
   {
   A += (D ^ (B & (C ^ D))) + M;
   A  = rotl<S>(A) + B;
   }

/*
* MD5 GG Function
*/
template<size_t S>
inline void GG(uint32_t& A, uint32_t B, uint32_t C, uint32_t D, uint32_t M)
   {
   A += (C ^ (D & (B ^ C))) + M;
   A  = rotl<S>(A) + B;
   }

/*
* MD5 HH Function
*/
template<size_t S>
inline void HH(uint32_t& A, uint32_t B, uint32_t C, uint32_t D, uint32_t M)
   {
   A += (B ^ C ^ D) + M;
   A  = rotl<S>(A) + B;
   }

/*
* MD5 II Function
*/
template<size_t S>
inline void II(uint32_t& A, uint32_t B, uint32_t C, uint32_t D, uint32_t M)
   {
   A += (C ^ (B | ~D)) + M;
   A  = rotl<S>(A) + B;
   }

}

/*
* MD5 Compression Function
*/
void MD5::compress_n(const uint8_t input[], size_t blocks)
   {
   uint32_t A = m_digest[0], B = m_digest[1], C = m_digest[2], D = m_digest[3];

   for(size_t i = 0; i != blocks; ++i)
      {
      load_le(m_M.data(), input, m_M.size());

      FF< 7>(A,B,C,D,m_M[ 0]+0xD76AA478);   FF<12>(D,A,B,C,m_M[ 1]+0xE8C7B756);
      FF<17>(C,D,A,B,m_M[ 2]+0x242070DB);   FF<22>(B,C,D,A,m_M[ 3]+0xC1BDCEEE);
      FF< 7>(A,B,C,D,m_M[ 4]+0xF57C0FAF);   FF<12>(D,A,B,C,m_M[ 5]+0x4787C62A);
      FF<17>(C,D,A,B,m_M[ 6]+0xA8304613);   FF<22>(B,C,D,A,m_M[ 7]+0xFD469501);
      FF< 7>(A,B,C,D,m_M[ 8]+0x698098D8);   FF<12>(D,A,B,C,m_M[ 9]+0x8B44F7AF);
      FF<17>(C,D,A,B,m_M[10]+0xFFFF5BB1);   FF<22>(B,C,D,A,m_M[11]+0x895CD7BE);
      FF< 7>(A,B,C,D,m_M[12]+0x6B901122);   FF<12>(D,A,B,C,m_M[13]+0xFD987193);
      FF<17>(C,D,A,B,m_M[14]+0xA679438E);   FF<22>(B,C,D,A,m_M[15]+0x49B40821);

      GG< 5>(A,B,C,D,m_M[ 1]+0xF61E2562);   GG< 9>(D,A,B,C,m_M[ 6]+0xC040B340);
      GG<14>(C,D,A,B,m_M[11]+0x265E5A51);   GG<20>(B,C,D,A,m_M[ 0]+0xE9B6C7AA);
      GG< 5>(A,B,C,D,m_M[ 5]+0xD62F105D);   GG< 9>(D,A,B,C,m_M[10]+0x02441453);
      GG<14>(C,D,A,B,m_M[15]+0xD8A1E681);   GG<20>(B,C,D,A,m_M[ 4]+0xE7D3FBC8);
      GG< 5>(A,B,C,D,m_M[ 9]+0x21E1CDE6);   GG< 9>(D,A,B,C,m_M[14]+0xC33707D6);
      GG<14>(C,D,A,B,m_M[ 3]+0xF4D50D87);   GG<20>(B,C,D,A,m_M[ 8]+0x455A14ED);
      GG< 5>(A,B,C,D,m_M[13]+0xA9E3E905);   GG< 9>(D,A,B,C,m_M[ 2]+0xFCEFA3F8);
      GG<14>(C,D,A,B,m_M[ 7]+0x676F02D9);   GG<20>(B,C,D,A,m_M[12]+0x8D2A4C8A);

      HH< 4>(A,B,C,D,m_M[ 5]+0xFFFA3942);   HH<11>(D,A,B,C,m_M[ 8]+0x8771F681);
      HH<16>(C,D,A,B,m_M[11]+0x6D9D6122);   HH<23>(B,C,D,A,m_M[14]+0xFDE5380C);
      HH< 4>(A,B,C,D,m_M[ 1]+0xA4BEEA44);   HH<11>(D,A,B,C,m_M[ 4]+0x4BDECFA9);
      HH<16>(C,D,A,B,m_M[ 7]+0xF6BB4B60);   HH<23>(B,C,D,A,m_M[10]+0xBEBFBC70);
      HH< 4>(A,B,C,D,m_M[13]+0x289B7EC6);   HH<11>(D,A,B,C,m_M[ 0]+0xEAA127FA);
      HH<16>(C,D,A,B,m_M[ 3]+0xD4EF3085);   HH<23>(B,C,D,A,m_M[ 6]+0x04881D05);
      HH< 4>(A,B,C,D,m_M[ 9]+0xD9D4D039);   HH<11>(D,A,B,C,m_M[12]+0xE6DB99E5);
      HH<16>(C,D,A,B,m_M[15]+0x1FA27CF8);   HH<23>(B,C,D,A,m_M[ 2]+0xC4AC5665);

      II< 6>(A,B,C,D,m_M[ 0]+0xF4292244);   II<10>(D,A,B,C,m_M[ 7]+0x432AFF97);
      II<15>(C,D,A,B,m_M[14]+0xAB9423A7);   II<21>(B,C,D,A,m_M[ 5]+0xFC93A039);
      II< 6>(A,B,C,D,m_M[12]+0x655B59C3);   II<10>(D,A,B,C,m_M[ 3]+0x8F0CCC92);
      II<15>(C,D,A,B,m_M[10]+0xFFEFF47D);   II<21>(B,C,D,A,m_M[ 1]+0x85845DD1);
      II< 6>(A,B,C,D,m_M[ 8]+0x6FA87E4F);   II<10>(D,A,B,C,m_M[15]+0xFE2CE6E0);
      II<15>(C,D,A,B,m_M[ 6]+0xA3014314);   II<21>(B,C,D,A,m_M[13]+0x4E0811A1);
      II< 6>(A,B,C,D,m_M[ 4]+0xF7537E82);   II<10>(D,A,B,C,m_M[11]+0xBD3AF235);
      II<15>(C,D,A,B,m_M[ 2]+0x2AD7D2BB);   II<21>(B,C,D,A,m_M[ 9]+0xEB86D391);

      A = (m_digest[0] += A);
      B = (m_digest[1] += B);
      C = (m_digest[2] += C);
      D = (m_digest[3] += D);

      input += hash_block_size();
      }
   }

/*
* Copy out the digest
*/
void MD5::copy_out(uint8_t output[])
   {
   copy_out_vec_le(output, output_length(), m_digest);
   }

/*
* Clear memory of sensitive data
*/
void MD5::clear()
   {
   MDx_HashFunction::clear();
   zeroise(m_M);
   m_digest[0] = 0x67452301;
   m_digest[1] = 0xEFCDAB89;
   m_digest[2] = 0x98BADCFE;
   m_digest[3] = 0x10325476;
   }

}
/*
* Merkle-Damgard Hash Function
* (C) 1999-2008,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/*
* MDx_HashFunction Constructor
*/
MDx_HashFunction::MDx_HashFunction(size_t block_len,
                                   bool byte_big_endian,
                                   bool bit_big_endian,
                                   uint8_t cnt_size) :
   m_pad_char(bit_big_endian == true ? 0x80 : 0x01),
   m_counter_size(cnt_size),
   m_block_bits(ceil_log2(block_len)),
   m_count_big_endian(byte_big_endian),
   m_count(0),
   m_buffer(block_len),
   m_position(0)
   {
   if(!is_power_of_2(block_len))
      throw Invalid_Argument("MDx_HashFunction block length must be a power of 2");
   if(m_block_bits < 3 || m_block_bits > 16)
      throw Invalid_Argument("MDx_HashFunction block size too large or too small");
   if(m_counter_size < 8 || m_counter_size > block_len)
      throw Invalid_State("MDx_HashFunction invalid counter length");
   }

/*
* Clear memory of sensitive data
*/
void MDx_HashFunction::clear()
   {
   zeroise(m_buffer);
   m_count = m_position = 0;
   }

/*
* Update the hash
*/
void MDx_HashFunction::add_data(const uint8_t input[], size_t length)
   {
   const size_t block_len = static_cast<size_t>(1) << m_block_bits;

   m_count += length;

   if(m_position)
      {
      buffer_insert(m_buffer, m_position, input, length);

      if(m_position + length >= block_len)
         {
         compress_n(m_buffer.data(), 1);
         input += (block_len - m_position);
         length -= (block_len - m_position);
         m_position = 0;
         }
      }

   // Just in case the compiler can't figure out block_len is a power of 2
   const size_t full_blocks = length >> m_block_bits;
   const size_t remaining   = length & (block_len - 1);

   if(full_blocks > 0)
      {
      compress_n(input, full_blocks);
      }

   buffer_insert(m_buffer, m_position, input + full_blocks * block_len, remaining);
   m_position += remaining;
   }

/*
* Finalize a hash
*/
void MDx_HashFunction::final_result(uint8_t output[])
   {
   const size_t block_len = static_cast<size_t>(1) << m_block_bits;

   clear_mem(&m_buffer[m_position], block_len - m_position);
   m_buffer[m_position] = m_pad_char;

   if(m_position >= block_len - m_counter_size)
      {
      compress_n(m_buffer.data(), 1);
      zeroise(m_buffer);
      }

   write_count(&m_buffer[block_len - m_counter_size]);

   compress_n(m_buffer.data(), 1);
   copy_out(output);
   clear();
   }

/*
* Write the count bits to the buffer
*/
void MDx_HashFunction::write_count(uint8_t out[])
   {
   BOTAN_ASSERT_NOMSG(m_counter_size <= output_length());
   BOTAN_ASSERT_NOMSG(m_counter_size >= 8);

   const uint64_t bit_count = m_count * 8;

   if(m_count_big_endian)
      store_be(bit_count, out + m_counter_size - 8);
   else
      store_le(bit_count, out + m_counter_size - 8);
   }

}
/*
* CBC Padding Methods
* (C) 1999-2007,2013,2018,2020 Jack Lloyd
* (C) 2016 René Korthaus, Rohde & Schwarz Cybersecurity
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/**
* Get a block cipher padding method by name
*/
BlockCipherModePaddingMethod* get_bc_pad(const std::string& algo_spec)
   {
   if(algo_spec == "NoPadding")
      return new Null_Padding;

   if(algo_spec == "PKCS7")
      return new PKCS7_Padding;

   if(algo_spec == "OneAndZeros")
      return new OneAndZeros_Padding;

   if(algo_spec == "X9.23")
      return new ANSI_X923_Padding;

   if(algo_spec == "ESP")
      return new ESP_Padding;

   return nullptr;
   }

/*
* Pad with PKCS #7 Method
*/
void PKCS7_Padding::add_padding(secure_vector<uint8_t>& buffer,
                                size_t last_byte_pos,
                                size_t BS) const
   {
   /*
   Padding format is
   01
   0202
   030303
   ...
   */
   BOTAN_DEBUG_ASSERT(last_byte_pos < BS);

   const uint8_t padding_len = static_cast<uint8_t>(BS - last_byte_pos);

   buffer.resize(buffer.size() + padding_len);

   CT::poison(&last_byte_pos, 1);
   CT::poison(buffer.data(), buffer.size());

   BOTAN_DEBUG_ASSERT(buffer.size() % BS == 0);
   BOTAN_DEBUG_ASSERT(buffer.size() >= BS);

   const size_t start_of_last_block = buffer.size() - BS;
   const size_t end_of_last_block = buffer.size();
   const size_t start_of_padding = buffer.size() - padding_len;

   for(size_t i = start_of_last_block; i != end_of_last_block; ++i)
      {
      auto needs_padding = CT::Mask<uint8_t>(CT::Mask<size_t>::is_gte(i, start_of_padding));
      buffer[i] = needs_padding.select(padding_len, buffer[i]);
      }

   CT::unpoison(buffer.data(), buffer.size());
   CT::unpoison(last_byte_pos);
   }

/*
* Unpad with PKCS #7 Method
*/
size_t PKCS7_Padding::unpad(const uint8_t input[], size_t input_length) const
   {
   if(!valid_blocksize(input_length))
      return input_length;

   CT::poison(input, input_length);

   const uint8_t last_byte = input[input_length-1];

   /*
   The input should == the block size so if the last byte exceeds
   that then the padding is certainly invalid
   */
   auto bad_input = CT::Mask<size_t>::is_gt(last_byte, input_length);

   const size_t pad_pos = input_length - last_byte;

   for(size_t i = 0; i != input_length - 1; ++i)
      {
      // Does this byte equal the expected pad byte?
      const auto pad_eq = CT::Mask<size_t>::is_equal(input[i], last_byte);

      // Ignore values that are not part of the padding
      const auto in_range = CT::Mask<size_t>::is_gte(i, pad_pos);
      bad_input |= in_range & (~pad_eq);
      }

   CT::unpoison(input, input_length);

   return bad_input.select_and_unpoison(input_length, pad_pos);
   }

/*
* Pad with ANSI X9.23 Method
*/
void ANSI_X923_Padding::add_padding(secure_vector<uint8_t>& buffer,
                                    size_t last_byte_pos,
                                    size_t BS) const
   {
   /*
   Padding format is
   01
   0002
   000003
   ...
   */
   BOTAN_DEBUG_ASSERT(last_byte_pos < BS);

   const uint8_t padding_len = static_cast<uint8_t>(BS - last_byte_pos);

   buffer.resize(buffer.size() + padding_len);

   CT::poison(&last_byte_pos, 1);
   CT::poison(buffer.data(), buffer.size());

   BOTAN_DEBUG_ASSERT(buffer.size() % BS == 0);
   BOTAN_DEBUG_ASSERT(buffer.size() >= BS);

   const size_t start_of_last_block = buffer.size() - BS;
   const size_t end_of_zero_padding = buffer.size() - 1;
   const size_t start_of_padding = buffer.size() - padding_len;

   for(size_t i = start_of_last_block; i != end_of_zero_padding; ++i)
      {
      auto needs_padding = CT::Mask<uint8_t>(CT::Mask<size_t>::is_gte(i, start_of_padding));
      buffer[i] = needs_padding.select(0, buffer[i]);
      }

   buffer[buffer.size()-1] = padding_len;
   CT::unpoison(buffer.data(), buffer.size());
   CT::unpoison(last_byte_pos);
   }

/*
* Unpad with ANSI X9.23 Method
*/
size_t ANSI_X923_Padding::unpad(const uint8_t input[], size_t input_length) const
   {
   if(!valid_blocksize(input_length))
      return input_length;

   CT::poison(input, input_length);

   const size_t last_byte = input[input_length-1];

   auto bad_input = CT::Mask<size_t>::is_gt(last_byte, input_length);

   const size_t pad_pos = input_length - last_byte;

   for(size_t i = 0; i != input_length - 1; ++i)
      {
      // Ignore values that are not part of the padding
      const auto in_range = CT::Mask<size_t>::is_gte(i, pad_pos);
      const auto pad_is_nonzero = CT::Mask<size_t>::expand(input[i]);
      bad_input |= pad_is_nonzero & in_range;
      }

   CT::unpoison(input, input_length);

   return bad_input.select_and_unpoison(input_length, pad_pos);
   }

/*
* Pad with One and Zeros Method
*/
void OneAndZeros_Padding::add_padding(secure_vector<uint8_t>& buffer,
                                      size_t last_byte_pos,
                                      size_t BS) const
   {
   /*
   Padding format is
   80
   8000
   800000
   ...
   */

   BOTAN_DEBUG_ASSERT(last_byte_pos < BS);

   const uint8_t padding_len = static_cast<uint8_t>(BS - last_byte_pos);

   buffer.resize(buffer.size() + padding_len);

   CT::poison(&last_byte_pos, 1);
   CT::poison(buffer.data(), buffer.size());

   BOTAN_DEBUG_ASSERT(buffer.size() % BS == 0);
   BOTAN_DEBUG_ASSERT(buffer.size() >= BS);

   const size_t start_of_last_block = buffer.size() - BS;
   const size_t end_of_last_block = buffer.size();
   const size_t start_of_padding = buffer.size() - padding_len;

   for(size_t i = start_of_last_block; i != end_of_last_block; ++i)
      {
      auto needs_80 = CT::Mask<uint8_t>(CT::Mask<size_t>::is_equal(i, start_of_padding));
      auto needs_00 = CT::Mask<uint8_t>(CT::Mask<size_t>::is_gt(i, start_of_padding));
      buffer[i] = needs_00.select(0x00, needs_80.select(0x80, buffer[i]));
      }

   CT::unpoison(buffer.data(), buffer.size());
   CT::unpoison(last_byte_pos);
   }

/*
* Unpad with One and Zeros Method
*/
size_t OneAndZeros_Padding::unpad(const uint8_t input[], size_t input_length) const
   {
   if(!valid_blocksize(input_length))
      return input_length;

   CT::poison(input, input_length);

   auto bad_input = CT::Mask<uint8_t>::cleared();
   auto seen_0x80 = CT::Mask<uint8_t>::cleared();

   size_t pad_pos = input_length - 1;
   size_t i = input_length;

   while(i)
      {
      const auto is_0x80 = CT::Mask<uint8_t>::is_equal(input[i-1], 0x80);
      const auto is_zero = CT::Mask<uint8_t>::is_zero(input[i-1]);

      seen_0x80 |= is_0x80;
      pad_pos -= seen_0x80.if_not_set_return(1);
      bad_input |= ~seen_0x80 & ~is_zero;
      i--;
      }
   bad_input |= ~seen_0x80;

   CT::unpoison(input, input_length);

   return CT::Mask<size_t>::expand(bad_input).select_and_unpoison(input_length, pad_pos);
   }

/*
* Pad with ESP Padding Method
*/
void ESP_Padding::add_padding(secure_vector<uint8_t>& buffer,
                              size_t last_byte_pos,
                              size_t BS) const
   {
   /*
   Padding format is
   01
   0102
   010203
   ...
   */
   BOTAN_DEBUG_ASSERT(last_byte_pos < BS);

   const uint8_t padding_len = static_cast<uint8_t>(BS - last_byte_pos);

   buffer.resize(buffer.size() + padding_len);

   CT::poison(&last_byte_pos, 1);
   CT::poison(buffer.data(), buffer.size());

   BOTAN_DEBUG_ASSERT(buffer.size() % BS == 0);
   BOTAN_DEBUG_ASSERT(buffer.size() >= BS);

   const size_t start_of_last_block = buffer.size() - BS;
   const size_t end_of_last_block = buffer.size();
   const size_t start_of_padding = buffer.size() - padding_len;

   uint8_t pad_ctr = 0x01;

   for(size_t i = start_of_last_block; i != end_of_last_block; ++i)
      {
      auto needs_padding = CT::Mask<uint8_t>(CT::Mask<size_t>::is_gte(i, start_of_padding));
      buffer[i] = needs_padding.select(pad_ctr, buffer[i]);
      pad_ctr = needs_padding.select(pad_ctr + 1, pad_ctr);
      }

   CT::unpoison(buffer.data(), buffer.size());
   CT::unpoison(last_byte_pos);
   }

/*
* Unpad with ESP Padding Method
*/
size_t ESP_Padding::unpad(const uint8_t input[], size_t input_length) const
   {
   if(!valid_blocksize(input_length))
      return input_length;

   CT::poison(input, input_length);

   const uint8_t input_length_8 = static_cast<uint8_t>(input_length);
   const uint8_t last_byte = input[input_length-1];

   auto bad_input = CT::Mask<uint8_t>::is_zero(last_byte) |
      CT::Mask<uint8_t>::is_gt(last_byte, input_length_8);

   const uint8_t pad_pos = input_length_8 - last_byte;
   size_t i = input_length_8 - 1;
   while(i)
      {
      const auto in_range = CT::Mask<size_t>::is_gt(i, pad_pos);
      const auto incrementing = CT::Mask<uint8_t>::is_equal(input[i-1], input[i]-1);

      bad_input |= CT::Mask<uint8_t>(in_range) & ~incrementing;
      --i;
      }

   CT::unpoison(input, input_length);
   return bad_input.select_and_unpoison(input_length_8, pad_pos);
   }


}
/*
* Cipher Modes
* (C) 2015 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_BLOCK_CIPHER)
#endif

#if defined(BOTAN_HAS_AEAD_MODES)
#endif

#if defined(BOTAN_HAS_MODE_CBC)
#endif

#if defined(BOTAN_HAS_MODE_CFB)
#endif

#if defined(BOTAN_HAS_MODE_XTS)
#endif

#if defined(BOTAN_HAS_OPENSSL)
#endif

#if defined(BOTAN_HAS_COMMONCRYPTO)
#endif

namespace Botan {

std::unique_ptr<Cipher_Mode> Cipher_Mode::create_or_throw(const std::string& algo,
                                                          Cipher_Dir direction,
                                                          const std::string& provider)
   {
   if(auto mode = Cipher_Mode::create(algo, direction, provider))
      return mode;

   throw Lookup_Error("Cipher mode", algo, provider);
   }

std::unique_ptr<Cipher_Mode> Cipher_Mode::create(const std::string& algo,
                                                 Cipher_Dir direction,
                                                 const std::string& provider)
   {
#if defined(BOTAN_HAS_COMMONCRYPTO)
   if(provider.empty() || provider == "commoncrypto")
      {
      std::unique_ptr<Cipher_Mode> commoncrypto_cipher(make_commoncrypto_cipher_mode(algo, direction));

      if(commoncrypto_cipher)
         return commoncrypto_cipher;

      if(!provider.empty())
         return std::unique_ptr<Cipher_Mode>();
      }
#endif

#if defined(BOTAN_HAS_OPENSSL)
   if(provider.empty() || provider == "openssl")
      {
      std::unique_ptr<Cipher_Mode> openssl_cipher(make_openssl_cipher_mode(algo, direction));

      if(openssl_cipher)
         return openssl_cipher;

      if(!provider.empty())
         return std::unique_ptr<Cipher_Mode>();
      }
#endif

#if defined(BOTAN_HAS_STREAM_CIPHER)
   if(auto sc = StreamCipher::create(algo))
      {
      return std::unique_ptr<Cipher_Mode>(new Stream_Cipher_Mode(sc.release()));
      }
#endif

#if defined(BOTAN_HAS_AEAD_MODES)
   if(auto aead = AEAD_Mode::create(algo, direction))
      {
      return std::unique_ptr<Cipher_Mode>(aead.release());
      }
#endif

   if(algo.find('/') != std::string::npos)
      {
      const std::vector<std::string> algo_parts = split_on(algo, '/');
      const std::string cipher_name = algo_parts[0];
      const std::vector<std::string> mode_info = parse_algorithm_name(algo_parts[1]);

      if(mode_info.empty())
         return std::unique_ptr<Cipher_Mode>();

      std::ostringstream alg_args;

      alg_args << '(' << cipher_name;
      for(size_t i = 1; i < mode_info.size(); ++i)
         alg_args << ',' << mode_info[i];
      for(size_t i = 2; i < algo_parts.size(); ++i)
         alg_args << ',' << algo_parts[i];
      alg_args << ')';

      const std::string mode_name = mode_info[0] + alg_args.str();
      return Cipher_Mode::create(mode_name, direction, provider);
      }

#if defined(BOTAN_HAS_BLOCK_CIPHER)

   SCAN_Name spec(algo);

   if(spec.arg_count() == 0)
      {
      return std::unique_ptr<Cipher_Mode>();
      }

   std::unique_ptr<BlockCipher> bc(BlockCipher::create(spec.arg(0), provider));

   if(!bc)
      {
      return std::unique_ptr<Cipher_Mode>();
      }

#if defined(BOTAN_HAS_MODE_CBC)
   if(spec.algo_name() == "CBC")
      {
      const std::string padding = spec.arg(1, "PKCS7");

      if(padding == "CTS")
         {
         if(direction == ENCRYPTION)
            return std::unique_ptr<Cipher_Mode>(new CTS_Encryption(bc.release()));
         else
            return std::unique_ptr<Cipher_Mode>(new CTS_Decryption(bc.release()));
         }
      else
         {
         std::unique_ptr<BlockCipherModePaddingMethod> pad(get_bc_pad(padding));

         if(pad)
            {
            if(direction == ENCRYPTION)
               return std::unique_ptr<Cipher_Mode>(new CBC_Encryption(bc.release(), pad.release()));
            else
               return std::unique_ptr<Cipher_Mode>(new CBC_Decryption(bc.release(), pad.release()));
            }
         }
      }
#endif

#if defined(BOTAN_HAS_MODE_XTS)
   if(spec.algo_name() == "XTS")
      {
      if(direction == ENCRYPTION)
         return std::unique_ptr<Cipher_Mode>(new XTS_Encryption(bc.release()));
      else
         return std::unique_ptr<Cipher_Mode>(new XTS_Decryption(bc.release()));
      }
#endif

#if defined(BOTAN_HAS_MODE_CFB)
   if(spec.algo_name() == "CFB")
      {
      const size_t feedback_bits = spec.arg_as_integer(1, 8*bc->block_size());
      if(direction == ENCRYPTION)
         return std::unique_ptr<Cipher_Mode>(new CFB_Encryption(bc.release(), feedback_bits));
      else
         return std::unique_ptr<Cipher_Mode>(new CFB_Decryption(bc.release(), feedback_bits));
      }
#endif

#endif

   return std::unique_ptr<Cipher_Mode>();
   }

//static
std::vector<std::string> Cipher_Mode::providers(const std::string& algo_spec)
   {
   const std::vector<std::string>& possible = { "base", "openssl", "commoncrypto" };
   std::vector<std::string> providers;
   for(auto&& prov : possible)
      {
      std::unique_ptr<Cipher_Mode> mode = Cipher_Mode::create(algo_spec, ENCRYPTION, prov);
      if(mode)
         {
         providers.push_back(prov); // available
         }
      }
   return providers;
   }

}
/*
* Comba Multiplication and Squaring
*
* This file was automatically generated by ./src/scripts/comba.py on 2018-05-08
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/*
* Comba 4x4 Squaring
*/
void bigint_comba_sqr4(word z[8], const word x[4])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd  (&w2, &w1, &w0, x[ 0], x[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 1]);
   z[ 1] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 2]);
   word3_muladd  (&w1, &w0, &w2, x[ 1], x[ 1]);
   z[ 2] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 3]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 2]);
   z[ 3] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 3]);
   word3_muladd  (&w0, &w2, &w1, x[ 2], x[ 2]);
   z[ 4] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 3]);
   z[ 5] = w2; w2 = 0;

   word3_muladd  (&w2, &w1, &w0, x[ 3], x[ 3]);
   z[ 6] = w0;
   z[ 7] = w1;
   }

/*
* Comba 4x4 Multiplication
*/
void bigint_comba_mul4(word z[8], const word x[4], const word y[4])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 0]);
   z[ 1] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 0]);
   z[ 2] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 0]);
   z[ 3] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 1]);
   z[ 4] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 2]);
   z[ 5] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 3]);
   z[ 6] = w0;
   z[ 7] = w1;
   }

/*
* Comba 6x6 Squaring
*/
void bigint_comba_sqr6(word z[12], const word x[6])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd  (&w2, &w1, &w0, x[ 0], x[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 1]);
   z[ 1] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 2]);
   word3_muladd  (&w1, &w0, &w2, x[ 1], x[ 1]);
   z[ 2] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 3]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 2]);
   z[ 3] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 4]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 3]);
   word3_muladd  (&w0, &w2, &w1, x[ 2], x[ 2]);
   z[ 4] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 5]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[ 4]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 3]);
   z[ 5] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 5]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[ 4]);
   word3_muladd  (&w2, &w1, &w0, x[ 3], x[ 3]);
   z[ 6] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[ 5]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[ 4]);
   z[ 7] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[ 5]);
   word3_muladd  (&w1, &w0, &w2, x[ 4], x[ 4]);
   z[ 8] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[ 5]);
   z[ 9] = w0; w0 = 0;

   word3_muladd  (&w0, &w2, &w1, x[ 5], x[ 5]);
   z[10] = w1;
   z[11] = w2;
   }

/*
* Comba 6x6 Multiplication
*/
void bigint_comba_mul6(word z[12], const word x[6], const word y[6])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 0]);
   z[ 1] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 0]);
   z[ 2] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 0]);
   z[ 3] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 0]);
   z[ 4] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 0]);
   z[ 5] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 1]);
   z[ 6] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 2]);
   z[ 7] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 3]);
   z[ 8] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 4]);
   z[ 9] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 5]);
   z[10] = w1;
   z[11] = w2;
   }

/*
* Comba 8x8 Squaring
*/
void bigint_comba_sqr8(word z[16], const word x[8])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd  (&w2, &w1, &w0, x[ 0], x[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 1]);
   z[ 1] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 2]);
   word3_muladd  (&w1, &w0, &w2, x[ 1], x[ 1]);
   z[ 2] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 3]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 2]);
   z[ 3] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 4]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 3]);
   word3_muladd  (&w0, &w2, &w1, x[ 2], x[ 2]);
   z[ 4] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 5]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[ 4]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 3]);
   z[ 5] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 6]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 5]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[ 4]);
   word3_muladd  (&w2, &w1, &w0, x[ 3], x[ 3]);
   z[ 6] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 7]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 6]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[ 5]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[ 4]);
   z[ 7] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[ 7]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 6]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[ 5]);
   word3_muladd  (&w1, &w0, &w2, x[ 4], x[ 4]);
   z[ 8] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[ 7]);
   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[ 6]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[ 5]);
   z[ 9] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[ 7]);
   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[ 6]);
   word3_muladd  (&w0, &w2, &w1, x[ 5], x[ 5]);
   z[10] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 4], x[ 7]);
   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[ 6]);
   z[11] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 5], x[ 7]);
   word3_muladd  (&w2, &w1, &w0, x[ 6], x[ 6]);
   z[12] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 6], x[ 7]);
   z[13] = w1; w1 = 0;

   word3_muladd  (&w1, &w0, &w2, x[ 7], x[ 7]);
   z[14] = w2;
   z[15] = w0;
   }

/*
* Comba 8x8 Multiplication
*/
void bigint_comba_mul8(word z[16], const word x[8], const word y[8])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 0]);
   z[ 1] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 0]);
   z[ 2] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 0]);
   z[ 3] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 0]);
   z[ 4] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 0]);
   z[ 5] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 0]);
   z[ 6] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 0]);
   z[ 7] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 1]);
   z[ 8] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[ 2]);
   z[ 9] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 3]);
   z[10] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 4]);
   z[11] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[ 5]);
   z[12] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 6]);
   z[13] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 7]);
   z[14] = w2;
   z[15] = w0;
   }

/*
* Comba 9x9 Squaring
*/
void bigint_comba_sqr9(word z[18], const word x[9])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd  (&w2, &w1, &w0, x[ 0], x[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 1]);
   z[ 1] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 2]);
   word3_muladd  (&w1, &w0, &w2, x[ 1], x[ 1]);
   z[ 2] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 3]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 2]);
   z[ 3] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 4]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 3]);
   word3_muladd  (&w0, &w2, &w1, x[ 2], x[ 2]);
   z[ 4] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 5]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[ 4]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 3]);
   z[ 5] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 6]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 5]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[ 4]);
   word3_muladd  (&w2, &w1, &w0, x[ 3], x[ 3]);
   z[ 6] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 7]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 6]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[ 5]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[ 4]);
   z[ 7] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 8]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[ 7]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 6]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[ 5]);
   word3_muladd  (&w1, &w0, &w2, x[ 4], x[ 4]);
   z[ 8] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 8]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[ 7]);
   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[ 6]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[ 5]);
   z[ 9] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[ 8]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[ 7]);
   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[ 6]);
   word3_muladd  (&w0, &w2, &w1, x[ 5], x[ 5]);
   z[10] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[ 8]);
   word3_muladd_2(&w1, &w0, &w2, x[ 4], x[ 7]);
   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[ 6]);
   z[11] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[ 8]);
   word3_muladd_2(&w2, &w1, &w0, x[ 5], x[ 7]);
   word3_muladd  (&w2, &w1, &w0, x[ 6], x[ 6]);
   z[12] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 5], x[ 8]);
   word3_muladd_2(&w0, &w2, &w1, x[ 6], x[ 7]);
   z[13] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 6], x[ 8]);
   word3_muladd  (&w1, &w0, &w2, x[ 7], x[ 7]);
   z[14] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 7], x[ 8]);
   z[15] = w0; w0 = 0;

   word3_muladd  (&w0, &w2, &w1, x[ 8], x[ 8]);
   z[16] = w1;
   z[17] = w2;
   }

/*
* Comba 9x9 Multiplication
*/
void bigint_comba_mul9(word z[18], const word x[9], const word y[9])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 0]);
   z[ 1] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 0]);
   z[ 2] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 0]);
   z[ 3] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 0]);
   z[ 4] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 0]);
   z[ 5] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 0]);
   z[ 6] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 0]);
   z[ 7] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[ 0]);
   z[ 8] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[ 1]);
   z[ 9] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[ 2]);
   z[10] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[ 3]);
   z[11] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[ 4]);
   z[12] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[ 5]);
   z[13] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 6], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[ 6]);
   z[14] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 7], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[ 7]);
   z[15] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 8], y[ 8]);
   z[16] = w1;
   z[17] = w2;
   }

/*
* Comba 16x16 Squaring
*/
void bigint_comba_sqr16(word z[32], const word x[16])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd  (&w2, &w1, &w0, x[ 0], x[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 1]);
   z[ 1] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 2]);
   word3_muladd  (&w1, &w0, &w2, x[ 1], x[ 1]);
   z[ 2] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 3]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 2]);
   z[ 3] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 4]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 3]);
   word3_muladd  (&w0, &w2, &w1, x[ 2], x[ 2]);
   z[ 4] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 5]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[ 4]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 3]);
   z[ 5] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 6]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 5]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[ 4]);
   word3_muladd  (&w2, &w1, &w0, x[ 3], x[ 3]);
   z[ 6] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 7]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 6]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[ 5]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[ 4]);
   z[ 7] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 8]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[ 7]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 6]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[ 5]);
   word3_muladd  (&w1, &w0, &w2, x[ 4], x[ 4]);
   z[ 8] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 9]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 8]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[ 7]);
   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[ 6]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[ 5]);
   z[ 9] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[10]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 9]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[ 8]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[ 7]);
   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[ 6]);
   word3_muladd  (&w0, &w2, &w1, x[ 5], x[ 5]);
   z[10] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[11]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[10]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 9]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[ 8]);
   word3_muladd_2(&w1, &w0, &w2, x[ 4], x[ 7]);
   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[ 6]);
   z[11] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[12]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[11]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[10]);
   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[ 9]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[ 8]);
   word3_muladd_2(&w2, &w1, &w0, x[ 5], x[ 7]);
   word3_muladd  (&w2, &w1, &w0, x[ 6], x[ 6]);
   z[12] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[13]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[12]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[11]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[10]);
   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[ 9]);
   word3_muladd_2(&w0, &w2, &w1, x[ 5], x[ 8]);
   word3_muladd_2(&w0, &w2, &w1, x[ 6], x[ 7]);
   z[13] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[14]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[13]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[12]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[11]);
   word3_muladd_2(&w1, &w0, &w2, x[ 4], x[10]);
   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[ 9]);
   word3_muladd_2(&w1, &w0, &w2, x[ 6], x[ 8]);
   word3_muladd  (&w1, &w0, &w2, x[ 7], x[ 7]);
   z[14] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[15]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[14]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[13]);
   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[12]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[11]);
   word3_muladd_2(&w2, &w1, &w0, x[ 5], x[10]);
   word3_muladd_2(&w2, &w1, &w0, x[ 6], x[ 9]);
   word3_muladd_2(&w2, &w1, &w0, x[ 7], x[ 8]);
   z[15] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[15]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[14]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[13]);
   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[12]);
   word3_muladd_2(&w0, &w2, &w1, x[ 5], x[11]);
   word3_muladd_2(&w0, &w2, &w1, x[ 6], x[10]);
   word3_muladd_2(&w0, &w2, &w1, x[ 7], x[ 9]);
   word3_muladd  (&w0, &w2, &w1, x[ 8], x[ 8]);
   z[16] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[15]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[14]);
   word3_muladd_2(&w1, &w0, &w2, x[ 4], x[13]);
   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[12]);
   word3_muladd_2(&w1, &w0, &w2, x[ 6], x[11]);
   word3_muladd_2(&w1, &w0, &w2, x[ 7], x[10]);
   word3_muladd_2(&w1, &w0, &w2, x[ 8], x[ 9]);
   z[17] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[15]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[14]);
   word3_muladd_2(&w2, &w1, &w0, x[ 5], x[13]);
   word3_muladd_2(&w2, &w1, &w0, x[ 6], x[12]);
   word3_muladd_2(&w2, &w1, &w0, x[ 7], x[11]);
   word3_muladd_2(&w2, &w1, &w0, x[ 8], x[10]);
   word3_muladd  (&w2, &w1, &w0, x[ 9], x[ 9]);
   z[18] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[15]);
   word3_muladd_2(&w0, &w2, &w1, x[ 5], x[14]);
   word3_muladd_2(&w0, &w2, &w1, x[ 6], x[13]);
   word3_muladd_2(&w0, &w2, &w1, x[ 7], x[12]);
   word3_muladd_2(&w0, &w2, &w1, x[ 8], x[11]);
   word3_muladd_2(&w0, &w2, &w1, x[ 9], x[10]);
   z[19] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[15]);
   word3_muladd_2(&w1, &w0, &w2, x[ 6], x[14]);
   word3_muladd_2(&w1, &w0, &w2, x[ 7], x[13]);
   word3_muladd_2(&w1, &w0, &w2, x[ 8], x[12]);
   word3_muladd_2(&w1, &w0, &w2, x[ 9], x[11]);
   word3_muladd  (&w1, &w0, &w2, x[10], x[10]);
   z[20] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 6], x[15]);
   word3_muladd_2(&w2, &w1, &w0, x[ 7], x[14]);
   word3_muladd_2(&w2, &w1, &w0, x[ 8], x[13]);
   word3_muladd_2(&w2, &w1, &w0, x[ 9], x[12]);
   word3_muladd_2(&w2, &w1, &w0, x[10], x[11]);
   z[21] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 7], x[15]);
   word3_muladd_2(&w0, &w2, &w1, x[ 8], x[14]);
   word3_muladd_2(&w0, &w2, &w1, x[ 9], x[13]);
   word3_muladd_2(&w0, &w2, &w1, x[10], x[12]);
   word3_muladd  (&w0, &w2, &w1, x[11], x[11]);
   z[22] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 8], x[15]);
   word3_muladd_2(&w1, &w0, &w2, x[ 9], x[14]);
   word3_muladd_2(&w1, &w0, &w2, x[10], x[13]);
   word3_muladd_2(&w1, &w0, &w2, x[11], x[12]);
   z[23] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 9], x[15]);
   word3_muladd_2(&w2, &w1, &w0, x[10], x[14]);
   word3_muladd_2(&w2, &w1, &w0, x[11], x[13]);
   word3_muladd  (&w2, &w1, &w0, x[12], x[12]);
   z[24] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[10], x[15]);
   word3_muladd_2(&w0, &w2, &w1, x[11], x[14]);
   word3_muladd_2(&w0, &w2, &w1, x[12], x[13]);
   z[25] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[11], x[15]);
   word3_muladd_2(&w1, &w0, &w2, x[12], x[14]);
   word3_muladd  (&w1, &w0, &w2, x[13], x[13]);
   z[26] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[12], x[15]);
   word3_muladd_2(&w2, &w1, &w0, x[13], x[14]);
   z[27] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[13], x[15]);
   word3_muladd  (&w0, &w2, &w1, x[14], x[14]);
   z[28] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[14], x[15]);
   z[29] = w2; w2 = 0;

   word3_muladd  (&w2, &w1, &w0, x[15], x[15]);
   z[30] = w0;
   z[31] = w1;
   }

/*
* Comba 16x16 Multiplication
*/
void bigint_comba_mul16(word z[32], const word x[16], const word y[16])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 0]);
   z[ 1] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 0]);
   z[ 2] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 0]);
   z[ 3] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 0]);
   z[ 4] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 0]);
   z[ 5] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 0]);
   z[ 6] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 0]);
   z[ 7] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[ 0]);
   z[ 8] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[ 0]);
   z[ 9] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[10]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[10], y[ 0]);
   z[10] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[11]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[10]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[10], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[11], y[ 0]);
   z[11] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[12]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[11]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[10]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[10], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[11], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[12], y[ 0]);
   z[12] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[13]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[12]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[11]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[10]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[10], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[11], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[12], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[13], y[ 0]);
   z[13] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[14]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[13]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[12]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[11]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[10]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[10], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[11], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[12], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[13], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[14], y[ 0]);
   z[14] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[15]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[14]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[13]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[12]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[11]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[10]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[10], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[11], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[12], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[13], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[14], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[15], y[ 0]);
   z[15] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 1], y[15]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[14]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[13]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[12]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[11]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[10]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[10], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[11], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[12], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[13], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[14], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[15], y[ 1]);
   z[16] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 2], y[15]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[14]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[13]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[12]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[11]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[10]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[10], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[11], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[12], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[13], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[14], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[15], y[ 2]);
   z[17] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 3], y[15]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[14]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[13]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[12]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[11]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[10]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[10], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[11], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[12], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[13], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[14], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[15], y[ 3]);
   z[18] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 4], y[15]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[14]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[13]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[12]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[11]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[10]);
   word3_muladd(&w0, &w2, &w1, x[10], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[11], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[12], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[13], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[14], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[15], y[ 4]);
   z[19] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 5], y[15]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[14]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[13]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[12]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[11]);
   word3_muladd(&w1, &w0, &w2, x[10], y[10]);
   word3_muladd(&w1, &w0, &w2, x[11], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[12], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[13], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[14], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[15], y[ 5]);
   z[20] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 6], y[15]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[14]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[13]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[12]);
   word3_muladd(&w2, &w1, &w0, x[10], y[11]);
   word3_muladd(&w2, &w1, &w0, x[11], y[10]);
   word3_muladd(&w2, &w1, &w0, x[12], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[13], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[14], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[15], y[ 6]);
   z[21] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 7], y[15]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[14]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[13]);
   word3_muladd(&w0, &w2, &w1, x[10], y[12]);
   word3_muladd(&w0, &w2, &w1, x[11], y[11]);
   word3_muladd(&w0, &w2, &w1, x[12], y[10]);
   word3_muladd(&w0, &w2, &w1, x[13], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[14], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[15], y[ 7]);
   z[22] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 8], y[15]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[14]);
   word3_muladd(&w1, &w0, &w2, x[10], y[13]);
   word3_muladd(&w1, &w0, &w2, x[11], y[12]);
   word3_muladd(&w1, &w0, &w2, x[12], y[11]);
   word3_muladd(&w1, &w0, &w2, x[13], y[10]);
   word3_muladd(&w1, &w0, &w2, x[14], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[15], y[ 8]);
   z[23] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 9], y[15]);
   word3_muladd(&w2, &w1, &w0, x[10], y[14]);
   word3_muladd(&w2, &w1, &w0, x[11], y[13]);
   word3_muladd(&w2, &w1, &w0, x[12], y[12]);
   word3_muladd(&w2, &w1, &w0, x[13], y[11]);
   word3_muladd(&w2, &w1, &w0, x[14], y[10]);
   word3_muladd(&w2, &w1, &w0, x[15], y[ 9]);
   z[24] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[10], y[15]);
   word3_muladd(&w0, &w2, &w1, x[11], y[14]);
   word3_muladd(&w0, &w2, &w1, x[12], y[13]);
   word3_muladd(&w0, &w2, &w1, x[13], y[12]);
   word3_muladd(&w0, &w2, &w1, x[14], y[11]);
   word3_muladd(&w0, &w2, &w1, x[15], y[10]);
   z[25] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[11], y[15]);
   word3_muladd(&w1, &w0, &w2, x[12], y[14]);
   word3_muladd(&w1, &w0, &w2, x[13], y[13]);
   word3_muladd(&w1, &w0, &w2, x[14], y[12]);
   word3_muladd(&w1, &w0, &w2, x[15], y[11]);
   z[26] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[12], y[15]);
   word3_muladd(&w2, &w1, &w0, x[13], y[14]);
   word3_muladd(&w2, &w1, &w0, x[14], y[13]);
   word3_muladd(&w2, &w1, &w0, x[15], y[12]);
   z[27] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[13], y[15]);
   word3_muladd(&w0, &w2, &w1, x[14], y[14]);
   word3_muladd(&w0, &w2, &w1, x[15], y[13]);
   z[28] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[14], y[15]);
   word3_muladd(&w1, &w0, &w2, x[15], y[14]);
   z[29] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[15], y[15]);
   z[30] = w0;
   z[31] = w1;
   }

/*
* Comba 24x24 Squaring
*/
void bigint_comba_sqr24(word z[48], const word x[24])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd  (&w2, &w1, &w0, x[ 0], x[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 1]);
   z[ 1] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 2]);
   word3_muladd  (&w1, &w0, &w2, x[ 1], x[ 1]);
   z[ 2] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 3]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 2]);
   z[ 3] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 4]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 3]);
   word3_muladd  (&w0, &w2, &w1, x[ 2], x[ 2]);
   z[ 4] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 5]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[ 4]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 3]);
   z[ 5] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 6]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 5]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[ 4]);
   word3_muladd  (&w2, &w1, &w0, x[ 3], x[ 3]);
   z[ 6] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[ 7]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 6]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[ 5]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[ 4]);
   z[ 7] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[ 8]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[ 7]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 6]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[ 5]);
   word3_muladd  (&w1, &w0, &w2, x[ 4], x[ 4]);
   z[ 8] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[ 9]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[ 8]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[ 7]);
   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[ 6]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[ 5]);
   z[ 9] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[10]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[ 9]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[ 8]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[ 7]);
   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[ 6]);
   word3_muladd  (&w0, &w2, &w1, x[ 5], x[ 5]);
   z[10] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[11]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[10]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[ 9]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[ 8]);
   word3_muladd_2(&w1, &w0, &w2, x[ 4], x[ 7]);
   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[ 6]);
   z[11] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[12]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[11]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[10]);
   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[ 9]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[ 8]);
   word3_muladd_2(&w2, &w1, &w0, x[ 5], x[ 7]);
   word3_muladd  (&w2, &w1, &w0, x[ 6], x[ 6]);
   z[12] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[13]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[12]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[11]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[10]);
   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[ 9]);
   word3_muladd_2(&w0, &w2, &w1, x[ 5], x[ 8]);
   word3_muladd_2(&w0, &w2, &w1, x[ 6], x[ 7]);
   z[13] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[14]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[13]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[12]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[11]);
   word3_muladd_2(&w1, &w0, &w2, x[ 4], x[10]);
   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[ 9]);
   word3_muladd_2(&w1, &w0, &w2, x[ 6], x[ 8]);
   word3_muladd  (&w1, &w0, &w2, x[ 7], x[ 7]);
   z[14] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[15]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[14]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[13]);
   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[12]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[11]);
   word3_muladd_2(&w2, &w1, &w0, x[ 5], x[10]);
   word3_muladd_2(&w2, &w1, &w0, x[ 6], x[ 9]);
   word3_muladd_2(&w2, &w1, &w0, x[ 7], x[ 8]);
   z[15] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[16]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[15]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[14]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[13]);
   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[12]);
   word3_muladd_2(&w0, &w2, &w1, x[ 5], x[11]);
   word3_muladd_2(&w0, &w2, &w1, x[ 6], x[10]);
   word3_muladd_2(&w0, &w2, &w1, x[ 7], x[ 9]);
   word3_muladd  (&w0, &w2, &w1, x[ 8], x[ 8]);
   z[16] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[17]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[16]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[15]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[14]);
   word3_muladd_2(&w1, &w0, &w2, x[ 4], x[13]);
   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[12]);
   word3_muladd_2(&w1, &w0, &w2, x[ 6], x[11]);
   word3_muladd_2(&w1, &w0, &w2, x[ 7], x[10]);
   word3_muladd_2(&w1, &w0, &w2, x[ 8], x[ 9]);
   z[17] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[18]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[17]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[16]);
   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[15]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[14]);
   word3_muladd_2(&w2, &w1, &w0, x[ 5], x[13]);
   word3_muladd_2(&w2, &w1, &w0, x[ 6], x[12]);
   word3_muladd_2(&w2, &w1, &w0, x[ 7], x[11]);
   word3_muladd_2(&w2, &w1, &w0, x[ 8], x[10]);
   word3_muladd  (&w2, &w1, &w0, x[ 9], x[ 9]);
   z[18] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[19]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[18]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[17]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[16]);
   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[15]);
   word3_muladd_2(&w0, &w2, &w1, x[ 5], x[14]);
   word3_muladd_2(&w0, &w2, &w1, x[ 6], x[13]);
   word3_muladd_2(&w0, &w2, &w1, x[ 7], x[12]);
   word3_muladd_2(&w0, &w2, &w1, x[ 8], x[11]);
   word3_muladd_2(&w0, &w2, &w1, x[ 9], x[10]);
   z[19] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[20]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[19]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[18]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[17]);
   word3_muladd_2(&w1, &w0, &w2, x[ 4], x[16]);
   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[15]);
   word3_muladd_2(&w1, &w0, &w2, x[ 6], x[14]);
   word3_muladd_2(&w1, &w0, &w2, x[ 7], x[13]);
   word3_muladd_2(&w1, &w0, &w2, x[ 8], x[12]);
   word3_muladd_2(&w1, &w0, &w2, x[ 9], x[11]);
   word3_muladd  (&w1, &w0, &w2, x[10], x[10]);
   z[20] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 0], x[21]);
   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[20]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[19]);
   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[18]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[17]);
   word3_muladd_2(&w2, &w1, &w0, x[ 5], x[16]);
   word3_muladd_2(&w2, &w1, &w0, x[ 6], x[15]);
   word3_muladd_2(&w2, &w1, &w0, x[ 7], x[14]);
   word3_muladd_2(&w2, &w1, &w0, x[ 8], x[13]);
   word3_muladd_2(&w2, &w1, &w0, x[ 9], x[12]);
   word3_muladd_2(&w2, &w1, &w0, x[10], x[11]);
   z[21] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 0], x[22]);
   word3_muladd_2(&w0, &w2, &w1, x[ 1], x[21]);
   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[20]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[19]);
   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[18]);
   word3_muladd_2(&w0, &w2, &w1, x[ 5], x[17]);
   word3_muladd_2(&w0, &w2, &w1, x[ 6], x[16]);
   word3_muladd_2(&w0, &w2, &w1, x[ 7], x[15]);
   word3_muladd_2(&w0, &w2, &w1, x[ 8], x[14]);
   word3_muladd_2(&w0, &w2, &w1, x[ 9], x[13]);
   word3_muladd_2(&w0, &w2, &w1, x[10], x[12]);
   word3_muladd  (&w0, &w2, &w1, x[11], x[11]);
   z[22] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 0], x[23]);
   word3_muladd_2(&w1, &w0, &w2, x[ 1], x[22]);
   word3_muladd_2(&w1, &w0, &w2, x[ 2], x[21]);
   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[20]);
   word3_muladd_2(&w1, &w0, &w2, x[ 4], x[19]);
   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[18]);
   word3_muladd_2(&w1, &w0, &w2, x[ 6], x[17]);
   word3_muladd_2(&w1, &w0, &w2, x[ 7], x[16]);
   word3_muladd_2(&w1, &w0, &w2, x[ 8], x[15]);
   word3_muladd_2(&w1, &w0, &w2, x[ 9], x[14]);
   word3_muladd_2(&w1, &w0, &w2, x[10], x[13]);
   word3_muladd_2(&w1, &w0, &w2, x[11], x[12]);
   z[23] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 1], x[23]);
   word3_muladd_2(&w2, &w1, &w0, x[ 2], x[22]);
   word3_muladd_2(&w2, &w1, &w0, x[ 3], x[21]);
   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[20]);
   word3_muladd_2(&w2, &w1, &w0, x[ 5], x[19]);
   word3_muladd_2(&w2, &w1, &w0, x[ 6], x[18]);
   word3_muladd_2(&w2, &w1, &w0, x[ 7], x[17]);
   word3_muladd_2(&w2, &w1, &w0, x[ 8], x[16]);
   word3_muladd_2(&w2, &w1, &w0, x[ 9], x[15]);
   word3_muladd_2(&w2, &w1, &w0, x[10], x[14]);
   word3_muladd_2(&w2, &w1, &w0, x[11], x[13]);
   word3_muladd  (&w2, &w1, &w0, x[12], x[12]);
   z[24] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 2], x[23]);
   word3_muladd_2(&w0, &w2, &w1, x[ 3], x[22]);
   word3_muladd_2(&w0, &w2, &w1, x[ 4], x[21]);
   word3_muladd_2(&w0, &w2, &w1, x[ 5], x[20]);
   word3_muladd_2(&w0, &w2, &w1, x[ 6], x[19]);
   word3_muladd_2(&w0, &w2, &w1, x[ 7], x[18]);
   word3_muladd_2(&w0, &w2, &w1, x[ 8], x[17]);
   word3_muladd_2(&w0, &w2, &w1, x[ 9], x[16]);
   word3_muladd_2(&w0, &w2, &w1, x[10], x[15]);
   word3_muladd_2(&w0, &w2, &w1, x[11], x[14]);
   word3_muladd_2(&w0, &w2, &w1, x[12], x[13]);
   z[25] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 3], x[23]);
   word3_muladd_2(&w1, &w0, &w2, x[ 4], x[22]);
   word3_muladd_2(&w1, &w0, &w2, x[ 5], x[21]);
   word3_muladd_2(&w1, &w0, &w2, x[ 6], x[20]);
   word3_muladd_2(&w1, &w0, &w2, x[ 7], x[19]);
   word3_muladd_2(&w1, &w0, &w2, x[ 8], x[18]);
   word3_muladd_2(&w1, &w0, &w2, x[ 9], x[17]);
   word3_muladd_2(&w1, &w0, &w2, x[10], x[16]);
   word3_muladd_2(&w1, &w0, &w2, x[11], x[15]);
   word3_muladd_2(&w1, &w0, &w2, x[12], x[14]);
   word3_muladd  (&w1, &w0, &w2, x[13], x[13]);
   z[26] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 4], x[23]);
   word3_muladd_2(&w2, &w1, &w0, x[ 5], x[22]);
   word3_muladd_2(&w2, &w1, &w0, x[ 6], x[21]);
   word3_muladd_2(&w2, &w1, &w0, x[ 7], x[20]);
   word3_muladd_2(&w2, &w1, &w0, x[ 8], x[19]);
   word3_muladd_2(&w2, &w1, &w0, x[ 9], x[18]);
   word3_muladd_2(&w2, &w1, &w0, x[10], x[17]);
   word3_muladd_2(&w2, &w1, &w0, x[11], x[16]);
   word3_muladd_2(&w2, &w1, &w0, x[12], x[15]);
   word3_muladd_2(&w2, &w1, &w0, x[13], x[14]);
   z[27] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 5], x[23]);
   word3_muladd_2(&w0, &w2, &w1, x[ 6], x[22]);
   word3_muladd_2(&w0, &w2, &w1, x[ 7], x[21]);
   word3_muladd_2(&w0, &w2, &w1, x[ 8], x[20]);
   word3_muladd_2(&w0, &w2, &w1, x[ 9], x[19]);
   word3_muladd_2(&w0, &w2, &w1, x[10], x[18]);
   word3_muladd_2(&w0, &w2, &w1, x[11], x[17]);
   word3_muladd_2(&w0, &w2, &w1, x[12], x[16]);
   word3_muladd_2(&w0, &w2, &w1, x[13], x[15]);
   word3_muladd  (&w0, &w2, &w1, x[14], x[14]);
   z[28] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 6], x[23]);
   word3_muladd_2(&w1, &w0, &w2, x[ 7], x[22]);
   word3_muladd_2(&w1, &w0, &w2, x[ 8], x[21]);
   word3_muladd_2(&w1, &w0, &w2, x[ 9], x[20]);
   word3_muladd_2(&w1, &w0, &w2, x[10], x[19]);
   word3_muladd_2(&w1, &w0, &w2, x[11], x[18]);
   word3_muladd_2(&w1, &w0, &w2, x[12], x[17]);
   word3_muladd_2(&w1, &w0, &w2, x[13], x[16]);
   word3_muladd_2(&w1, &w0, &w2, x[14], x[15]);
   z[29] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[ 7], x[23]);
   word3_muladd_2(&w2, &w1, &w0, x[ 8], x[22]);
   word3_muladd_2(&w2, &w1, &w0, x[ 9], x[21]);
   word3_muladd_2(&w2, &w1, &w0, x[10], x[20]);
   word3_muladd_2(&w2, &w1, &w0, x[11], x[19]);
   word3_muladd_2(&w2, &w1, &w0, x[12], x[18]);
   word3_muladd_2(&w2, &w1, &w0, x[13], x[17]);
   word3_muladd_2(&w2, &w1, &w0, x[14], x[16]);
   word3_muladd  (&w2, &w1, &w0, x[15], x[15]);
   z[30] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[ 8], x[23]);
   word3_muladd_2(&w0, &w2, &w1, x[ 9], x[22]);
   word3_muladd_2(&w0, &w2, &w1, x[10], x[21]);
   word3_muladd_2(&w0, &w2, &w1, x[11], x[20]);
   word3_muladd_2(&w0, &w2, &w1, x[12], x[19]);
   word3_muladd_2(&w0, &w2, &w1, x[13], x[18]);
   word3_muladd_2(&w0, &w2, &w1, x[14], x[17]);
   word3_muladd_2(&w0, &w2, &w1, x[15], x[16]);
   z[31] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[ 9], x[23]);
   word3_muladd_2(&w1, &w0, &w2, x[10], x[22]);
   word3_muladd_2(&w1, &w0, &w2, x[11], x[21]);
   word3_muladd_2(&w1, &w0, &w2, x[12], x[20]);
   word3_muladd_2(&w1, &w0, &w2, x[13], x[19]);
   word3_muladd_2(&w1, &w0, &w2, x[14], x[18]);
   word3_muladd_2(&w1, &w0, &w2, x[15], x[17]);
   word3_muladd  (&w1, &w0, &w2, x[16], x[16]);
   z[32] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[10], x[23]);
   word3_muladd_2(&w2, &w1, &w0, x[11], x[22]);
   word3_muladd_2(&w2, &w1, &w0, x[12], x[21]);
   word3_muladd_2(&w2, &w1, &w0, x[13], x[20]);
   word3_muladd_2(&w2, &w1, &w0, x[14], x[19]);
   word3_muladd_2(&w2, &w1, &w0, x[15], x[18]);
   word3_muladd_2(&w2, &w1, &w0, x[16], x[17]);
   z[33] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[11], x[23]);
   word3_muladd_2(&w0, &w2, &w1, x[12], x[22]);
   word3_muladd_2(&w0, &w2, &w1, x[13], x[21]);
   word3_muladd_2(&w0, &w2, &w1, x[14], x[20]);
   word3_muladd_2(&w0, &w2, &w1, x[15], x[19]);
   word3_muladd_2(&w0, &w2, &w1, x[16], x[18]);
   word3_muladd  (&w0, &w2, &w1, x[17], x[17]);
   z[34] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[12], x[23]);
   word3_muladd_2(&w1, &w0, &w2, x[13], x[22]);
   word3_muladd_2(&w1, &w0, &w2, x[14], x[21]);
   word3_muladd_2(&w1, &w0, &w2, x[15], x[20]);
   word3_muladd_2(&w1, &w0, &w2, x[16], x[19]);
   word3_muladd_2(&w1, &w0, &w2, x[17], x[18]);
   z[35] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[13], x[23]);
   word3_muladd_2(&w2, &w1, &w0, x[14], x[22]);
   word3_muladd_2(&w2, &w1, &w0, x[15], x[21]);
   word3_muladd_2(&w2, &w1, &w0, x[16], x[20]);
   word3_muladd_2(&w2, &w1, &w0, x[17], x[19]);
   word3_muladd  (&w2, &w1, &w0, x[18], x[18]);
   z[36] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[14], x[23]);
   word3_muladd_2(&w0, &w2, &w1, x[15], x[22]);
   word3_muladd_2(&w0, &w2, &w1, x[16], x[21]);
   word3_muladd_2(&w0, &w2, &w1, x[17], x[20]);
   word3_muladd_2(&w0, &w2, &w1, x[18], x[19]);
   z[37] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[15], x[23]);
   word3_muladd_2(&w1, &w0, &w2, x[16], x[22]);
   word3_muladd_2(&w1, &w0, &w2, x[17], x[21]);
   word3_muladd_2(&w1, &w0, &w2, x[18], x[20]);
   word3_muladd  (&w1, &w0, &w2, x[19], x[19]);
   z[38] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[16], x[23]);
   word3_muladd_2(&w2, &w1, &w0, x[17], x[22]);
   word3_muladd_2(&w2, &w1, &w0, x[18], x[21]);
   word3_muladd_2(&w2, &w1, &w0, x[19], x[20]);
   z[39] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[17], x[23]);
   word3_muladd_2(&w0, &w2, &w1, x[18], x[22]);
   word3_muladd_2(&w0, &w2, &w1, x[19], x[21]);
   word3_muladd  (&w0, &w2, &w1, x[20], x[20]);
   z[40] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[18], x[23]);
   word3_muladd_2(&w1, &w0, &w2, x[19], x[22]);
   word3_muladd_2(&w1, &w0, &w2, x[20], x[21]);
   z[41] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[19], x[23]);
   word3_muladd_2(&w2, &w1, &w0, x[20], x[22]);
   word3_muladd  (&w2, &w1, &w0, x[21], x[21]);
   z[42] = w0; w0 = 0;

   word3_muladd_2(&w0, &w2, &w1, x[20], x[23]);
   word3_muladd_2(&w0, &w2, &w1, x[21], x[22]);
   z[43] = w1; w1 = 0;

   word3_muladd_2(&w1, &w0, &w2, x[21], x[23]);
   word3_muladd  (&w1, &w0, &w2, x[22], x[22]);
   z[44] = w2; w2 = 0;

   word3_muladd_2(&w2, &w1, &w0, x[22], x[23]);
   z[45] = w0; w0 = 0;

   word3_muladd  (&w0, &w2, &w1, x[23], x[23]);
   z[46] = w1;
   z[47] = w2;
   }

/*
* Comba 24x24 Multiplication
*/
void bigint_comba_mul24(word z[48], const word x[24], const word y[24])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 0]);
   z[ 0] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 0]);
   z[ 1] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 0]);
   z[ 2] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 0]);
   z[ 3] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 0]);
   z[ 4] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 0]);
   z[ 5] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 0]);
   z[ 6] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 0]);
   z[ 7] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[ 0]);
   z[ 8] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[ 0]);
   z[ 9] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[10]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[10], y[ 0]);
   z[10] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[11]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[10]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[10], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[11], y[ 0]);
   z[11] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[12]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[11]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[10]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[10], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[11], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[12], y[ 0]);
   z[12] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[13]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[12]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[11]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[10]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[10], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[11], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[12], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[13], y[ 0]);
   z[13] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[14]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[13]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[12]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[11]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[10]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[10], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[11], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[12], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[13], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[14], y[ 0]);
   z[14] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[15]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[14]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[13]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[12]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[11]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[10]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[10], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[11], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[12], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[13], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[14], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[15], y[ 0]);
   z[15] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[16]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[15]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[14]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[13]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[12]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[11]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[10]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[10], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[11], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[12], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[13], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[14], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[15], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[16], y[ 0]);
   z[16] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[17]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[16]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[15]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[14]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[13]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[12]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[11]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[10]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[10], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[11], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[12], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[13], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[14], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[15], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[16], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[17], y[ 0]);
   z[17] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[18]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[17]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[16]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[15]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[14]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[13]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[12]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[11]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[10]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[10], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[11], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[12], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[13], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[14], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[15], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[16], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[17], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[18], y[ 0]);
   z[18] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[19]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[18]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[17]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[16]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[15]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[14]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[13]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[12]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[11]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[10]);
   word3_muladd(&w0, &w2, &w1, x[10], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[11], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[12], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[13], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[14], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[15], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[16], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[17], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[18], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[19], y[ 0]);
   z[19] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[20]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[19]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[18]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[17]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[16]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[15]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[14]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[13]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[12]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[11]);
   word3_muladd(&w1, &w0, &w2, x[10], y[10]);
   word3_muladd(&w1, &w0, &w2, x[11], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[12], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[13], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[14], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[15], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[16], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[17], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[18], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[19], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[20], y[ 0]);
   z[20] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 0], y[21]);
   word3_muladd(&w2, &w1, &w0, x[ 1], y[20]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[19]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[18]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[17]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[16]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[15]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[14]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[13]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[12]);
   word3_muladd(&w2, &w1, &w0, x[10], y[11]);
   word3_muladd(&w2, &w1, &w0, x[11], y[10]);
   word3_muladd(&w2, &w1, &w0, x[12], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[13], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[14], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[15], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[16], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[17], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[18], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[19], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[20], y[ 1]);
   word3_muladd(&w2, &w1, &w0, x[21], y[ 0]);
   z[21] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 0], y[22]);
   word3_muladd(&w0, &w2, &w1, x[ 1], y[21]);
   word3_muladd(&w0, &w2, &w1, x[ 2], y[20]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[19]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[18]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[17]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[16]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[15]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[14]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[13]);
   word3_muladd(&w0, &w2, &w1, x[10], y[12]);
   word3_muladd(&w0, &w2, &w1, x[11], y[11]);
   word3_muladd(&w0, &w2, &w1, x[12], y[10]);
   word3_muladd(&w0, &w2, &w1, x[13], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[14], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[15], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[16], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[17], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[18], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[19], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[20], y[ 2]);
   word3_muladd(&w0, &w2, &w1, x[21], y[ 1]);
   word3_muladd(&w0, &w2, &w1, x[22], y[ 0]);
   z[22] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 0], y[23]);
   word3_muladd(&w1, &w0, &w2, x[ 1], y[22]);
   word3_muladd(&w1, &w0, &w2, x[ 2], y[21]);
   word3_muladd(&w1, &w0, &w2, x[ 3], y[20]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[19]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[18]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[17]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[16]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[15]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[14]);
   word3_muladd(&w1, &w0, &w2, x[10], y[13]);
   word3_muladd(&w1, &w0, &w2, x[11], y[12]);
   word3_muladd(&w1, &w0, &w2, x[12], y[11]);
   word3_muladd(&w1, &w0, &w2, x[13], y[10]);
   word3_muladd(&w1, &w0, &w2, x[14], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[15], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[16], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[17], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[18], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[19], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[20], y[ 3]);
   word3_muladd(&w1, &w0, &w2, x[21], y[ 2]);
   word3_muladd(&w1, &w0, &w2, x[22], y[ 1]);
   word3_muladd(&w1, &w0, &w2, x[23], y[ 0]);
   z[23] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 1], y[23]);
   word3_muladd(&w2, &w1, &w0, x[ 2], y[22]);
   word3_muladd(&w2, &w1, &w0, x[ 3], y[21]);
   word3_muladd(&w2, &w1, &w0, x[ 4], y[20]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[19]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[18]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[17]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[16]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[15]);
   word3_muladd(&w2, &w1, &w0, x[10], y[14]);
   word3_muladd(&w2, &w1, &w0, x[11], y[13]);
   word3_muladd(&w2, &w1, &w0, x[12], y[12]);
   word3_muladd(&w2, &w1, &w0, x[13], y[11]);
   word3_muladd(&w2, &w1, &w0, x[14], y[10]);
   word3_muladd(&w2, &w1, &w0, x[15], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[16], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[17], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[18], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[19], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[20], y[ 4]);
   word3_muladd(&w2, &w1, &w0, x[21], y[ 3]);
   word3_muladd(&w2, &w1, &w0, x[22], y[ 2]);
   word3_muladd(&w2, &w1, &w0, x[23], y[ 1]);
   z[24] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 2], y[23]);
   word3_muladd(&w0, &w2, &w1, x[ 3], y[22]);
   word3_muladd(&w0, &w2, &w1, x[ 4], y[21]);
   word3_muladd(&w0, &w2, &w1, x[ 5], y[20]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[19]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[18]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[17]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[16]);
   word3_muladd(&w0, &w2, &w1, x[10], y[15]);
   word3_muladd(&w0, &w2, &w1, x[11], y[14]);
   word3_muladd(&w0, &w2, &w1, x[12], y[13]);
   word3_muladd(&w0, &w2, &w1, x[13], y[12]);
   word3_muladd(&w0, &w2, &w1, x[14], y[11]);
   word3_muladd(&w0, &w2, &w1, x[15], y[10]);
   word3_muladd(&w0, &w2, &w1, x[16], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[17], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[18], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[19], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[20], y[ 5]);
   word3_muladd(&w0, &w2, &w1, x[21], y[ 4]);
   word3_muladd(&w0, &w2, &w1, x[22], y[ 3]);
   word3_muladd(&w0, &w2, &w1, x[23], y[ 2]);
   z[25] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 3], y[23]);
   word3_muladd(&w1, &w0, &w2, x[ 4], y[22]);
   word3_muladd(&w1, &w0, &w2, x[ 5], y[21]);
   word3_muladd(&w1, &w0, &w2, x[ 6], y[20]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[19]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[18]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[17]);
   word3_muladd(&w1, &w0, &w2, x[10], y[16]);
   word3_muladd(&w1, &w0, &w2, x[11], y[15]);
   word3_muladd(&w1, &w0, &w2, x[12], y[14]);
   word3_muladd(&w1, &w0, &w2, x[13], y[13]);
   word3_muladd(&w1, &w0, &w2, x[14], y[12]);
   word3_muladd(&w1, &w0, &w2, x[15], y[11]);
   word3_muladd(&w1, &w0, &w2, x[16], y[10]);
   word3_muladd(&w1, &w0, &w2, x[17], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[18], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[19], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[20], y[ 6]);
   word3_muladd(&w1, &w0, &w2, x[21], y[ 5]);
   word3_muladd(&w1, &w0, &w2, x[22], y[ 4]);
   word3_muladd(&w1, &w0, &w2, x[23], y[ 3]);
   z[26] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 4], y[23]);
   word3_muladd(&w2, &w1, &w0, x[ 5], y[22]);
   word3_muladd(&w2, &w1, &w0, x[ 6], y[21]);
   word3_muladd(&w2, &w1, &w0, x[ 7], y[20]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[19]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[18]);
   word3_muladd(&w2, &w1, &w0, x[10], y[17]);
   word3_muladd(&w2, &w1, &w0, x[11], y[16]);
   word3_muladd(&w2, &w1, &w0, x[12], y[15]);
   word3_muladd(&w2, &w1, &w0, x[13], y[14]);
   word3_muladd(&w2, &w1, &w0, x[14], y[13]);
   word3_muladd(&w2, &w1, &w0, x[15], y[12]);
   word3_muladd(&w2, &w1, &w0, x[16], y[11]);
   word3_muladd(&w2, &w1, &w0, x[17], y[10]);
   word3_muladd(&w2, &w1, &w0, x[18], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[19], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[20], y[ 7]);
   word3_muladd(&w2, &w1, &w0, x[21], y[ 6]);
   word3_muladd(&w2, &w1, &w0, x[22], y[ 5]);
   word3_muladd(&w2, &w1, &w0, x[23], y[ 4]);
   z[27] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 5], y[23]);
   word3_muladd(&w0, &w2, &w1, x[ 6], y[22]);
   word3_muladd(&w0, &w2, &w1, x[ 7], y[21]);
   word3_muladd(&w0, &w2, &w1, x[ 8], y[20]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[19]);
   word3_muladd(&w0, &w2, &w1, x[10], y[18]);
   word3_muladd(&w0, &w2, &w1, x[11], y[17]);
   word3_muladd(&w0, &w2, &w1, x[12], y[16]);
   word3_muladd(&w0, &w2, &w1, x[13], y[15]);
   word3_muladd(&w0, &w2, &w1, x[14], y[14]);
   word3_muladd(&w0, &w2, &w1, x[15], y[13]);
   word3_muladd(&w0, &w2, &w1, x[16], y[12]);
   word3_muladd(&w0, &w2, &w1, x[17], y[11]);
   word3_muladd(&w0, &w2, &w1, x[18], y[10]);
   word3_muladd(&w0, &w2, &w1, x[19], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[20], y[ 8]);
   word3_muladd(&w0, &w2, &w1, x[21], y[ 7]);
   word3_muladd(&w0, &w2, &w1, x[22], y[ 6]);
   word3_muladd(&w0, &w2, &w1, x[23], y[ 5]);
   z[28] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 6], y[23]);
   word3_muladd(&w1, &w0, &w2, x[ 7], y[22]);
   word3_muladd(&w1, &w0, &w2, x[ 8], y[21]);
   word3_muladd(&w1, &w0, &w2, x[ 9], y[20]);
   word3_muladd(&w1, &w0, &w2, x[10], y[19]);
   word3_muladd(&w1, &w0, &w2, x[11], y[18]);
   word3_muladd(&w1, &w0, &w2, x[12], y[17]);
   word3_muladd(&w1, &w0, &w2, x[13], y[16]);
   word3_muladd(&w1, &w0, &w2, x[14], y[15]);
   word3_muladd(&w1, &w0, &w2, x[15], y[14]);
   word3_muladd(&w1, &w0, &w2, x[16], y[13]);
   word3_muladd(&w1, &w0, &w2, x[17], y[12]);
   word3_muladd(&w1, &w0, &w2, x[18], y[11]);
   word3_muladd(&w1, &w0, &w2, x[19], y[10]);
   word3_muladd(&w1, &w0, &w2, x[20], y[ 9]);
   word3_muladd(&w1, &w0, &w2, x[21], y[ 8]);
   word3_muladd(&w1, &w0, &w2, x[22], y[ 7]);
   word3_muladd(&w1, &w0, &w2, x[23], y[ 6]);
   z[29] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[ 7], y[23]);
   word3_muladd(&w2, &w1, &w0, x[ 8], y[22]);
   word3_muladd(&w2, &w1, &w0, x[ 9], y[21]);
   word3_muladd(&w2, &w1, &w0, x[10], y[20]);
   word3_muladd(&w2, &w1, &w0, x[11], y[19]);
   word3_muladd(&w2, &w1, &w0, x[12], y[18]);
   word3_muladd(&w2, &w1, &w0, x[13], y[17]);
   word3_muladd(&w2, &w1, &w0, x[14], y[16]);
   word3_muladd(&w2, &w1, &w0, x[15], y[15]);
   word3_muladd(&w2, &w1, &w0, x[16], y[14]);
   word3_muladd(&w2, &w1, &w0, x[17], y[13]);
   word3_muladd(&w2, &w1, &w0, x[18], y[12]);
   word3_muladd(&w2, &w1, &w0, x[19], y[11]);
   word3_muladd(&w2, &w1, &w0, x[20], y[10]);
   word3_muladd(&w2, &w1, &w0, x[21], y[ 9]);
   word3_muladd(&w2, &w1, &w0, x[22], y[ 8]);
   word3_muladd(&w2, &w1, &w0, x[23], y[ 7]);
   z[30] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[ 8], y[23]);
   word3_muladd(&w0, &w2, &w1, x[ 9], y[22]);
   word3_muladd(&w0, &w2, &w1, x[10], y[21]);
   word3_muladd(&w0, &w2, &w1, x[11], y[20]);
   word3_muladd(&w0, &w2, &w1, x[12], y[19]);
   word3_muladd(&w0, &w2, &w1, x[13], y[18]);
   word3_muladd(&w0, &w2, &w1, x[14], y[17]);
   word3_muladd(&w0, &w2, &w1, x[15], y[16]);
   word3_muladd(&w0, &w2, &w1, x[16], y[15]);
   word3_muladd(&w0, &w2, &w1, x[17], y[14]);
   word3_muladd(&w0, &w2, &w1, x[18], y[13]);
   word3_muladd(&w0, &w2, &w1, x[19], y[12]);
   word3_muladd(&w0, &w2, &w1, x[20], y[11]);
   word3_muladd(&w0, &w2, &w1, x[21], y[10]);
   word3_muladd(&w0, &w2, &w1, x[22], y[ 9]);
   word3_muladd(&w0, &w2, &w1, x[23], y[ 8]);
   z[31] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[ 9], y[23]);
   word3_muladd(&w1, &w0, &w2, x[10], y[22]);
   word3_muladd(&w1, &w0, &w2, x[11], y[21]);
   word3_muladd(&w1, &w0, &w2, x[12], y[20]);
   word3_muladd(&w1, &w0, &w2, x[13], y[19]);
   word3_muladd(&w1, &w0, &w2, x[14], y[18]);
   word3_muladd(&w1, &w0, &w2, x[15], y[17]);
   word3_muladd(&w1, &w0, &w2, x[16], y[16]);
   word3_muladd(&w1, &w0, &w2, x[17], y[15]);
   word3_muladd(&w1, &w0, &w2, x[18], y[14]);
   word3_muladd(&w1, &w0, &w2, x[19], y[13]);
   word3_muladd(&w1, &w0, &w2, x[20], y[12]);
   word3_muladd(&w1, &w0, &w2, x[21], y[11]);
   word3_muladd(&w1, &w0, &w2, x[22], y[10]);
   word3_muladd(&w1, &w0, &w2, x[23], y[ 9]);
   z[32] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[10], y[23]);
   word3_muladd(&w2, &w1, &w0, x[11], y[22]);
   word3_muladd(&w2, &w1, &w0, x[12], y[21]);
   word3_muladd(&w2, &w1, &w0, x[13], y[20]);
   word3_muladd(&w2, &w1, &w0, x[14], y[19]);
   word3_muladd(&w2, &w1, &w0, x[15], y[18]);
   word3_muladd(&w2, &w1, &w0, x[16], y[17]);
   word3_muladd(&w2, &w1, &w0, x[17], y[16]);
   word3_muladd(&w2, &w1, &w0, x[18], y[15]);
   word3_muladd(&w2, &w1, &w0, x[19], y[14]);
   word3_muladd(&w2, &w1, &w0, x[20], y[13]);
   word3_muladd(&w2, &w1, &w0, x[21], y[12]);
   word3_muladd(&w2, &w1, &w0, x[22], y[11]);
   word3_muladd(&w2, &w1, &w0, x[23], y[10]);
   z[33] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[11], y[23]);
   word3_muladd(&w0, &w2, &w1, x[12], y[22]);
   word3_muladd(&w0, &w2, &w1, x[13], y[21]);
   word3_muladd(&w0, &w2, &w1, x[14], y[20]);
   word3_muladd(&w0, &w2, &w1, x[15], y[19]);
   word3_muladd(&w0, &w2, &w1, x[16], y[18]);
   word3_muladd(&w0, &w2, &w1, x[17], y[17]);
   word3_muladd(&w0, &w2, &w1, x[18], y[16]);
   word3_muladd(&w0, &w2, &w1, x[19], y[15]);
   word3_muladd(&w0, &w2, &w1, x[20], y[14]);
   word3_muladd(&w0, &w2, &w1, x[21], y[13]);
   word3_muladd(&w0, &w2, &w1, x[22], y[12]);
   word3_muladd(&w0, &w2, &w1, x[23], y[11]);
   z[34] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[12], y[23]);
   word3_muladd(&w1, &w0, &w2, x[13], y[22]);
   word3_muladd(&w1, &w0, &w2, x[14], y[21]);
   word3_muladd(&w1, &w0, &w2, x[15], y[20]);
   word3_muladd(&w1, &w0, &w2, x[16], y[19]);
   word3_muladd(&w1, &w0, &w2, x[17], y[18]);
   word3_muladd(&w1, &w0, &w2, x[18], y[17]);
   word3_muladd(&w1, &w0, &w2, x[19], y[16]);
   word3_muladd(&w1, &w0, &w2, x[20], y[15]);
   word3_muladd(&w1, &w0, &w2, x[21], y[14]);
   word3_muladd(&w1, &w0, &w2, x[22], y[13]);
   word3_muladd(&w1, &w0, &w2, x[23], y[12]);
   z[35] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[13], y[23]);
   word3_muladd(&w2, &w1, &w0, x[14], y[22]);
   word3_muladd(&w2, &w1, &w0, x[15], y[21]);
   word3_muladd(&w2, &w1, &w0, x[16], y[20]);
   word3_muladd(&w2, &w1, &w0, x[17], y[19]);
   word3_muladd(&w2, &w1, &w0, x[18], y[18]);
   word3_muladd(&w2, &w1, &w0, x[19], y[17]);
   word3_muladd(&w2, &w1, &w0, x[20], y[16]);
   word3_muladd(&w2, &w1, &w0, x[21], y[15]);
   word3_muladd(&w2, &w1, &w0, x[22], y[14]);
   word3_muladd(&w2, &w1, &w0, x[23], y[13]);
   z[36] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[14], y[23]);
   word3_muladd(&w0, &w2, &w1, x[15], y[22]);
   word3_muladd(&w0, &w2, &w1, x[16], y[21]);
   word3_muladd(&w0, &w2, &w1, x[17], y[20]);
   word3_muladd(&w0, &w2, &w1, x[18], y[19]);
   word3_muladd(&w0, &w2, &w1, x[19], y[18]);
   word3_muladd(&w0, &w2, &w1, x[20], y[17]);
   word3_muladd(&w0, &w2, &w1, x[21], y[16]);
   word3_muladd(&w0, &w2, &w1, x[22], y[15]);
   word3_muladd(&w0, &w2, &w1, x[23], y[14]);
   z[37] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[15], y[23]);
   word3_muladd(&w1, &w0, &w2, x[16], y[22]);
   word3_muladd(&w1, &w0, &w2, x[17], y[21]);
   word3_muladd(&w1, &w0, &w2, x[18], y[20]);
   word3_muladd(&w1, &w0, &w2, x[19], y[19]);
   word3_muladd(&w1, &w0, &w2, x[20], y[18]);
   word3_muladd(&w1, &w0, &w2, x[21], y[17]);
   word3_muladd(&w1, &w0, &w2, x[22], y[16]);
   word3_muladd(&w1, &w0, &w2, x[23], y[15]);
   z[38] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[16], y[23]);
   word3_muladd(&w2, &w1, &w0, x[17], y[22]);
   word3_muladd(&w2, &w1, &w0, x[18], y[21]);
   word3_muladd(&w2, &w1, &w0, x[19], y[20]);
   word3_muladd(&w2, &w1, &w0, x[20], y[19]);
   word3_muladd(&w2, &w1, &w0, x[21], y[18]);
   word3_muladd(&w2, &w1, &w0, x[22], y[17]);
   word3_muladd(&w2, &w1, &w0, x[23], y[16]);
   z[39] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[17], y[23]);
   word3_muladd(&w0, &w2, &w1, x[18], y[22]);
   word3_muladd(&w0, &w2, &w1, x[19], y[21]);
   word3_muladd(&w0, &w2, &w1, x[20], y[20]);
   word3_muladd(&w0, &w2, &w1, x[21], y[19]);
   word3_muladd(&w0, &w2, &w1, x[22], y[18]);
   word3_muladd(&w0, &w2, &w1, x[23], y[17]);
   z[40] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[18], y[23]);
   word3_muladd(&w1, &w0, &w2, x[19], y[22]);
   word3_muladd(&w1, &w0, &w2, x[20], y[21]);
   word3_muladd(&w1, &w0, &w2, x[21], y[20]);
   word3_muladd(&w1, &w0, &w2, x[22], y[19]);
   word3_muladd(&w1, &w0, &w2, x[23], y[18]);
   z[41] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[19], y[23]);
   word3_muladd(&w2, &w1, &w0, x[20], y[22]);
   word3_muladd(&w2, &w1, &w0, x[21], y[21]);
   word3_muladd(&w2, &w1, &w0, x[22], y[20]);
   word3_muladd(&w2, &w1, &w0, x[23], y[19]);
   z[42] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[20], y[23]);
   word3_muladd(&w0, &w2, &w1, x[21], y[22]);
   word3_muladd(&w0, &w2, &w1, x[22], y[21]);
   word3_muladd(&w0, &w2, &w1, x[23], y[20]);
   z[43] = w1; w1 = 0;

   word3_muladd(&w1, &w0, &w2, x[21], y[23]);
   word3_muladd(&w1, &w0, &w2, x[22], y[22]);
   word3_muladd(&w1, &w0, &w2, x[23], y[21]);
   z[44] = w2; w2 = 0;

   word3_muladd(&w2, &w1, &w0, x[22], y[23]);
   word3_muladd(&w2, &w1, &w0, x[23], y[22]);
   z[45] = w0; w0 = 0;

   word3_muladd(&w0, &w2, &w1, x[23], y[23]);
   z[46] = w1;
   z[47] = w2;
   }

}
/*
* Multiplication and Squaring
* (C) 1999-2010,2018 Jack Lloyd
*     2016 Matthias Gierlings
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

const size_t KARATSUBA_MULTIPLY_THRESHOLD = 32;
const size_t KARATSUBA_SQUARE_THRESHOLD = 32;

/*
* Simple O(N^2) Multiplication
*/
void basecase_mul(word z[], size_t z_size,
                  const word x[], size_t x_size,
                  const word y[], size_t y_size)
   {
   if(z_size < x_size + y_size)
      throw Invalid_Argument("basecase_mul z_size too small");

   const size_t x_size_8 = x_size - (x_size % 8);

   clear_mem(z, z_size);

   for(size_t i = 0; i != y_size; ++i)
      {
      const word y_i = y[i];

      word carry = 0;

      for(size_t j = 0; j != x_size_8; j += 8)
         carry = word8_madd3(z + i + j, x + j, y_i, carry);

      for(size_t j = x_size_8; j != x_size; ++j)
         z[i+j] = word_madd3(x[j], y_i, z[i+j], &carry);

      z[x_size+i] = carry;
      }
   }

void basecase_sqr(word z[], size_t z_size,
                  const word x[], size_t x_size)
   {
   if(z_size < 2*x_size)
      throw Invalid_Argument("basecase_sqr z_size too small");

   const size_t x_size_8 = x_size - (x_size % 8);

   clear_mem(z, z_size);

   for(size_t i = 0; i != x_size; ++i)
      {
      const word x_i = x[i];

      word carry = 0;

      for(size_t j = 0; j != x_size_8; j += 8)
         carry = word8_madd3(z + i + j, x + j, x_i, carry);

      for(size_t j = x_size_8; j != x_size; ++j)
         z[i+j] = word_madd3(x[j], x_i, z[i+j], &carry);

      z[x_size+i] = carry;
      }
   }

/*
* Karatsuba Multiplication Operation
*/
void karatsuba_mul(word z[], const word x[], const word y[], size_t N,
                   word workspace[])
   {
   if(N < KARATSUBA_MULTIPLY_THRESHOLD || N % 2)
      {
      switch(N)
         {
         case 6:
            return bigint_comba_mul6(z, x, y);
         case 8:
            return bigint_comba_mul8(z, x, y);
         case 9:
            return bigint_comba_mul9(z, x, y);
         case 16:
            return bigint_comba_mul16(z, x, y);
         case 24:
            return bigint_comba_mul24(z, x, y);
         default:
            return basecase_mul(z, 2*N, x, N, y, N);
         }
      }

   const size_t N2 = N / 2;

   const word* x0 = x;
   const word* x1 = x + N2;
   const word* y0 = y;
   const word* y1 = y + N2;
   word* z0 = z;
   word* z1 = z + N;

   word* ws0 = workspace;
   word* ws1 = workspace + N;

   clear_mem(workspace, 2*N);

   /*
   * If either of cmp0 or cmp1 is zero then z0 or z1 resp is zero here,
   * resulting in a no-op - z0*z1 will be equal to zero so we don't need to do
   * anything, clear_mem above already set the correct result.
   *
   * However we ignore the result of the comparisons and always perform the
   * subtractions and recursively multiply to avoid the timing channel.
   */

   // First compute (X_lo - X_hi)*(Y_hi - Y_lo)
   const auto cmp0 = bigint_sub_abs(z0, x0, x1, N2, workspace);
   const auto cmp1 = bigint_sub_abs(z1, y1, y0, N2, workspace);
   const auto neg_mask = ~(cmp0 ^ cmp1);

   karatsuba_mul(ws0, z0, z1, N2, ws1);

   // Compute X_lo * Y_lo
   karatsuba_mul(z0, x0, y0, N2, ws1);

   // Compute X_hi * Y_hi
   karatsuba_mul(z1, x1, y1, N2, ws1);

   const word ws_carry = bigint_add3_nc(ws1, z0, N, z1, N);
   word z_carry = bigint_add2_nc(z + N2, N, ws1, N);

   z_carry += bigint_add2_nc(z + N + N2, N2, &ws_carry, 1);
   bigint_add2_nc(z + N + N2, N2, &z_carry, 1);

   clear_mem(workspace + N, N2);

   bigint_cnd_add_or_sub(neg_mask, z + N2, workspace, 2*N-N2);
   }

/*
* Karatsuba Squaring Operation
*/
void karatsuba_sqr(word z[], const word x[], size_t N, word workspace[])
   {
   if(N < KARATSUBA_SQUARE_THRESHOLD || N % 2)
      {
      switch(N)
         {
         case 6:
            return bigint_comba_sqr6(z, x);
         case 8:
            return bigint_comba_sqr8(z, x);
         case 9:
            return bigint_comba_sqr9(z, x);
         case 16:
            return bigint_comba_sqr16(z, x);
         case 24:
            return bigint_comba_sqr24(z, x);
         default:
            return basecase_sqr(z, 2*N, x, N);
         }
      }

   const size_t N2 = N / 2;

   const word* x0 = x;
   const word* x1 = x + N2;
   word* z0 = z;
   word* z1 = z + N;

   word* ws0 = workspace;
   word* ws1 = workspace + N;

   clear_mem(workspace, 2*N);

   // See comment in karatsuba_mul
   bigint_sub_abs(z0, x0, x1, N2, workspace);
   karatsuba_sqr(ws0, z0, N2, ws1);

   karatsuba_sqr(z0, x0, N2, ws1);
   karatsuba_sqr(z1, x1, N2, ws1);

   const word ws_carry = bigint_add3_nc(ws1, z0, N, z1, N);
   word z_carry = bigint_add2_nc(z + N2, N, ws1, N);

   z_carry += bigint_add2_nc(z + N + N2, N2, &ws_carry, 1);
   bigint_add2_nc(z + N + N2, N2, &z_carry, 1);

   /*
   * This is only actually required if cmp (result of bigint_sub_abs) is != 0,
   * however if cmp==0 then ws0[0:N] == 0 and avoiding the jump hides a
   * timing channel.
   */
   bigint_sub2(z + N2, 2*N-N2, ws0, N);
   }

/*
* Pick a good size for the Karatsuba multiply
*/
size_t karatsuba_size(size_t z_size,
                      size_t x_size, size_t x_sw,
                      size_t y_size, size_t y_sw)
   {
   if(x_sw > x_size || x_sw > y_size || y_sw > x_size || y_sw > y_size)
      return 0;

   if(((x_size == x_sw) && (x_size % 2)) ||
      ((y_size == y_sw) && (y_size % 2)))
      return 0;

   const size_t start = (x_sw > y_sw) ? x_sw : y_sw;
   const size_t end = (x_size < y_size) ? x_size : y_size;

   if(start == end)
      {
      if(start % 2)
         return 0;
      return start;
      }

   for(size_t j = start; j <= end; ++j)
      {
      if(j % 2)
         continue;

      if(2*j > z_size)
         return 0;

      if(x_sw <= j && j <= x_size && y_sw <= j && j <= y_size)
         {
         if(j % 4 == 2 &&
            (j+2) <= x_size && (j+2) <= y_size && 2*(j+2) <= z_size)
            return j+2;
         return j;
         }
      }

   return 0;
   }

/*
* Pick a good size for the Karatsuba squaring
*/
size_t karatsuba_size(size_t z_size, size_t x_size, size_t x_sw)
   {
   if(x_sw == x_size)
      {
      if(x_sw % 2)
         return 0;
      return x_sw;
      }

   for(size_t j = x_sw; j <= x_size; ++j)
      {
      if(j % 2)
         continue;

      if(2*j > z_size)
         return 0;

      if(j % 4 == 2 && (j+2) <= x_size && 2*(j+2) <= z_size)
         return j+2;
      return j;
      }

   return 0;
   }

template<size_t SZ>
inline bool sized_for_comba_mul(size_t x_sw, size_t x_size,
                                size_t y_sw, size_t y_size,
                                size_t z_size)
   {
   return (x_sw <= SZ && x_size >= SZ &&
           y_sw <= SZ && y_size >= SZ &&
           z_size >= 2*SZ);
   }

template<size_t SZ>
inline bool sized_for_comba_sqr(size_t x_sw, size_t x_size,
                                size_t z_size)
   {
   return (x_sw <= SZ && x_size >= SZ && z_size >= 2*SZ);
   }

}

void bigint_mul(word z[], size_t z_size,
                const word x[], size_t x_size, size_t x_sw,
                const word y[], size_t y_size, size_t y_sw,
                word workspace[], size_t ws_size)
   {
   clear_mem(z, z_size);

   if(x_sw == 1)
      {
      bigint_linmul3(z, y, y_sw, x[0]);
      }
   else if(y_sw == 1)
      {
      bigint_linmul3(z, x, x_sw, y[0]);
      }
   else if(sized_for_comba_mul<4>(x_sw, x_size, y_sw, y_size, z_size))
      {
      bigint_comba_mul4(z, x, y);
      }
   else if(sized_for_comba_mul<6>(x_sw, x_size, y_sw, y_size, z_size))
      {
      bigint_comba_mul6(z, x, y);
      }
   else if(sized_for_comba_mul<8>(x_sw, x_size, y_sw, y_size, z_size))
      {
      bigint_comba_mul8(z, x, y);
      }
   else if(sized_for_comba_mul<9>(x_sw, x_size, y_sw, y_size, z_size))
      {
      bigint_comba_mul9(z, x, y);
      }
   else if(sized_for_comba_mul<16>(x_sw, x_size, y_sw, y_size, z_size))
      {
      bigint_comba_mul16(z, x, y);
      }
   else if(sized_for_comba_mul<24>(x_sw, x_size, y_sw, y_size, z_size))
      {
      bigint_comba_mul24(z, x, y);
      }
   else if(x_sw < KARATSUBA_MULTIPLY_THRESHOLD ||
           y_sw < KARATSUBA_MULTIPLY_THRESHOLD ||
           !workspace)
      {
      basecase_mul(z, z_size, x, x_sw, y, y_sw);
      }
   else
      {
      const size_t N = karatsuba_size(z_size, x_size, x_sw, y_size, y_sw);

      if(N && z_size >= 2*N && ws_size >= 2*N)
         karatsuba_mul(z, x, y, N, workspace);
      else
         basecase_mul(z, z_size, x, x_sw, y, y_sw);
      }
   }

/*
* Squaring Algorithm Dispatcher
*/
void bigint_sqr(word z[], size_t z_size,
                const word x[], size_t x_size, size_t x_sw,
                word workspace[], size_t ws_size)
   {
   clear_mem(z, z_size);

   BOTAN_ASSERT(z_size/2 >= x_sw, "Output size is sufficient");

   if(x_sw == 1)
      {
      bigint_linmul3(z, x, x_sw, x[0]);
      }
   else if(sized_for_comba_sqr<4>(x_sw, x_size, z_size))
      {
      bigint_comba_sqr4(z, x);
      }
   else if(sized_for_comba_sqr<6>(x_sw, x_size, z_size))
      {
      bigint_comba_sqr6(z, x);
      }
   else if(sized_for_comba_sqr<8>(x_sw, x_size, z_size))
      {
      bigint_comba_sqr8(z, x);
      }
   else if(sized_for_comba_sqr<9>(x_sw, x_size, z_size))
      {
      bigint_comba_sqr9(z, x);
      }
   else if(sized_for_comba_sqr<16>(x_sw, x_size, z_size))
      {
      bigint_comba_sqr16(z, x);
      }
   else if(sized_for_comba_sqr<24>(x_sw, x_size, z_size))
      {
      bigint_comba_sqr24(z, x);
      }
   else if(x_size < KARATSUBA_SQUARE_THRESHOLD || !workspace)
      {
      basecase_sqr(z, z_size, x, x_sw);
      }
   else
      {
      const size_t N = karatsuba_size(z_size, x_size, x_sw);

      if(N && z_size >= 2*N && ws_size >= 2*N)
         karatsuba_sqr(z, x, N, workspace);
      else
         basecase_sqr(z, z_size, x, x_sw);
      }
   }

}
/*
* Montgomery Reduction
* (C) 1999-2011 Jack Lloyd
*     2006 Luca Piccarreta
*     2016 Matthias Gierlings
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

/*
* Montgomery reduction - product scanning form
*
* https://www.iacr.org/archive/ches2005/006.pdf
* https://eprint.iacr.org/2013/882.pdf
* https://www.microsoft.com/en-us/research/wp-content/uploads/1996/01/j37acmon.pdf
*/
void bigint_monty_redc_generic(word z[], size_t z_size,
                               const word p[], size_t p_size, word p_dash,
                               word ws[])
   {
   word w2 = 0, w1 = 0, w0 = 0;

   w0 = z[0];

   ws[0] = w0 * p_dash;

   word3_muladd(&w2, &w1, &w0, ws[0], p[0]);

   w0 = w1;
   w1 = w2;
   w2 = 0;

   for(size_t i = 1; i != p_size; ++i)
      {
      for(size_t j = 0; j < i; ++j)
         {
         word3_muladd(&w2, &w1, &w0, ws[j], p[i-j]);
         }

      word3_add(&w2, &w1, &w0, z[i]);

      ws[i] = w0 * p_dash;

      word3_muladd(&w2, &w1, &w0, ws[i], p[0]);

      w0 = w1;
      w1 = w2;
      w2 = 0;
      }

   for(size_t i = 0; i != p_size; ++i)
      {
      for(size_t j = i + 1; j != p_size; ++j)
         {
         word3_muladd(&w2, &w1, &w0, ws[j], p[p_size + i-j]);
         }

      word3_add(&w2, &w1, &w0, z[p_size+i]);

      ws[i] = w0;
      w0 = w1;
      w1 = w2;
      w2 = 0;
      }

   word3_add(&w2, &w1, &w0, z[z_size-1]);

   ws[p_size] = w0;
   ws[p_size+1] = w1;

   /*
   * The result might need to be reduced mod p. To avoid a timing
   * channel, always perform the subtraction. If in the compution
   * of x - p a borrow is required then x was already < p.
   *
   * x starts at ws[0] and is p_size+1 bytes long.
   * x - p starts at ws[p_size+1] and is also p_size+1 bytes log
   *
   * Select which address to copy from indexing off of the final
   * borrow.
   */

   // word borrow = bigint_sub3(ws + p_size + 1, ws, p_size + 1, p, p_size);
   word borrow = 0;
   for(size_t i = 0; i != p_size; ++i)
      ws[p_size + 1 + i] = word_sub(ws[i], p[i], &borrow);
   ws[2*p_size+1] = word_sub(ws[p_size], 0, &borrow);

   BOTAN_DEBUG_ASSERT(borrow == 0 || borrow == 1);

   CT::conditional_copy_mem(borrow, z, ws, ws + (p_size + 1), (p_size + 1));
   clear_mem(z + p_size, z_size - p_size - 2);
   }

}

void bigint_monty_redc(word z[],
                       const word p[], size_t p_size, word p_dash,
                       word ws[], size_t ws_size)
   {
   const size_t z_size = 2*(p_size+1);

   BOTAN_ARG_CHECK(ws_size >= z_size, "workspace too small");

   if(p_size == 4)
      bigint_monty_redc_4(z, p, p_dash, ws);
   else if(p_size == 6)
      bigint_monty_redc_6(z, p, p_dash, ws);
   else if(p_size == 8)
      bigint_monty_redc_8(z, p, p_dash, ws);
   else if(p_size == 16)
      bigint_monty_redc_16(z, p, p_dash, ws);
   else if(p_size == 24)
      bigint_monty_redc_24(z, p, p_dash, ws);
   else if(p_size == 32)
      bigint_monty_redc_32(z, p, p_dash, ws);
   else
      bigint_monty_redc_generic(z, z_size, p, p_size, p_dash, ws);
   }

}
/*
* This file was automatically generated by ./src/scripts/monty.py on 2018-06-11
* All manual changes will be lost. Edit the script instead.
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

void bigint_monty_redc_4(word z[], const word p[4], word p_dash, word ws[])
   {
   word w2 = 0, w1 = 0, w0 = 0;
   w0 = z[0];
   ws[0] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[0], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[1]);
   word3_add(&w2, &w1, &w0, z[1]);
   ws[1] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[1], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[1]);
   word3_add(&w2, &w1, &w0, z[2]);
   ws[2] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[2], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[1]);
   word3_add(&w2, &w1, &w0, z[3]);
   ws[3] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[3], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[1], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[1]);
   word3_add(&w2, &w1, &w0, z[4]);
   ws[0] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[2], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[2]);
   word3_add(&w2, &w1, &w0, z[5]);
   ws[1] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[3], p[3]);
   word3_add(&w2, &w1, &w0, z[6]);
   ws[2] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[7]);
   ws[3] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[9]);
   ws[4] = w0;
   ws[5] = w1;
   word borrow = 0;
   ws[5] = word_sub(ws[0], p[0], &borrow);
   ws[6] = word_sub(ws[1], p[1], &borrow);
   ws[7] = word_sub(ws[2], p[2], &borrow);
   ws[8] = word_sub(ws[3], p[3], &borrow);
   ws[9] = word_sub(ws[4], 0, &borrow);
   CT::conditional_copy_mem(borrow, z, ws, ws + 5, 5);
   clear_mem(z + 4, 2*(4+1) - 4);
   }

void bigint_monty_redc_6(word z[], const word p[6], word p_dash, word ws[])
   {
   word w2 = 0, w1 = 0, w0 = 0;
   w0 = z[0];
   ws[0] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[0], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[1]);
   word3_add(&w2, &w1, &w0, z[1]);
   ws[1] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[1], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[1]);
   word3_add(&w2, &w1, &w0, z[2]);
   ws[2] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[2], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[1]);
   word3_add(&w2, &w1, &w0, z[3]);
   ws[3] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[3], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[1]);
   word3_add(&w2, &w1, &w0, z[4]);
   ws[4] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[4], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[1]);
   word3_add(&w2, &w1, &w0, z[5]);
   ws[5] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[5], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[1], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[1]);
   word3_add(&w2, &w1, &w0, z[6]);
   ws[0] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[2], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[2]);
   word3_add(&w2, &w1, &w0, z[7]);
   ws[1] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[3], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[3]);
   word3_add(&w2, &w1, &w0, z[8]);
   ws[2] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[4], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[4]);
   word3_add(&w2, &w1, &w0, z[9]);
   ws[3] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[5], p[5]);
   word3_add(&w2, &w1, &w0, z[10]);
   ws[4] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[11]);
   ws[5] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[13]);
   ws[6] = w0;
   ws[7] = w1;
   word borrow = 0;
   ws[7] = word_sub(ws[0], p[0], &borrow);
   ws[8] = word_sub(ws[1], p[1], &borrow);
   ws[9] = word_sub(ws[2], p[2], &borrow);
   ws[10] = word_sub(ws[3], p[3], &borrow);
   ws[11] = word_sub(ws[4], p[4], &borrow);
   ws[12] = word_sub(ws[5], p[5], &borrow);
   ws[13] = word_sub(ws[6], 0, &borrow);
   CT::conditional_copy_mem(borrow, z, ws, ws + 7, 7);
   clear_mem(z + 6, 2*(6+1) - 6);
   }

void bigint_monty_redc_8(word z[], const word p[8], word p_dash, word ws[])
   {
   word w2 = 0, w1 = 0, w0 = 0;
   w0 = z[0];
   ws[0] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[0], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[1]);
   word3_add(&w2, &w1, &w0, z[1]);
   ws[1] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[1], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[1]);
   word3_add(&w2, &w1, &w0, z[2]);
   ws[2] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[2], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[1]);
   word3_add(&w2, &w1, &w0, z[3]);
   ws[3] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[3], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[1]);
   word3_add(&w2, &w1, &w0, z[4]);
   ws[4] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[4], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[1]);
   word3_add(&w2, &w1, &w0, z[5]);
   ws[5] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[5], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[1]);
   word3_add(&w2, &w1, &w0, z[6]);
   ws[6] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[6], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[1]);
   word3_add(&w2, &w1, &w0, z[7]);
   ws[7] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[7], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[1], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[1]);
   word3_add(&w2, &w1, &w0, z[8]);
   ws[0] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[2], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[2]);
   word3_add(&w2, &w1, &w0, z[9]);
   ws[1] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[3], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[3]);
   word3_add(&w2, &w1, &w0, z[10]);
   ws[2] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[4], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[4]);
   word3_add(&w2, &w1, &w0, z[11]);
   ws[3] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[5], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[5]);
   word3_add(&w2, &w1, &w0, z[12]);
   ws[4] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[6], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[6]);
   word3_add(&w2, &w1, &w0, z[13]);
   ws[5] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[7], p[7]);
   word3_add(&w2, &w1, &w0, z[14]);
   ws[6] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[15]);
   ws[7] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[17]);
   ws[8] = w0;
   ws[9] = w1;
   word borrow = 0;
   ws[9] = word_sub(ws[0], p[0], &borrow);
   ws[10] = word_sub(ws[1], p[1], &borrow);
   ws[11] = word_sub(ws[2], p[2], &borrow);
   ws[12] = word_sub(ws[3], p[3], &borrow);
   ws[13] = word_sub(ws[4], p[4], &borrow);
   ws[14] = word_sub(ws[5], p[5], &borrow);
   ws[15] = word_sub(ws[6], p[6], &borrow);
   ws[16] = word_sub(ws[7], p[7], &borrow);
   ws[17] = word_sub(ws[8], 0, &borrow);
   CT::conditional_copy_mem(borrow, z, ws, ws + 9, 9);
   clear_mem(z + 8, 2*(8+1) - 8);
   }

void bigint_monty_redc_16(word z[], const word p[16], word p_dash, word ws[])
   {
   word w2 = 0, w1 = 0, w0 = 0;
   w0 = z[0];
   ws[0] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[0], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[1]);
   word3_add(&w2, &w1, &w0, z[1]);
   ws[1] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[1], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[1]);
   word3_add(&w2, &w1, &w0, z[2]);
   ws[2] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[2], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[1]);
   word3_add(&w2, &w1, &w0, z[3]);
   ws[3] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[3], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[1]);
   word3_add(&w2, &w1, &w0, z[4]);
   ws[4] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[4], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[1]);
   word3_add(&w2, &w1, &w0, z[5]);
   ws[5] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[5], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[1]);
   word3_add(&w2, &w1, &w0, z[6]);
   ws[6] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[6], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[1]);
   word3_add(&w2, &w1, &w0, z[7]);
   ws[7] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[7], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[1]);
   word3_add(&w2, &w1, &w0, z[8]);
   ws[8] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[8], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[1]);
   word3_add(&w2, &w1, &w0, z[9]);
   ws[9] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[9], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[1]);
   word3_add(&w2, &w1, &w0, z[10]);
   ws[10] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[10], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[1]);
   word3_add(&w2, &w1, &w0, z[11]);
   ws[11] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[11], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[1]);
   word3_add(&w2, &w1, &w0, z[12]);
   ws[12] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[12], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[1]);
   word3_add(&w2, &w1, &w0, z[13]);
   ws[13] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[13], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[1]);
   word3_add(&w2, &w1, &w0, z[14]);
   ws[14] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[14], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[1]);
   word3_add(&w2, &w1, &w0, z[15]);
   ws[15] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[15], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[1], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[1]);
   word3_add(&w2, &w1, &w0, z[16]);
   ws[0] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[2], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[2]);
   word3_add(&w2, &w1, &w0, z[17]);
   ws[1] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[3], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[3]);
   word3_add(&w2, &w1, &w0, z[18]);
   ws[2] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[4], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[4]);
   word3_add(&w2, &w1, &w0, z[19]);
   ws[3] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[5], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[5]);
   word3_add(&w2, &w1, &w0, z[20]);
   ws[4] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[6], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[6]);
   word3_add(&w2, &w1, &w0, z[21]);
   ws[5] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[7], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[7]);
   word3_add(&w2, &w1, &w0, z[22]);
   ws[6] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[8], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[8]);
   word3_add(&w2, &w1, &w0, z[23]);
   ws[7] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[9], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[9]);
   word3_add(&w2, &w1, &w0, z[24]);
   ws[8] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[10], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[10]);
   word3_add(&w2, &w1, &w0, z[25]);
   ws[9] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[11], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[11]);
   word3_add(&w2, &w1, &w0, z[26]);
   ws[10] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[12], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[12]);
   word3_add(&w2, &w1, &w0, z[27]);
   ws[11] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[13], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[13]);
   word3_add(&w2, &w1, &w0, z[28]);
   ws[12] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[14], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[14]);
   word3_add(&w2, &w1, &w0, z[29]);
   ws[13] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[15], p[15]);
   word3_add(&w2, &w1, &w0, z[30]);
   ws[14] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[31]);
   ws[15] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[33]);
   ws[16] = w0;
   ws[17] = w1;
   word borrow = bigint_sub3(ws + 16 + 1, ws, 16 + 1, p, 16);
   CT::conditional_copy_mem(borrow, z, ws, ws + 17, 17);
   clear_mem(z + 16, 2*(16+1) - 16);
   }

void bigint_monty_redc_24(word z[], const word p[24], word p_dash, word ws[])
   {
   word w2 = 0, w1 = 0, w0 = 0;
   w0 = z[0];
   ws[0] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[0], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[1]);
   word3_add(&w2, &w1, &w0, z[1]);
   ws[1] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[1], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[1]);
   word3_add(&w2, &w1, &w0, z[2]);
   ws[2] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[2], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[1]);
   word3_add(&w2, &w1, &w0, z[3]);
   ws[3] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[3], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[1]);
   word3_add(&w2, &w1, &w0, z[4]);
   ws[4] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[4], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[1]);
   word3_add(&w2, &w1, &w0, z[5]);
   ws[5] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[5], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[1]);
   word3_add(&w2, &w1, &w0, z[6]);
   ws[6] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[6], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[1]);
   word3_add(&w2, &w1, &w0, z[7]);
   ws[7] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[7], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[1]);
   word3_add(&w2, &w1, &w0, z[8]);
   ws[8] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[8], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[1]);
   word3_add(&w2, &w1, &w0, z[9]);
   ws[9] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[9], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[1]);
   word3_add(&w2, &w1, &w0, z[10]);
   ws[10] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[10], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[1]);
   word3_add(&w2, &w1, &w0, z[11]);
   ws[11] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[11], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[1]);
   word3_add(&w2, &w1, &w0, z[12]);
   ws[12] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[12], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[1]);
   word3_add(&w2, &w1, &w0, z[13]);
   ws[13] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[13], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[1]);
   word3_add(&w2, &w1, &w0, z[14]);
   ws[14] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[14], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[1]);
   word3_add(&w2, &w1, &w0, z[15]);
   ws[15] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[15], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[1]);
   word3_add(&w2, &w1, &w0, z[16]);
   ws[16] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[16], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[1]);
   word3_add(&w2, &w1, &w0, z[17]);
   ws[17] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[17], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[1]);
   word3_add(&w2, &w1, &w0, z[18]);
   ws[18] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[18], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[1]);
   word3_add(&w2, &w1, &w0, z[19]);
   ws[19] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[19], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[1]);
   word3_add(&w2, &w1, &w0, z[20]);
   ws[20] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[20], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[1]);
   word3_add(&w2, &w1, &w0, z[21]);
   ws[21] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[21], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[1]);
   word3_add(&w2, &w1, &w0, z[22]);
   ws[22] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[22], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[1]);
   word3_add(&w2, &w1, &w0, z[23]);
   ws[23] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[23], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[1], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[1]);
   word3_add(&w2, &w1, &w0, z[24]);
   ws[0] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[2], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[2]);
   word3_add(&w2, &w1, &w0, z[25]);
   ws[1] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[3], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[3]);
   word3_add(&w2, &w1, &w0, z[26]);
   ws[2] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[4], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[4]);
   word3_add(&w2, &w1, &w0, z[27]);
   ws[3] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[5], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[5]);
   word3_add(&w2, &w1, &w0, z[28]);
   ws[4] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[6], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[6]);
   word3_add(&w2, &w1, &w0, z[29]);
   ws[5] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[7], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[7]);
   word3_add(&w2, &w1, &w0, z[30]);
   ws[6] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[8], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[8]);
   word3_add(&w2, &w1, &w0, z[31]);
   ws[7] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[9], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[9]);
   word3_add(&w2, &w1, &w0, z[32]);
   ws[8] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[10], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[10]);
   word3_add(&w2, &w1, &w0, z[33]);
   ws[9] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[11], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[11]);
   word3_add(&w2, &w1, &w0, z[34]);
   ws[10] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[12], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[12]);
   word3_add(&w2, &w1, &w0, z[35]);
   ws[11] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[13], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[13]);
   word3_add(&w2, &w1, &w0, z[36]);
   ws[12] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[14], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[14]);
   word3_add(&w2, &w1, &w0, z[37]);
   ws[13] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[15], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[15]);
   word3_add(&w2, &w1, &w0, z[38]);
   ws[14] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[16], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[16]);
   word3_add(&w2, &w1, &w0, z[39]);
   ws[15] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[17], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[17]);
   word3_add(&w2, &w1, &w0, z[40]);
   ws[16] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[18], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[18]);
   word3_add(&w2, &w1, &w0, z[41]);
   ws[17] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[19], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[19]);
   word3_add(&w2, &w1, &w0, z[42]);
   ws[18] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[20], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[20]);
   word3_add(&w2, &w1, &w0, z[43]);
   ws[19] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[21], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[21]);
   word3_add(&w2, &w1, &w0, z[44]);
   ws[20] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[22], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[22]);
   word3_add(&w2, &w1, &w0, z[45]);
   ws[21] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[23], p[23]);
   word3_add(&w2, &w1, &w0, z[46]);
   ws[22] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[47]);
   ws[23] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[49]);
   ws[24] = w0;
   ws[25] = w1;
   word borrow = bigint_sub3(ws + 24 + 1, ws, 24 + 1, p, 24);
   CT::conditional_copy_mem(borrow, z, ws, ws + 25, 25);
   clear_mem(z + 24, 2*(24+1) - 24);
   }

void bigint_monty_redc_32(word z[], const word p[32], word p_dash, word ws[])
   {
   word w2 = 0, w1 = 0, w0 = 0;
   w0 = z[0];
   ws[0] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[0], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[1]);
   word3_add(&w2, &w1, &w0, z[1]);
   ws[1] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[1], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[1]);
   word3_add(&w2, &w1, &w0, z[2]);
   ws[2] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[2], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[1]);
   word3_add(&w2, &w1, &w0, z[3]);
   ws[3] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[3], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[1]);
   word3_add(&w2, &w1, &w0, z[4]);
   ws[4] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[4], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[1]);
   word3_add(&w2, &w1, &w0, z[5]);
   ws[5] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[5], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[1]);
   word3_add(&w2, &w1, &w0, z[6]);
   ws[6] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[6], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[1]);
   word3_add(&w2, &w1, &w0, z[7]);
   ws[7] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[7], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[1]);
   word3_add(&w2, &w1, &w0, z[8]);
   ws[8] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[8], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[1]);
   word3_add(&w2, &w1, &w0, z[9]);
   ws[9] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[9], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[1]);
   word3_add(&w2, &w1, &w0, z[10]);
   ws[10] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[10], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[1]);
   word3_add(&w2, &w1, &w0, z[11]);
   ws[11] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[11], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[1]);
   word3_add(&w2, &w1, &w0, z[12]);
   ws[12] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[12], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[1]);
   word3_add(&w2, &w1, &w0, z[13]);
   ws[13] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[13], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[1]);
   word3_add(&w2, &w1, &w0, z[14]);
   ws[14] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[14], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[1]);
   word3_add(&w2, &w1, &w0, z[15]);
   ws[15] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[15], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[1]);
   word3_add(&w2, &w1, &w0, z[16]);
   ws[16] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[16], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[1]);
   word3_add(&w2, &w1, &w0, z[17]);
   ws[17] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[17], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[1]);
   word3_add(&w2, &w1, &w0, z[18]);
   ws[18] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[18], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[1]);
   word3_add(&w2, &w1, &w0, z[19]);
   ws[19] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[19], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[1]);
   word3_add(&w2, &w1, &w0, z[20]);
   ws[20] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[20], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[1]);
   word3_add(&w2, &w1, &w0, z[21]);
   ws[21] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[21], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[1]);
   word3_add(&w2, &w1, &w0, z[22]);
   ws[22] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[22], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[1]);
   word3_add(&w2, &w1, &w0, z[23]);
   ws[23] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[23], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[1]);
   word3_add(&w2, &w1, &w0, z[24]);
   ws[24] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[24], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[1]);
   word3_add(&w2, &w1, &w0, z[25]);
   ws[25] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[25], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[1]);
   word3_add(&w2, &w1, &w0, z[26]);
   ws[26] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[26], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[1]);
   word3_add(&w2, &w1, &w0, z[27]);
   ws[27] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[27], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[1]);
   word3_add(&w2, &w1, &w0, z[28]);
   ws[28] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[28], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[1]);
   word3_add(&w2, &w1, &w0, z[29]);
   ws[29] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[29], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[1]);
   word3_add(&w2, &w1, &w0, z[30]);
   ws[30] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[30], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[0], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[1], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[1]);
   word3_add(&w2, &w1, &w0, z[31]);
   ws[31] = w0 * p_dash;
   word3_muladd(&w2, &w1, &w0, ws[31], p[0]);
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[1], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[2], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[2]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[1]);
   word3_add(&w2, &w1, &w0, z[32]);
   ws[0] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[2], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[3], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[3]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[2]);
   word3_add(&w2, &w1, &w0, z[33]);
   ws[1] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[3], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[4], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[4]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[3]);
   word3_add(&w2, &w1, &w0, z[34]);
   ws[2] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[4], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[5], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[5]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[4]);
   word3_add(&w2, &w1, &w0, z[35]);
   ws[3] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[5], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[6], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[6]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[5]);
   word3_add(&w2, &w1, &w0, z[36]);
   ws[4] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[6], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[7], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[7]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[6]);
   word3_add(&w2, &w1, &w0, z[37]);
   ws[5] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[7], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[8], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[8]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[7]);
   word3_add(&w2, &w1, &w0, z[38]);
   ws[6] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[8], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[9], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[9]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[8]);
   word3_add(&w2, &w1, &w0, z[39]);
   ws[7] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[9], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[10], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[10]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[9]);
   word3_add(&w2, &w1, &w0, z[40]);
   ws[8] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[10], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[11], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[11]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[10]);
   word3_add(&w2, &w1, &w0, z[41]);
   ws[9] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[11], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[12], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[12]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[11]);
   word3_add(&w2, &w1, &w0, z[42]);
   ws[10] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[12], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[13], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[13]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[12]);
   word3_add(&w2, &w1, &w0, z[43]);
   ws[11] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[13], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[14], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[14]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[13]);
   word3_add(&w2, &w1, &w0, z[44]);
   ws[12] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[14], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[15], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[15]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[14]);
   word3_add(&w2, &w1, &w0, z[45]);
   ws[13] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[15], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[16], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[16]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[15]);
   word3_add(&w2, &w1, &w0, z[46]);
   ws[14] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[16], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[17], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[17]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[16]);
   word3_add(&w2, &w1, &w0, z[47]);
   ws[15] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[17], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[18], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[18]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[17]);
   word3_add(&w2, &w1, &w0, z[48]);
   ws[16] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[18], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[19], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[19]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[18]);
   word3_add(&w2, &w1, &w0, z[49]);
   ws[17] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[19], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[20], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[20]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[19]);
   word3_add(&w2, &w1, &w0, z[50]);
   ws[18] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[20], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[21], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[21]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[20]);
   word3_add(&w2, &w1, &w0, z[51]);
   ws[19] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[21], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[22], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[22]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[21]);
   word3_add(&w2, &w1, &w0, z[52]);
   ws[20] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[22], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[23], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[23]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[22]);
   word3_add(&w2, &w1, &w0, z[53]);
   ws[21] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[23], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[24], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[24]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[23]);
   word3_add(&w2, &w1, &w0, z[54]);
   ws[22] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[24], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[25], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[25]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[24]);
   word3_add(&w2, &w1, &w0, z[55]);
   ws[23] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[25], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[26], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[26]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[25]);
   word3_add(&w2, &w1, &w0, z[56]);
   ws[24] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[26], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[27], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[27]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[26]);
   word3_add(&w2, &w1, &w0, z[57]);
   ws[25] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[27], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[28], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[28]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[27]);
   word3_add(&w2, &w1, &w0, z[58]);
   ws[26] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[28], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[29], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[29]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[28]);
   word3_add(&w2, &w1, &w0, z[59]);
   ws[27] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[29], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[30], p[30]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[29]);
   word3_add(&w2, &w1, &w0, z[60]);
   ws[28] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[30], p[31]);
   word3_muladd(&w2, &w1, &w0, ws[31], p[30]);
   word3_add(&w2, &w1, &w0, z[61]);
   ws[29] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_muladd(&w2, &w1, &w0, ws[31], p[31]);
   word3_add(&w2, &w1, &w0, z[62]);
   ws[30] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[63]);
   ws[31] = w0;
   w0 = w1; w1 = w2; w2 = 0;
   word3_add(&w2, &w1, &w0, z[65]);
   ws[32] = w0;
   ws[33] = w1;
   word borrow = bigint_sub3(ws + 32 + 1, ws, 32 + 1, p, 32);
   CT::conditional_copy_mem(borrow, z, ws, ws + 33, 33);
   clear_mem(z + 32, 2*(32+1) - 32);
   }

}
/*
* DSA Parameter Generation
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

/*
* Check if this size is allowed by FIPS 186-3
*/
bool fips186_3_valid_size(size_t pbits, size_t qbits)
   {
   if(qbits == 160)
      return (pbits == 1024);

   if(qbits == 224)
      return (pbits == 2048);

   if(qbits == 256)
      return (pbits == 2048 || pbits == 3072);

   return false;
   }

}

/*
* Attempt DSA prime generation with given seed
*/
bool generate_dsa_primes(RandomNumberGenerator& rng,
                         BigInt& p, BigInt& q,
                         size_t pbits, size_t qbits,
                         const std::vector<uint8_t>& seed_c,
                         size_t offset)
   {
   if(!fips186_3_valid_size(pbits, qbits))
      throw Invalid_Argument(
         "FIPS 186-3 does not allow DSA domain parameters of " +
         std::to_string(pbits) + "/" + std::to_string(qbits) + " bits long");

   if(seed_c.size() * 8 < qbits)
      throw Invalid_Argument(
         "Generating a DSA parameter set with a " + std::to_string(qbits) +
         " bit long q requires a seed at least as many bits long");

   const std::string hash_name = "SHA-" + std::to_string(qbits);
   std::unique_ptr<HashFunction> hash(HashFunction::create_or_throw(hash_name));

   const size_t HASH_SIZE = hash->output_length();

   class Seed final
      {
      public:
         explicit Seed(const std::vector<uint8_t>& s) : m_seed(s) {}

         const std::vector<uint8_t>& value() const { return m_seed; }

         Seed& operator++()
            {
            for(size_t j = m_seed.size(); j > 0; --j)
               if(++m_seed[j-1])
                  break;
            return (*this);
            }
      private:
         std::vector<uint8_t> m_seed;
      };

   Seed seed(seed_c);

   q.binary_decode(hash->process(seed.value()));
   q.set_bit(qbits-1);
   q.set_bit(0);

   if(!is_prime(q, rng, 128, true))
      return false;

   const size_t n = (pbits-1) / (HASH_SIZE * 8),
                b = (pbits-1) % (HASH_SIZE * 8);

   BigInt X;
   std::vector<uint8_t> V(HASH_SIZE * (n+1));

   Modular_Reducer mod_2q(2*q);

   for(size_t j = 0; j != 4*pbits; ++j)
      {
      for(size_t k = 0; k <= n; ++k)
         {
         ++seed;
         hash->update(seed.value());
         hash->final(&V[HASH_SIZE * (n-k)]);
         }

      if(j >= offset)
         {
         X.binary_decode(&V[HASH_SIZE - 1 - b/8],
                         V.size() - (HASH_SIZE - 1 - b/8));
         X.set_bit(pbits-1);

         p = X - (mod_2q.reduce(X) - 1);

         if(p.bits() == pbits && is_prime(p, rng, 128, true))
            return true;
         }
      }
   return false;
   }

/*
* Generate DSA Primes
*/
std::vector<uint8_t> generate_dsa_primes(RandomNumberGenerator& rng,
                                      BigInt& p, BigInt& q,
                                      size_t pbits, size_t qbits)
   {
   while(true)
      {
      std::vector<uint8_t> seed(qbits / 8);
      rng.randomize(seed.data(), seed.size());

      if(generate_dsa_primes(rng, p, q, pbits, qbits, seed))
         return seed;
      }
   }

}
/*
* Jacobi Function
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/*
* Calculate the Jacobi symbol
*/
int32_t jacobi(const BigInt& a, const BigInt& n)
   {
   if(n.is_even() || n < 2)
      throw Invalid_Argument("jacobi: second argument must be odd and > 1");

   BigInt x = a % n;
   BigInt y = n;
   int32_t J = 1;

   while(y > 1)
      {
      x %= y;
      if(x > y / 2)
         {
         x = y - x;
         if(y % 4 == 3)
            J = -J;
         }
      if(x.is_zero())
         return 0;

      size_t shifts = low_zero_bits(x);
      x >>= shifts;
      if(shifts % 2)
         {
         word y_mod_8 = y % 8;
         if(y_mod_8 == 3 || y_mod_8 == 5)
            J = -J;
         }

      if(x % 4 == 3 && y % 4 == 3)
         J = -J;
      std::swap(x, y);
      }
   return J;
   }

}
/*
* Prime Generation
* (C) 1999-2007,2018,2019 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

class Prime_Sieve final
   {
   public:
      Prime_Sieve(const BigInt& init_value, size_t sieve_size) :
         m_sieve(std::min(sieve_size, PRIME_TABLE_SIZE))
         {
         for(size_t i = 0; i != m_sieve.size(); ++i)
            m_sieve[i] = static_cast<uint16_t>(init_value % PRIMES[i]);
         }

      void step(word increment)
         {
         for(size_t i = 0; i != m_sieve.size(); ++i)
            {
            m_sieve[i] = (m_sieve[i] + increment) % PRIMES[i];
            }
         }

      bool passes(bool check_2p1 = false) const
         {
         for(size_t i = 0; i != m_sieve.size(); ++i)
            {
            /*
            In this case, p is a multiple of PRIMES[i]
            */
            if(m_sieve[i] == 0)
               return false;

            if(check_2p1)
               {
               /*
               In this case, 2*p+1 will be a multiple of PRIMES[i]

               So if potentially generating a safe prime, we want to
               avoid this value because 2*p+1 will certainly not be prime.

               See "Safe Prime Generation with a Combined Sieve" M. Wiener
               https://eprint.iacr.org/2003/186.pdf
               */
               if(m_sieve[i] == (PRIMES[i] - 1) / 2)
                  return false;
               }
            }

         return true;
         }

   private:
      std::vector<uint16_t> m_sieve;
   };

}


/*
* Generate a random prime
*/
BigInt random_prime(RandomNumberGenerator& rng,
                    size_t bits, const BigInt& coprime,
                    size_t equiv, size_t modulo,
                    size_t prob)
   {
   if(bits <= 1)
      {
      throw Invalid_Argument("random_prime: Can't make a prime of " +
                             std::to_string(bits) + " bits");
      }
   if(coprime.is_negative() || (!coprime.is_zero() && coprime.is_even()) || coprime.bits() >= bits)
      {
      throw Invalid_Argument("random_prime: invalid coprime");
      }
   if(modulo == 0)
      {
      throw Invalid_Argument("random_prime: Invalid modulo value");
      }

   equiv %= modulo;

   if(equiv == 0)
      throw Invalid_Argument("random_prime Invalid value for equiv/modulo");

   // Handle small values:

   if(bits <= 16)
      {
      if(equiv != 1 || modulo != 2 || coprime != 0)
         throw Not_Implemented("random_prime equiv/modulo/coprime options not usable for small primes");

      if(bits == 2)
         {
         return ((rng.next_byte() % 2) ? 2 : 3);
         }
      else if(bits == 3)
         {
         return ((rng.next_byte() % 2) ? 5 : 7);
         }
      else if(bits == 4)
         {
         return ((rng.next_byte() % 2) ? 11 : 13);
         }
      else
         {
         for(;;)
            {
            // This is slightly biased, but for small primes it does not seem to matter
            uint8_t b[4];
            rng.randomize(b, 4);
            const size_t idx = load_le<uint32_t>(b, 0) % PRIME_TABLE_SIZE;
            const uint16_t small_prime = PRIMES[idx];

            if(high_bit(small_prime) == bits)
               return small_prime;
            }
         }
      }

   const size_t MAX_ATTEMPTS = 32*1024;

   const size_t mr_trials = miller_rabin_test_iterations(bits, prob, true);

   while(true)
      {
      BigInt p(rng, bits);

      // Force lowest and two top bits on
      p.set_bit(bits - 1);
      p.set_bit(bits - 2);
      p.set_bit(0);

      // Force p to be equal to equiv mod modulo
      p += (modulo - (p % modulo)) + equiv;

      Prime_Sieve sieve(p, bits);

      for(size_t attempt = 0; attempt <= MAX_ATTEMPTS; ++attempt)
         {
         p += modulo;

         sieve.step(modulo);

         // p can be even if modulo is odd, continue on in that case
         if(p.is_even() || sieve.passes(true) == false)
            continue;

         Modular_Reducer mod_p(p);

         if(coprime > 1)
            {
            /*
            First do a single M-R iteration to quickly elimate most non-primes,
            before doing the coprimality check which is expensive
            */
            if(is_miller_rabin_probable_prime(p, mod_p, rng, 1) == false)
               continue;

            /*
            * Check if p - 1 and coprime are relatively prime, using gcd.
            * The gcd computation is const-time
            */
            if(gcd(p - 1, coprime) > 1)
               continue;
            }

         if(p.bits() > bits)
            break;

         if(is_miller_rabin_probable_prime(p, mod_p, rng, mr_trials) == false)
            continue;

         if(prob > 32 && !is_lucas_probable_prime(p, mod_p))
            continue;

         return p;
         }
      }
   }

BigInt generate_rsa_prime(RandomNumberGenerator& keygen_rng,
                          RandomNumberGenerator& prime_test_rng,
                          size_t bits,
                          const BigInt& coprime,
                          size_t prob)
   {
   if(bits < 512)
      throw Invalid_Argument("generate_rsa_prime bits too small");

   /*
   * The restriction on coprime <= 64 bits is arbitrary but generally speaking
   * very large RSA public exponents are a bad idea both for performance and due
   * to attacks on small d.
   */
   if(coprime <= 1 || coprime.is_even() || coprime.bits() > 64)
      throw Invalid_Argument("generate_rsa_prime coprime must be small odd positive integer");

   const size_t MAX_ATTEMPTS = 32*1024;

   const size_t mr_trials = miller_rabin_test_iterations(bits, prob, true);

   while(true)
      {
      BigInt p(keygen_rng, bits);

      // Force high two bits so multiplication always results in expected n bit integer
      p.set_bit(bits - 1);
      p.set_bit(bits - 2);
      p.set_bit(0);

      const word step = 2;

      Prime_Sieve sieve(p, bits);

      for(size_t attempt = 0; attempt <= MAX_ATTEMPTS; ++attempt)
         {
         p += step;

         sieve.step(step);

         if(sieve.passes() == false)
            continue;

         Modular_Reducer mod_p(p);

         /*
         * Do a single primality test first before checking coprimality, since
         * currently a single Miller-Rabin test is faster than computing gcd,
         * and this eliminates almost all wasted gcd computations.
         */
         if(is_miller_rabin_probable_prime(p, mod_p, prime_test_rng, 1) == false)
            continue;

         /*
         * Check if p - 1 and coprime are relatively prime.
         */
         if(gcd(p - 1, coprime) > 1)
            continue;

         if(p.bits() > bits)
            break;

         if(is_miller_rabin_probable_prime(p, mod_p, prime_test_rng, mr_trials) == true)
            return p;
         }
      }
   }

/*
* Generate a random safe prime
*/
BigInt random_safe_prime(RandomNumberGenerator& rng, size_t bits)
   {
   if(bits <= 64)
      throw Invalid_Argument("random_safe_prime: Can't make a prime of " +
                             std::to_string(bits) + " bits");

   const size_t error_bound = 128;

   BigInt q, p;
   for(;;)
      {
      /*
      Generate q == 2 (mod 3), since otherwise [in the case of q == 1 (mod 3)],
      2*q+1 == 3 (mod 3) and so certainly not prime.
      */
      q = random_prime(rng, bits - 1, 0, 2, 3, error_bound);
      p = (q << 1) + 1;

      if(is_prime(p, rng, error_bound, true))
         {
         return p;
         }
      }
   }

}
/*
* (C) 1999-2011,2016,2018,2019,2020 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/*
Sets result to a^-1 * 2^k mod a
with n <= k <= 2n
Returns k

"The Montgomery Modular Inverse - Revisited" Çetin Koç, E. Savas
https://citeseerx.ist.psu.edu/viewdoc/citations?doi=10.1.1.75.8377

A const time implementation of this algorithm is described in
"Constant Time Modular Inversion" Joppe W. Bos
http://www.joppebos.com/files/CTInversion.pdf
*/
size_t almost_montgomery_inverse(BigInt& result,
                                 const BigInt& a,
                                 const BigInt& p)
   {
   size_t k = 0;

   BigInt u = p, v = a, r = 0, s = 1;

   while(v > 0)
      {
      if(u.is_even())
         {
         u >>= 1;
         s <<= 1;
         }
      else if(v.is_even())
         {
         v >>= 1;
         r <<= 1;
         }
      else if(u > v)
         {
         u -= v;
         u >>= 1;
         r += s;
         s <<= 1;
         }
      else
         {
         v -= u;
         v >>= 1;
         s += r;
         r <<= 1;
         }

      ++k;
      }

   if(r >= p)
      {
      r -= p;
      }

   result = p - r;

   return k;
   }

BigInt normalized_montgomery_inverse(const BigInt& a, const BigInt& p)
   {
   BigInt r;
   size_t k = almost_montgomery_inverse(r, a, p);

   for(size_t i = 0; i != k; ++i)
      {
      if(r.is_odd())
         r += p;
      r >>= 1;
      }

   return r;
   }

namespace {

BigInt inverse_mod_odd_modulus(const BigInt& n, const BigInt& mod)
   {
   // Caller should assure these preconditions:
   BOTAN_DEBUG_ASSERT(n.is_positive());
   BOTAN_DEBUG_ASSERT(mod.is_positive());
   BOTAN_DEBUG_ASSERT(n < mod);
   BOTAN_DEBUG_ASSERT(mod >= 3 && mod.is_odd());

   /*
   This uses a modular inversion algorithm designed by Niels Möller
   and implemented in Nettle. The same algorithm was later also
   adapted to GMP in mpn_sec_invert.

   It can be easily implemented in a way that does not depend on
   secret branches or memory lookups, providing resistance against
   some forms of side channel attack.

   There is also a description of the algorithm in Appendix 5 of "Fast
   Software Polynomial Multiplication on ARM Processors using the NEON Engine"
   by Danilo Câmara, Conrado P. L. Gouvêa, Julio López, and Ricardo
   Dahab in LNCS 8182
      https://conradoplg.cryptoland.net/files/2010/12/mocrysen13.pdf

   Thanks to Niels for creating the algorithm, explaining some things
   about it, and the reference to the paper.
   */

   const size_t mod_words = mod.sig_words();
   BOTAN_ASSERT(mod_words > 0, "Not empty");

   secure_vector<word> tmp_mem(5*mod_words);

   word* v_w = &tmp_mem[0];
   word* u_w = &tmp_mem[1*mod_words];
   word* b_w = &tmp_mem[2*mod_words];
   word* a_w = &tmp_mem[3*mod_words];
   word* mp1o2 = &tmp_mem[4*mod_words];

   CT::poison(tmp_mem.data(), tmp_mem.size());

   copy_mem(a_w, n.data(), std::min(n.size(), mod_words));
   copy_mem(b_w, mod.data(), std::min(mod.size(), mod_words));
   u_w[0] = 1;
   // v_w = 0

   // compute (mod + 1) / 2 which [because mod is odd] is equal to
   // (mod / 2) + 1
   copy_mem(mp1o2, mod.data(), std::min(mod.size(), mod_words));
   bigint_shr1(mp1o2, mod_words, 0, 1);
   word carry = bigint_add2_nc(mp1o2, mod_words, u_w, 1);
   BOTAN_ASSERT_NOMSG(carry == 0);

   // Only n.bits() + mod.bits() iterations are required, but avoid leaking the size of n
   const size_t execs = 2 * mod.bits();

   for(size_t i = 0; i != execs; ++i)
      {
      const word odd_a = a_w[0] & 1;

      //if(odd_a) a -= b
      word underflow = bigint_cnd_sub(odd_a, a_w, b_w, mod_words);

      //if(underflow) { b -= a; a = abs(a); swap(u, v); }
      bigint_cnd_add(underflow, b_w, a_w, mod_words);
      bigint_cnd_abs(underflow, a_w, mod_words);
      bigint_cnd_swap(underflow, u_w, v_w, mod_words);

      // a >>= 1
      bigint_shr1(a_w, mod_words, 0, 1);

      //if(odd_a) u -= v;
      word borrow = bigint_cnd_sub(odd_a, u_w, v_w, mod_words);

      // if(borrow) u += p
      bigint_cnd_add(borrow, u_w, mod.data(), mod_words);

      const word odd_u = u_w[0] & 1;

      // u >>= 1
      bigint_shr1(u_w, mod_words, 0, 1);

      //if(odd_u) u += mp1o2;
      bigint_cnd_add(odd_u, u_w, mp1o2, mod_words);
      }

   auto a_is_0 = CT::Mask<word>::set();
   for(size_t i = 0; i != mod_words; ++i)
      a_is_0 &= CT::Mask<word>::is_zero(a_w[i]);

   auto b_is_1 = CT::Mask<word>::is_equal(b_w[0], 1);
   for(size_t i = 1; i != mod_words; ++i)
      b_is_1 &= CT::Mask<word>::is_zero(b_w[i]);

   BOTAN_ASSERT(a_is_0.is_set(), "A is zero");

   // if b != 1 then gcd(n,mod) > 1 and inverse does not exist
   // in which case zero out the result to indicate this
   (~b_is_1).if_set_zero_out(v_w, mod_words);

   /*
   * We've placed the result in the lowest words of the temp buffer.
   * So just clear out the other values and then give that buffer to a
   * BigInt.
   */
   clear_mem(&tmp_mem[mod_words], 4*mod_words);

   CT::unpoison(tmp_mem.data(), tmp_mem.size());

   BigInt r;
   r.swap_reg(tmp_mem);
   return r;
   }

BigInt inverse_mod_pow2(const BigInt& a1, size_t k)
   {
   /*
   * From "A New Algorithm for Inversion mod p^k" by Çetin Kaya Koç
   * https://eprint.iacr.org/2017/411.pdf sections 5 and 7.
   */

   if(a1.is_even())
      return 0;

   BigInt a = a1;
   a.mask_bits(k);

   BigInt b = 1;
   BigInt X = 0;
   BigInt newb;

   const size_t a_words = a.sig_words();

   X.grow_to(round_up(k, BOTAN_MP_WORD_BITS) / BOTAN_MP_WORD_BITS);
   b.grow_to(a_words);

   /*
   Hide the exact value of k. k is anyway known to word length
   granularity because of the length of a, so no point in doing more
   than this.
   */
   const size_t iter = round_up(k, BOTAN_MP_WORD_BITS);

   for(size_t i = 0; i != iter; ++i)
      {
      const bool b0 = b.get_bit(0);
      X.conditionally_set_bit(i, b0);
      newb = b - a;
      b.ct_cond_assign(b0, newb);
      b >>= 1;
      }

   X.mask_bits(k);
   X.const_time_unpoison();
   return X;
   }

}

BigInt inverse_mod(const BigInt& n, const BigInt& mod)
   {
   if(mod.is_zero())
      throw BigInt::DivideByZero();
   if(mod.is_negative() || n.is_negative())
      throw Invalid_Argument("inverse_mod: arguments must be non-negative");
   if(n.is_zero() || (n.is_even() && mod.is_even()))
      return 0;

   if(mod.is_odd())
      {
      /*
      Fastpath for common case. This leaks information if n > mod
      but we don't guarantee const time behavior in that case.
      */
      if(n < mod)
         return inverse_mod_odd_modulus(n, mod);
      else
         return inverse_mod_odd_modulus(ct_modulo(n, mod), mod);
      }

   const size_t mod_lz = low_zero_bits(mod);
   BOTAN_ASSERT_NOMSG(mod_lz > 0);
   const size_t mod_bits = mod.bits();
   BOTAN_ASSERT_NOMSG(mod_bits > mod_lz);

   if(mod_lz == mod_bits - 1)
      {
      // In this case we are performing an inversion modulo 2^k
      return inverse_mod_pow2(n, mod_lz);
      }

   /*
   * In this case we are performing an inversion modulo 2^k*o for
   * some k > 1 and some odd (not necessarily prime) integer.
   * Compute the inversions modulo 2^k and modulo o, then combine them
   * using CRT, which is possible because 2^k and o are relatively prime.
   */

   const BigInt o = mod >> mod_lz;
   const BigInt n_redc = ct_modulo(n, o);
   const BigInt inv_o = inverse_mod_odd_modulus(n_redc, o);
   const BigInt inv_2k = inverse_mod_pow2(n, mod_lz);

   // No modular inverse in this case:
   if(inv_o == 0 || inv_2k == 0)
      return 0;

   const BigInt m2k = BigInt::power_of_2(mod_lz);
   // Compute the CRT parameter
   const BigInt c = inverse_mod_pow2(o, mod_lz);

   // Compute h = c*(inv_2k-inv_o) mod 2^k
   BigInt h = c * (inv_2k - inv_o);
   const bool h_neg = h.is_negative();
   h.set_sign(BigInt::Positive);
   h.mask_bits(mod_lz);
   const bool h_nonzero = h.is_nonzero();
   h.ct_cond_assign(h_nonzero && h_neg, m2k - h);

   // Return result inv_o + h * o
   h *= o;
   h += inv_o;
   return h;
   }

// Deprecated forwarding functions:
BigInt inverse_euclid(const BigInt& x, const BigInt& modulus)
   {
   return inverse_mod(x, modulus);
   }

BigInt ct_inverse_mod_odd_modulus(const BigInt& n, const BigInt& mod)
   {
   return inverse_mod_odd_modulus(n, mod);
   }

word monty_inverse(word a)
   {
   if(a % 2 == 0)
      throw Invalid_Argument("monty_inverse only valid for odd integers");

   /*
   * From "A New Algorithm for Inversion mod p^k" by Çetin Kaya Koç
   * https://eprint.iacr.org/2017/411.pdf sections 5 and 7.
   */

   word b = 1;
   word r = 0;

   for(size_t i = 0; i != BOTAN_MP_WORD_BITS; ++i)
      {
      const word bi = b % 2;
      r >>= 1;
      r += bi << (BOTAN_MP_WORD_BITS - 1);

      b -= a * bi;
      b >>= 1;
      }

   // Now invert in addition space
   r = (MP_WORD_MAX - r) + 1;

   return r;
   }

}
/*
* (C) 2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

Montgomery_Params::Montgomery_Params(const BigInt& p,
                                     const Modular_Reducer& mod_p)
   {
   if(p.is_even() || p < 3)
      throw Invalid_Argument("Montgomery_Params invalid modulus");

   m_p = p;
   m_p_words = m_p.sig_words();
   m_p_dash = monty_inverse(m_p.word_at(0));

   const BigInt r = BigInt::power_of_2(m_p_words * BOTAN_MP_WORD_BITS);

   m_r1 = mod_p.reduce(r);
   m_r2 = mod_p.square(m_r1);
   m_r3 = mod_p.multiply(m_r1, m_r2);
   }

Montgomery_Params::Montgomery_Params(const BigInt& p)
   {

   if(p.is_negative() || p.is_even())
      throw Invalid_Argument("Montgomery_Params invalid modulus");

   m_p = p;
   m_p_words = m_p.sig_words();
   m_p_dash = monty_inverse(m_p.word_at(0));

   const BigInt r = BigInt::power_of_2(m_p_words * BOTAN_MP_WORD_BITS);

   // It might be faster to use ct_modulo here vs setting up Barrett reduction?
   Modular_Reducer mod_p(p);

   m_r1 = mod_p.reduce(r);
   m_r2 = mod_p.square(m_r1);
   m_r3 = mod_p.multiply(m_r1, m_r2);
   }

BigInt Montgomery_Params::inv_mod_p(const BigInt& x) const
   {
   // TODO use Montgomery inverse here?
   return inverse_mod(x, p());
   }

BigInt Montgomery_Params::redc(const BigInt& x, secure_vector<word>& ws) const
   {
   const size_t output_size = 2*m_p_words + 2;

   if(ws.size() < output_size)
      ws.resize(output_size);

   BigInt z = x;
   z.grow_to(output_size);

   bigint_monty_redc(z.mutable_data(),
                     m_p.data(), m_p_words, m_p_dash,
                     ws.data(), ws.size());

   return z;
   }

BigInt Montgomery_Params::mul(const BigInt& x, const BigInt& y,
                              secure_vector<word>& ws) const
   {
   const size_t output_size = 2*m_p_words + 2;

   if(ws.size() < output_size)
      ws.resize(output_size);

   BOTAN_DEBUG_ASSERT(x.sig_words() <= m_p_words);
   BOTAN_DEBUG_ASSERT(y.sig_words() <= m_p_words);

   BigInt z(BigInt::Positive, output_size);
   bigint_mul(z.mutable_data(), z.size(),
              x.data(), x.size(), std::min(m_p_words, x.size()),
              y.data(), y.size(), std::min(m_p_words, y.size()),
              ws.data(), ws.size());

   bigint_monty_redc(z.mutable_data(),
                     m_p.data(), m_p_words, m_p_dash,
                     ws.data(), ws.size());

   return z;
   }

BigInt Montgomery_Params::mul(const BigInt& x,
                              const secure_vector<word>& y,
                              secure_vector<word>& ws) const
   {
   const size_t output_size = 2*m_p_words + 2;
   if(ws.size() < output_size)
      ws.resize(output_size);
   BigInt z(BigInt::Positive, output_size);

   BOTAN_DEBUG_ASSERT(x.sig_words() <= m_p_words);

   bigint_mul(z.mutable_data(), z.size(),
              x.data(), x.size(), std::min(m_p_words, x.size()),
              y.data(), y.size(), std::min(m_p_words, y.size()),
              ws.data(), ws.size());

   bigint_monty_redc(z.mutable_data(),
                     m_p.data(), m_p_words, m_p_dash,
                     ws.data(), ws.size());

   return z;
   }

void Montgomery_Params::mul_by(BigInt& x,
                               const secure_vector<word>& y,
                               secure_vector<word>& ws) const
   {
   const size_t output_size = 2*m_p_words + 2;

   if(ws.size() < 2*output_size)
      ws.resize(2*output_size);

   word* z_data = &ws[0];
   word* ws_data = &ws[output_size];

   BOTAN_DEBUG_ASSERT(x.sig_words() <= m_p_words);

   bigint_mul(z_data, output_size,
              x.data(), x.size(), std::min(m_p_words, x.size()),
              y.data(), y.size(), std::min(m_p_words, y.size()),
              ws_data, output_size);

   bigint_monty_redc(z_data,
                     m_p.data(), m_p_words, m_p_dash,
                     ws_data, output_size);

   if(x.size() < output_size)
      x.grow_to(output_size);
   copy_mem(x.mutable_data(), z_data, output_size);
   }

void Montgomery_Params::mul_by(BigInt& x,
                               const BigInt& y,
                               secure_vector<word>& ws) const
   {
   const size_t output_size = 2*m_p_words + 2;

   if(ws.size() < 2*output_size)
      ws.resize(2*output_size);

   word* z_data = &ws[0];
   word* ws_data = &ws[output_size];

   BOTAN_DEBUG_ASSERT(x.sig_words() <= m_p_words);

   bigint_mul(z_data, output_size,
              x.data(), x.size(), std::min(m_p_words, x.size()),
              y.data(), y.size(), std::min(m_p_words, y.size()),
              ws_data, output_size);

   bigint_monty_redc(z_data,
                     m_p.data(), m_p_words, m_p_dash,
                     ws_data, output_size);

   if(x.size() < output_size)
      x.grow_to(output_size);
   copy_mem(x.mutable_data(), z_data, output_size);
   }

BigInt Montgomery_Params::sqr(const BigInt& x, secure_vector<word>& ws) const
   {
   const size_t output_size = 2*m_p_words + 2;

   if(ws.size() < output_size)
      ws.resize(output_size);

   BigInt z(BigInt::Positive, output_size);

   BOTAN_DEBUG_ASSERT(x.sig_words() <= m_p_words);

   bigint_sqr(z.mutable_data(), z.size(),
              x.data(), x.size(), std::min(m_p_words, x.size()),
              ws.data(), ws.size());

   bigint_monty_redc(z.mutable_data(),
                     m_p.data(), m_p_words, m_p_dash,
                     ws.data(), ws.size());

   return z;
   }

void Montgomery_Params::square_this(BigInt& x,
                                    secure_vector<word>& ws) const
   {
   const size_t output_size = 2*m_p_words + 2;

   if(ws.size() < 2*output_size)
      ws.resize(2*output_size);

   word* z_data = &ws[0];
   word* ws_data = &ws[output_size];

   BOTAN_DEBUG_ASSERT(x.sig_words() <= m_p_words);

   bigint_sqr(z_data, output_size,
              x.data(), x.size(), std::min(m_p_words, x.size()),
              ws_data, output_size);

   bigint_monty_redc(z_data,
                     m_p.data(), m_p_words, m_p_dash,
                     ws_data, output_size);

   if(x.size() < output_size)
      x.grow_to(output_size);
   copy_mem(x.mutable_data(), z_data, output_size);
   }

Montgomery_Int::Montgomery_Int(const std::shared_ptr<const Montgomery_Params> params,
                               const BigInt& v,
                               bool redc_needed) :
   m_params(params)
   {
   if(redc_needed == false)
      {
      m_v = v;
      }
   else
      {
      BOTAN_ASSERT_NOMSG(m_v < m_params->p());
      secure_vector<word> ws;
      m_v = m_params->mul(v, m_params->R2(), ws);
      }
   }

Montgomery_Int::Montgomery_Int(std::shared_ptr<const Montgomery_Params> params,
                               const uint8_t bits[], size_t len,
                               bool redc_needed) :
   m_params(params),
   m_v(bits, len)
   {
   if(redc_needed)
      {
      BOTAN_ASSERT_NOMSG(m_v < m_params->p());
      secure_vector<word> ws;
      m_v = m_params->mul(m_v, m_params->R2(), ws);
      }
   }

Montgomery_Int::Montgomery_Int(std::shared_ptr<const Montgomery_Params> params,
                               const word words[], size_t len,
                               bool redc_needed) :
   m_params(params),
   m_v(words, len)
   {
   if(redc_needed)
      {
      BOTAN_ASSERT_NOMSG(m_v < m_params->p());
      secure_vector<word> ws;
      m_v = m_params->mul(m_v, m_params->R2(), ws);
      }
   }

void Montgomery_Int::fix_size()
   {
   const size_t p_words = m_params->p_words();

   if(m_v.sig_words() > p_words)
      throw Internal_Error("Montgomery_Int::fix_size v too large");

   m_v.grow_to(p_words);
   }

bool Montgomery_Int::operator==(const Montgomery_Int& other) const
   {
   return m_v == other.m_v && m_params->p() == other.m_params->p();
   }

std::vector<uint8_t> Montgomery_Int::serialize() const
   {
   std::vector<uint8_t> v(size());
   BigInt::encode_1363(v.data(), v.size(), value());
   return v;
   }

size_t Montgomery_Int::size() const
   {
   return m_params->p().bytes();
   }

bool Montgomery_Int::is_one() const
   {
   return m_v == m_params->R1();
   }

bool Montgomery_Int::is_zero() const
   {
   return m_v.is_zero();
   }

BigInt Montgomery_Int::value() const
   {
   secure_vector<word> ws;
   return m_params->redc(m_v, ws);
   }

Montgomery_Int Montgomery_Int::operator+(const Montgomery_Int& other) const
   {
   secure_vector<word> ws;
   BigInt z = m_v;
   z.mod_add(other.m_v, m_params->p(), ws);
   return Montgomery_Int(m_params, z, false);
   }

Montgomery_Int Montgomery_Int::operator-(const Montgomery_Int& other) const
   {
   secure_vector<word> ws;
   BigInt z = m_v;
   z.mod_sub(other.m_v, m_params->p(), ws);
   return Montgomery_Int(m_params, z, false);
   }

Montgomery_Int& Montgomery_Int::operator+=(const Montgomery_Int& other)
   {
   secure_vector<word> ws;
   return this->add(other, ws);
   }

Montgomery_Int& Montgomery_Int::add(const Montgomery_Int& other, secure_vector<word>& ws)
   {
   m_v.mod_add(other.m_v, m_params->p(), ws);
   return (*this);
   }

Montgomery_Int& Montgomery_Int::operator-=(const Montgomery_Int& other)
   {
   secure_vector<word> ws;
   return this->sub(other, ws);
   }

Montgomery_Int& Montgomery_Int::sub(const Montgomery_Int& other, secure_vector<word>& ws)
   {
   m_v.mod_sub(other.m_v, m_params->p(), ws);
   return (*this);
   }

Montgomery_Int Montgomery_Int::operator*(const Montgomery_Int& other) const
   {
   secure_vector<word> ws;
   return Montgomery_Int(m_params, m_params->mul(m_v, other.m_v, ws), false);
   }

Montgomery_Int Montgomery_Int::mul(const Montgomery_Int& other,
                                   secure_vector<word>& ws) const
   {
   return Montgomery_Int(m_params, m_params->mul(m_v, other.m_v, ws), false);
   }

Montgomery_Int& Montgomery_Int::mul_by(const Montgomery_Int& other,
                                       secure_vector<word>& ws)
   {
   m_params->mul_by(m_v, other.m_v, ws);
   return (*this);
   }

Montgomery_Int& Montgomery_Int::mul_by(const secure_vector<word>& other,
                                       secure_vector<word>& ws)
   {
   m_params->mul_by(m_v, other, ws);
   return (*this);
   }

Montgomery_Int& Montgomery_Int::operator*=(const Montgomery_Int& other)
   {
   secure_vector<word> ws;
   return mul_by(other, ws);
   }

Montgomery_Int& Montgomery_Int::operator*=(const secure_vector<word>& other)
   {
   secure_vector<word> ws;
   return mul_by(other, ws);
   }

Montgomery_Int& Montgomery_Int::square_this_n_times(secure_vector<word>& ws, size_t n)
   {
   for(size_t i = 0; i != n; ++i)
      m_params->square_this(m_v, ws);
   return (*this);
   }

Montgomery_Int& Montgomery_Int::square_this(secure_vector<word>& ws)
   {
   m_params->square_this(m_v, ws);
   return (*this);
   }

Montgomery_Int Montgomery_Int::square(secure_vector<word>& ws) const
   {
   return Montgomery_Int(m_params, m_params->sqr(m_v, ws), false);
   }

Montgomery_Int Montgomery_Int::multiplicative_inverse() const
   {
   secure_vector<word> ws;
   const BigInt iv = m_params->mul(m_params->inv_mod_p(m_v), m_params->R3(), ws);
   return Montgomery_Int(m_params, iv, false);
   }

Montgomery_Int Montgomery_Int::additive_inverse() const
   {
   return Montgomery_Int(m_params, m_params->p()) - (*this);
   }

Montgomery_Int& Montgomery_Int::mul_by_2(secure_vector<word>& ws)
   {
   m_v.mod_mul(2, m_params->p(), ws);
   return (*this);
   }

Montgomery_Int& Montgomery_Int::mul_by_3(secure_vector<word>& ws)
   {
   m_v.mod_mul(3, m_params->p(), ws);
   return (*this);
   }

Montgomery_Int& Montgomery_Int::mul_by_4(secure_vector<word>& ws)
   {
   m_v.mod_mul(4, m_params->p(), ws);
   return (*this);
   }

Montgomery_Int& Montgomery_Int::mul_by_8(secure_vector<word>& ws)
   {
   m_v.mod_mul(8, m_params->p(), ws);
   return (*this);
   }

}
/*
* Montgomery Exponentiation
* (C) 1999-2010,2012,2018 Jack Lloyd
*     2016 Matthias Gierlings
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

class Montgomery_Exponentation_State
   {
   public:
      Montgomery_Exponentation_State(std::shared_ptr<const Montgomery_Params> params,
                                     const BigInt& g,
                                     size_t window_bits,
                                     bool const_time);

      BigInt exponentiation(const BigInt& k, size_t max_k_bits) const;

      BigInt exponentiation_vartime(const BigInt& k) const;
   private:
      std::shared_ptr<const Montgomery_Params> m_params;
      std::vector<Montgomery_Int> m_g;
      size_t m_window_bits;
      bool m_const_time;
   };

Montgomery_Exponentation_State::Montgomery_Exponentation_State(std::shared_ptr<const Montgomery_Params> params,
                                                               const BigInt& g,
                                                               size_t window_bits,
                                                               bool const_time) :
   m_params(params),
   m_window_bits(window_bits == 0 ? 4 : window_bits),
   m_const_time(const_time)
   {
   BOTAN_ARG_CHECK(g < m_params->p(), "Montgomery base too big");

   if(m_window_bits < 1 || m_window_bits > 12) // really even 8 is too large ...
      throw Invalid_Argument("Invalid window bits for Montgomery exponentiation");

   const size_t window_size = (static_cast<size_t>(1) << m_window_bits);

   m_g.reserve(window_size);

   m_g.push_back(Montgomery_Int(m_params, m_params->R1(), false));

   m_g.push_back(Montgomery_Int(m_params, g));

   for(size_t i = 2; i != window_size; ++i)
      {
      m_g.push_back(m_g[1] * m_g[i - 1]);
      }

   // Resize each element to exactly p words
   for(size_t i = 0; i != window_size; ++i)
      {
      m_g[i].fix_size();
      if(const_time)
         m_g[i].const_time_poison();
      }
   }

namespace {

void const_time_lookup(secure_vector<word>& output,
                       const std::vector<Montgomery_Int>& g,
                       size_t nibble)
   {
   BOTAN_ASSERT_NOMSG(g.size() % 2 == 0); // actually a power of 2

   const size_t words = output.size();

   clear_mem(output.data(), output.size());

   for(size_t i = 0; i != g.size(); i += 2)
      {
      const secure_vector<word>& vec_0 = g[i  ].repr().get_word_vector();
      const secure_vector<word>& vec_1 = g[i+1].repr().get_word_vector();

      BOTAN_ASSERT_NOMSG(vec_0.size() >= words && vec_1.size() >= words);

      const auto mask_0 = CT::Mask<word>::is_equal(nibble, i);
      const auto mask_1 = CT::Mask<word>::is_equal(nibble, i+1);

      for(size_t w = 0; w != words; ++w)
         {
         output[w] |= mask_0.if_set_return(vec_0[w]);
         output[w] |= mask_1.if_set_return(vec_1[w]);
         }
      }
   }

}

BigInt Montgomery_Exponentation_State::exponentiation(const BigInt& scalar, size_t max_k_bits) const
   {
   BOTAN_DEBUG_ASSERT(scalar.bits() <= max_k_bits);
   // TODO add a const-time implementation of above assert and use it in release builds

   const size_t exp_nibbles = (max_k_bits + m_window_bits - 1) / m_window_bits;

   if(exp_nibbles == 0)
      return 1;

   secure_vector<word> e_bits(m_params->p_words());
   secure_vector<word> ws;

   const_time_lookup(e_bits, m_g, scalar.get_substring(m_window_bits*(exp_nibbles-1), m_window_bits));
   Montgomery_Int x(m_params, e_bits.data(), e_bits.size(), false);

   for(size_t i = exp_nibbles - 1; i > 0; --i)
      {
      x.square_this_n_times(ws, m_window_bits);
      const_time_lookup(e_bits, m_g, scalar.get_substring(m_window_bits*(i-1), m_window_bits));
      x.mul_by(e_bits, ws);
      }

   x.const_time_unpoison();
   return x.value();
   }

BigInt Montgomery_Exponentation_State::exponentiation_vartime(const BigInt& scalar) const
   {
   BOTAN_ASSERT_NOMSG(m_const_time == false);

   const size_t exp_nibbles = (scalar.bits() + m_window_bits - 1) / m_window_bits;

   secure_vector<word> ws;

   if(exp_nibbles == 0)
      return 1;

   Montgomery_Int x = m_g[scalar.get_substring(m_window_bits*(exp_nibbles-1), m_window_bits)];

   for(size_t i = exp_nibbles - 1; i > 0; --i)
      {
      x.square_this_n_times(ws, m_window_bits);

      const uint32_t nibble = scalar.get_substring(m_window_bits*(i-1), m_window_bits);
      if(nibble > 0)
         x.mul_by(m_g[nibble], ws);
      }

   x.const_time_unpoison();
   return x.value();
   }

std::shared_ptr<const Montgomery_Exponentation_State>
monty_precompute(std::shared_ptr<const Montgomery_Params> params,
                 const BigInt& g,
                 size_t window_bits,
                 bool const_time)
   {
   return std::make_shared<const Montgomery_Exponentation_State>(params, g, window_bits, const_time);
   }

BigInt monty_execute(const Montgomery_Exponentation_State& precomputed_state,
                     const BigInt& k, size_t max_k_bits)
   {
   return precomputed_state.exponentiation(k, max_k_bits);
   }

BigInt monty_execute_vartime(const Montgomery_Exponentation_State& precomputed_state,
                             const BigInt& k)
   {
   return precomputed_state.exponentiation_vartime(k);
   }

BigInt monty_multi_exp(std::shared_ptr<const Montgomery_Params> params_p,
                       const BigInt& x_bn,
                       const BigInt& z1,
                       const BigInt& y_bn,
                       const BigInt& z2)
   {
   if(z1.is_negative() || z2.is_negative())
      throw Invalid_Argument("multi_exponentiate exponents must be positive");

   const size_t z_bits = round_up(std::max(z1.bits(), z2.bits()), 2);

   secure_vector<word> ws;

   const Montgomery_Int one(params_p, params_p->R1(), false);
   //const Montgomery_Int one(params_p, 1);

   const Montgomery_Int x1(params_p, x_bn);
   const Montgomery_Int x2 = x1.square(ws);
   const Montgomery_Int x3 = x2.mul(x1, ws);

   const Montgomery_Int y1(params_p, y_bn);
   const Montgomery_Int y2 = y1.square(ws);
   const Montgomery_Int y3 = y2.mul(y1, ws);

   const Montgomery_Int y1x1 = y1.mul(x1, ws);
   const Montgomery_Int y1x2 = y1.mul(x2, ws);
   const Montgomery_Int y1x3 = y1.mul(x3, ws);

   const Montgomery_Int y2x1 = y2.mul(x1, ws);
   const Montgomery_Int y2x2 = y2.mul(x2, ws);
   const Montgomery_Int y2x3 = y2.mul(x3, ws);

   const Montgomery_Int y3x1 = y3.mul(x1, ws);
   const Montgomery_Int y3x2 = y3.mul(x2, ws);
   const Montgomery_Int y3x3 = y3.mul(x3, ws);

   const Montgomery_Int* M[16] = {
      &one,
      &x1,                    // 0001
      &x2,                    // 0010
      &x3,                    // 0011
      &y1,                    // 0100
      &y1x1,
      &y1x2,
      &y1x3,
      &y2,                    // 1000
      &y2x1,
      &y2x2,
      &y2x3,
      &y3,                    // 1100
      &y3x1,
      &y3x2,
      &y3x3
   };

   Montgomery_Int H = one;

   for(size_t i = 0; i != z_bits; i += 2)
      {
      if(i > 0)
         {
         H.square_this(ws);
         H.square_this(ws);
         }

      const uint32_t z1_b = z1.get_substring(z_bits - i - 2, 2);
      const uint32_t z2_b = z2.get_substring(z_bits - i - 2, 2);

      const uint32_t z12 = (4*z2_b) + z1_b;

      H.mul_by(*M[z12], ws);
      }

   return H.value();
   }

}

/*
* Fused and Important MP Algorithms
* (C) 1999-2007 Jack Lloyd
*     2016 Matthias Gierlings
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/*
* Square a BigInt
*/
BigInt square(const BigInt& x)
   {
   BigInt z = x;
   secure_vector<word> ws;
   z.square(ws);
   return z;
   }

/*
* Multiply-Add Operation
*/
BigInt mul_add(const BigInt& a, const BigInt& b, const BigInt& c)
   {
   if(c.is_negative())
      throw Invalid_Argument("mul_add: Third argument must be > 0");

   BigInt::Sign sign = BigInt::Positive;
   if(a.sign() != b.sign())
      sign = BigInt::Negative;

   const size_t a_sw = a.sig_words();
   const size_t b_sw = b.sig_words();
   const size_t c_sw = c.sig_words();

   BigInt r(sign, std::max(a_sw + b_sw, c_sw) + 1);
   secure_vector<word> workspace(r.size());

   bigint_mul(r.mutable_data(), r.size(),
              a.data(), a.size(), a_sw,
              b.data(), b.size(), b_sw,
              workspace.data(), workspace.size());

   const size_t r_size = std::max(r.sig_words(), c_sw);
   bigint_add2(r.mutable_data(), r_size, c.data(), c_sw);
   return r;
   }

/*
* Subtract-Multiply Operation
*/
BigInt sub_mul(const BigInt& a, const BigInt& b, const BigInt& c)
   {
   if(a.is_negative() || b.is_negative())
      throw Invalid_Argument("sub_mul: First two arguments must be >= 0");

   BigInt r = a;
   r -= b;
   r *= c;
   return r;
   }

/*
* Multiply-Subtract Operation
*/
BigInt mul_sub(const BigInt& a, const BigInt& b, const BigInt& c)
   {
   if(c.is_negative() || c.is_zero())
      throw Invalid_Argument("mul_sub: Third argument must be > 0");

   BigInt r = a;
   r *= b;
   r -= c;
   return r;
   }

}
/*
* NIST prime reductions
* (C) 2014,2015,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

const BigInt& prime_p521()
   {
   static const BigInt p521("0x1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
                               "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");

   return p521;
   }

void redc_p521(BigInt& x, secure_vector<word>& ws)
   {
   const size_t p_full_words = 521 / BOTAN_MP_WORD_BITS;
   const size_t p_top_bits = 521 % BOTAN_MP_WORD_BITS;
   const size_t p_words = p_full_words + 1;

#if (BOTAN_MP_WORD_BITS == 64)
   static const word p521_words[p_words] = {
      0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
      0x1FF };
#else
   static const word p521_words[p_words] = {
      0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
      0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
      0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
      0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
      0x1FF };
#endif

   if(ws.size() < p_words + 1)
      ws.resize(p_words + 1);

   clear_mem(ws.data(), ws.size());
   bigint_shr2(ws.data(), x.data(), std::min(x.size(), 2*p_words), p_full_words, p_top_bits);

   x.mask_bits(521);
   x.grow_to(p_words);

   // Word-level carry will be zero
   word carry = bigint_add3_nc(x.mutable_data(), x.data(), p_words, ws.data(), p_words);
   BOTAN_ASSERT_EQUAL(carry, 0, "Final carry in P-521 reduction");

   const word top_word = x.word_at(p_full_words);

   /*
   * Check if we need to reduce modulo P
   * There are two possible cases:
   * - The result overflowed past 521 bits, in which case bit 522 will be set
   * - The result is exactly 2**521 - 1
   */
   const auto bit_522_set = CT::Mask<word>::expand(top_word >> p_top_bits);

   word and_512 = MP_WORD_MAX;
   for(size_t i = 0; i != p_full_words; ++i)
      and_512 &= x.word_at(i);
   const auto all_512_low_bits_set = CT::Mask<word>::is_equal(and_512, MP_WORD_MAX);
   const auto has_p521_top_word = CT::Mask<word>::is_equal(top_word, 0x1FF);
   const auto is_p521 = all_512_low_bits_set & has_p521_top_word;

   const auto needs_reduction = is_p521 | bit_522_set;

   bigint_cnd_sub(needs_reduction.value(), x.mutable_data(), p521_words, p_words);
   }

namespace {

/**
* Treating this MPI as a sequence of 32-bit words in big-endian
* order, return word i. The array is assumed to be large enough.
*/
inline uint32_t get_uint32(const word xw[], size_t i)
   {
#if (BOTAN_MP_WORD_BITS == 32)
   return xw[i];
#else
   return static_cast<uint32_t>(xw[i/2] >> ((i % 2)*32));
#endif
   }

inline void set_words(word x[], size_t i, uint32_t R0, uint32_t R1)
   {
#if (BOTAN_MP_WORD_BITS == 32)
   x[i] = R0;
   x[i+1] = R1;
#else
   x[i/2] = (static_cast<uint64_t>(R1) << 32) | R0;
#endif
   }

}

const BigInt& prime_p192()
   {
   static const BigInt p192("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFF");
   return p192;
   }

void redc_p192(BigInt& x, secure_vector<word>& ws)
   {
   BOTAN_UNUSED(ws);

   static const size_t p192_limbs = 192 / BOTAN_MP_WORD_BITS;

   x.grow_to(2*p192_limbs);
   word* xw = x.mutable_data();

   const uint64_t X00 = get_uint32(xw,  0);
   const uint64_t X01 = get_uint32(xw,  1);
   const uint64_t X02 = get_uint32(xw,  2);
   const uint64_t X03 = get_uint32(xw,  3);
   const uint64_t X04 = get_uint32(xw,  4);
   const uint64_t X05 = get_uint32(xw,  5);
   const uint64_t X06 = get_uint32(xw,  6);
   const uint64_t X07 = get_uint32(xw,  7);
   const uint64_t X08 = get_uint32(xw,  8);
   const uint64_t X09 = get_uint32(xw,  9);
   const uint64_t X10 = get_uint32(xw, 10);
   const uint64_t X11 = get_uint32(xw, 11);

   const uint64_t S0 = X00 + X06 + X10;
   const uint64_t S1 = X01 + X07 + X11;
   const uint64_t S2 = X02 + X06 + X08 + X10;
   const uint64_t S3 = X03 + X07 + X09 + X11;
   const uint64_t S4 = X04 + X08 + X10;
   const uint64_t S5 = X05 + X09 + X11;

   uint64_t S = 0;
   uint32_t R0 = 0, R1 = 0;

   S += S0;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S1;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 0, R0, R1);

   S += S2;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S3;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 2, R0, R1);

   S += S4;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S5;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 4, R0, R1);

   // No underflow possible

   /*
   This is a table of (i*P-192) % 2**192 for i in 1...3
   */
   static const word p192_mults[3][p192_limbs] = {
#if (BOTAN_MP_WORD_BITS == 64)
      {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF},
      {0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFD, 0xFFFFFFFFFFFFFFFF},
      {0xFFFFFFFFFFFFFFFD, 0xFFFFFFFFFFFFFFFC, 0xFFFFFFFFFFFFFFFF},
#else
      {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
      {0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFD, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
      {0xFFFFFFFD, 0xFFFFFFFF, 0xFFFFFFFC, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
#endif
   };

   CT::unpoison(S);
   BOTAN_ASSERT(S <= 2, "Expected overflow");

   BOTAN_ASSERT_NOMSG(x.size() >= p192_limbs + 1);
   x.mask_bits(192);
   word borrow = bigint_sub2(x.mutable_data(), p192_limbs + 1, p192_mults[S], p192_limbs);
   BOTAN_DEBUG_ASSERT(borrow == 0 || borrow == 1);
   bigint_cnd_add(borrow, x.mutable_data(), p192_limbs + 1, p192_mults[0], p192_limbs);
   }

const BigInt& prime_p224()
   {
   static const BigInt p224("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000000000000000000001");
   return p224;
   }

void redc_p224(BigInt& x, secure_vector<word>& ws)
   {
   static const size_t p224_limbs = (BOTAN_MP_WORD_BITS == 32) ? 7 : 4;

   BOTAN_UNUSED(ws);

   x.grow_to(2*p224_limbs);
   word* xw = x.mutable_data();

   const int64_t X00 = get_uint32(xw,  0);
   const int64_t X01 = get_uint32(xw,  1);
   const int64_t X02 = get_uint32(xw,  2);
   const int64_t X03 = get_uint32(xw,  3);
   const int64_t X04 = get_uint32(xw,  4);
   const int64_t X05 = get_uint32(xw,  5);
   const int64_t X06 = get_uint32(xw,  6);
   const int64_t X07 = get_uint32(xw,  7);
   const int64_t X08 = get_uint32(xw,  8);
   const int64_t X09 = get_uint32(xw,  9);
   const int64_t X10 = get_uint32(xw, 10);
   const int64_t X11 = get_uint32(xw, 11);
   const int64_t X12 = get_uint32(xw, 12);
   const int64_t X13 = get_uint32(xw, 13);

   // One full copy of P224 is added, so the result is always positive

   const int64_t S0 = 0x00000001 + X00 - X07 - X11;
   const int64_t S1 = 0x00000000 + X01 - X08 - X12;
   const int64_t S2 = 0x00000000 + X02 - X09 - X13;
   const int64_t S3 = 0xFFFFFFFF + X03 + X07 + X11 - X10;
   const int64_t S4 = 0xFFFFFFFF + X04 + X08 + X12 - X11;
   const int64_t S5 = 0xFFFFFFFF + X05 + X09 + X13 - X12;
   const int64_t S6 = 0xFFFFFFFF + X06 + X10 - X13;

   int64_t S = 0;
   uint32_t R0 = 0, R1 = 0;

   S += S0;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S1;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 0, R0, R1);

   S += S2;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S3;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 2, R0, R1);

   S += S4;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S5;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 4, R0, R1);

   S += S6;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 6, R0, 0);

   static const word p224_mults[3][p224_limbs] = {
#if (BOTAN_MP_WORD_BITS == 64)
    {0x0000000000000001, 0xFFFFFFFF00000000, 0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF},
    {0x0000000000000002, 0xFFFFFFFE00000000, 0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF},
    {0x0000000000000003, 0xFFFFFFFD00000000, 0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF},
#else
    {0x00000001, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
    {0x00000002, 0x00000000, 0x00000000, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
    {0x00000003, 0x00000000, 0x00000000, 0xFFFFFFFD, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}
#endif

   };

   CT::unpoison(S);
   BOTAN_ASSERT(S >= 0 && S <= 2, "Expected overflow");

   BOTAN_ASSERT_NOMSG(x.size() >= p224_limbs + 1);
   x.mask_bits(224);
   word borrow = bigint_sub2(x.mutable_data(), p224_limbs + 1, p224_mults[S], p224_limbs);
   BOTAN_DEBUG_ASSERT(borrow == 0 || borrow == 1);
   bigint_cnd_add(borrow, x.mutable_data(), p224_limbs + 1, p224_mults[0], p224_limbs);
   }

const BigInt& prime_p256()
   {
   static const BigInt p256("0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF");
   return p256;
   }

void redc_p256(BigInt& x, secure_vector<word>& ws)
   {
   static const size_t p256_limbs = (BOTAN_MP_WORD_BITS == 32) ? 8 : 4;

   BOTAN_UNUSED(ws);

   x.grow_to(2*p256_limbs);
   word* xw = x.mutable_data();

   const int64_t X00 = get_uint32(xw,  0);
   const int64_t X01 = get_uint32(xw,  1);
   const int64_t X02 = get_uint32(xw,  2);
   const int64_t X03 = get_uint32(xw,  3);
   const int64_t X04 = get_uint32(xw,  4);
   const int64_t X05 = get_uint32(xw,  5);
   const int64_t X06 = get_uint32(xw,  6);
   const int64_t X07 = get_uint32(xw,  7);
   const int64_t X08 = get_uint32(xw,  8);
   const int64_t X09 = get_uint32(xw,  9);
   const int64_t X10 = get_uint32(xw, 10);
   const int64_t X11 = get_uint32(xw, 11);
   const int64_t X12 = get_uint32(xw, 12);
   const int64_t X13 = get_uint32(xw, 13);
   const int64_t X14 = get_uint32(xw, 14);
   const int64_t X15 = get_uint32(xw, 15);

   // Adds 6 * P-256 to prevent underflow
   const int64_t S0 = 0xFFFFFFFA + X00 + X08 + X09 - (X11 + X12 + X13) - X14;
   const int64_t S1 = 0xFFFFFFFF + X01 + X09 + X10 - X12 - (X13 + X14 + X15);
   const int64_t S2 = 0xFFFFFFFF + X02 + X10 + X11 - (X13 + X14 + X15);
   const int64_t S3 = 0x00000005 + X03 + (X11 + X12)*2 + X13 - X15 - X08 - X09;
   const int64_t S4 = 0x00000000 + X04 + (X12 + X13)*2 + X14 - X09 - X10;
   const int64_t S5 = 0x00000000 + X05 + (X13 + X14)*2 + X15 - X10 - X11;
   const int64_t S6 = 0x00000006 + X06 + X13 + X14*3 + X15*2 - X08 - X09;
   const int64_t S7 = 0xFFFFFFFA + X07 + X15*3 + X08 - X10 - (X11 + X12 + X13);

   int64_t S = 0;

   uint32_t R0 = 0, R1 = 0;

   S += S0;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S1;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 0, R0, R1);

   S += S2;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S3;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 2, R0, R1);

   S += S4;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S5;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 4, R0, R1);

   S += S6;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S7;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;
   set_words(xw, 6, R0, R1);

   S += 5; // the top digits of 6*P-256

   /*
   This is a table of (i*P-256) % 2**256 for i in 1...10
   */
   static const word p256_mults[11][p256_limbs] = {
#if (BOTAN_MP_WORD_BITS == 64)
      {0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001},
      {0xFFFFFFFFFFFFFFFE, 0x00000001FFFFFFFF, 0x0000000000000000, 0xFFFFFFFE00000002},
      {0xFFFFFFFFFFFFFFFD, 0x00000002FFFFFFFF, 0x0000000000000000, 0xFFFFFFFD00000003},
      {0xFFFFFFFFFFFFFFFC, 0x00000003FFFFFFFF, 0x0000000000000000, 0xFFFFFFFC00000004},
      {0xFFFFFFFFFFFFFFFB, 0x00000004FFFFFFFF, 0x0000000000000000, 0xFFFFFFFB00000005},
      {0xFFFFFFFFFFFFFFFA, 0x00000005FFFFFFFF, 0x0000000000000000, 0xFFFFFFFA00000006},
      {0xFFFFFFFFFFFFFFF9, 0x00000006FFFFFFFF, 0x0000000000000000, 0xFFFFFFF900000007},
      {0xFFFFFFFFFFFFFFF8, 0x00000007FFFFFFFF, 0x0000000000000000, 0xFFFFFFF800000008},
      {0xFFFFFFFFFFFFFFF7, 0x00000008FFFFFFFF, 0x0000000000000000, 0xFFFFFFF700000009},
      {0xFFFFFFFFFFFFFFF6, 0x00000009FFFFFFFF, 0x0000000000000000, 0xFFFFFFF60000000A},
      {0xFFFFFFFFFFFFFFF5, 0x0000000AFFFFFFFF, 0x0000000000000000, 0xFFFFFFF50000000B},
#else
      {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0xFFFFFFFF},
      {0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000001, 0x00000000, 0x00000000, 0x00000002, 0xFFFFFFFE},
      {0xFFFFFFFD, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000002, 0x00000000, 0x00000000, 0x00000003, 0xFFFFFFFD},
      {0xFFFFFFFC, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000003, 0x00000000, 0x00000000, 0x00000004, 0xFFFFFFFC},
      {0xFFFFFFFB, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000004, 0x00000000, 0x00000000, 0x00000005, 0xFFFFFFFB},
      {0xFFFFFFFA, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000005, 0x00000000, 0x00000000, 0x00000006, 0xFFFFFFFA},
      {0xFFFFFFF9, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000006, 0x00000000, 0x00000000, 0x00000007, 0xFFFFFFF9},
      {0xFFFFFFF8, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000007, 0x00000000, 0x00000000, 0x00000008, 0xFFFFFFF8},
      {0xFFFFFFF7, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000008, 0x00000000, 0x00000000, 0x00000009, 0xFFFFFFF7},
      {0xFFFFFFF6, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000009, 0x00000000, 0x00000000, 0x0000000A, 0xFFFFFFF6},
      {0xFFFFFFF5, 0xFFFFFFFF, 0xFFFFFFFF, 0x0000000A, 0x00000000, 0x00000000, 0x0000000B, 0xFFFFFFF5},
#endif
   };

   CT::unpoison(S);
   BOTAN_ASSERT(S >= 0 && S <= 10, "Expected overflow");

   BOTAN_ASSERT_NOMSG(x.size() >= p256_limbs + 1);
   x.mask_bits(256);
   word borrow = bigint_sub2(x.mutable_data(), p256_limbs + 1, p256_mults[S], p256_limbs);
   BOTAN_DEBUG_ASSERT(borrow == 0 || borrow == 1);
   bigint_cnd_add(borrow, x.mutable_data(), p256_limbs + 1, p256_mults[0], p256_limbs);
   }

const BigInt& prime_p384()
   {
   static const BigInt p384("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFF0000000000000000FFFFFFFF");
   return p384;
   }

void redc_p384(BigInt& x, secure_vector<word>& ws)
   {
   BOTAN_UNUSED(ws);

   static const size_t p384_limbs = (BOTAN_MP_WORD_BITS == 32) ? 12 : 6;

   x.grow_to(2*p384_limbs);
   word* xw = x.mutable_data();

   const int64_t X00 = get_uint32(xw,  0);
   const int64_t X01 = get_uint32(xw,  1);
   const int64_t X02 = get_uint32(xw,  2);
   const int64_t X03 = get_uint32(xw,  3);
   const int64_t X04 = get_uint32(xw,  4);
   const int64_t X05 = get_uint32(xw,  5);
   const int64_t X06 = get_uint32(xw,  6);
   const int64_t X07 = get_uint32(xw,  7);
   const int64_t X08 = get_uint32(xw,  8);
   const int64_t X09 = get_uint32(xw,  9);
   const int64_t X10 = get_uint32(xw, 10);
   const int64_t X11 = get_uint32(xw, 11);
   const int64_t X12 = get_uint32(xw, 12);
   const int64_t X13 = get_uint32(xw, 13);
   const int64_t X14 = get_uint32(xw, 14);
   const int64_t X15 = get_uint32(xw, 15);
   const int64_t X16 = get_uint32(xw, 16);
   const int64_t X17 = get_uint32(xw, 17);
   const int64_t X18 = get_uint32(xw, 18);
   const int64_t X19 = get_uint32(xw, 19);
   const int64_t X20 = get_uint32(xw, 20);
   const int64_t X21 = get_uint32(xw, 21);
   const int64_t X22 = get_uint32(xw, 22);
   const int64_t X23 = get_uint32(xw, 23);

   // One copy of P-384 is added to prevent underflow
   const int64_t S0 = 0xFFFFFFFF + X00 + X12 + X20 + X21 - X23;
   const int64_t S1 = 0x00000000 + X01 + X13 + X22 + X23 - X12 - X20;
   const int64_t S2 = 0x00000000 + X02 + X14 + X23 - X13 - X21;
   const int64_t S3 = 0xFFFFFFFF + X03 + X12 + X15 + X20 + X21 - X14 - X22 - X23;
   const int64_t S4 = 0xFFFFFFFE + X04 + X12 + X13 + X16 + X20 + X21*2 + X22 - X15 - X23*2;
   const int64_t S5 = 0xFFFFFFFF + X05 + X13 + X14 + X17 + X21 + X22*2 + X23 - X16;
   const int64_t S6 = 0xFFFFFFFF + X06 + X14 + X15 + X18 + X22 + X23*2 - X17;
   const int64_t S7 = 0xFFFFFFFF + X07 + X15 + X16 + X19 + X23 - X18;
   const int64_t S8 = 0xFFFFFFFF + X08 + X16 + X17 + X20 - X19;
   const int64_t S9 = 0xFFFFFFFF + X09 + X17 + X18 + X21 - X20;
   const int64_t SA = 0xFFFFFFFF + X10 + X18 + X19 + X22 - X21;
   const int64_t SB = 0xFFFFFFFF + X11 + X19 + X20 + X23 - X22;

   int64_t S = 0;

   uint32_t R0 = 0, R1 = 0;

   S += S0;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S1;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 0, R0, R1);

   S += S2;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S3;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 2, R0, R1);

   S += S4;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S5;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 4, R0, R1);

   S += S6;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S7;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 6, R0, R1);

   S += S8;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += S9;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 8, R0, R1);

   S += SA;
   R0 = static_cast<uint32_t>(S);
   S >>= 32;

   S += SB;
   R1 = static_cast<uint32_t>(S);
   S >>= 32;

   set_words(xw, 10, R0, R1);

   /*
   This is a table of (i*P-384) % 2**384 for i in 1...4
   */
   static const word p384_mults[5][p384_limbs] = {
#if (BOTAN_MP_WORD_BITS == 64)
      {0x00000000FFFFFFFF, 0xFFFFFFFF00000000, 0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF},
      {0x00000001FFFFFFFE, 0xFFFFFFFE00000000, 0xFFFFFFFFFFFFFFFD, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF},
      {0x00000002FFFFFFFD, 0xFFFFFFFD00000000, 0xFFFFFFFFFFFFFFFC, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF},
      {0x00000003FFFFFFFC, 0xFFFFFFFC00000000, 0xFFFFFFFFFFFFFFFB, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF},
      {0x00000004FFFFFFFB, 0xFFFFFFFB00000000, 0xFFFFFFFFFFFFFFFA, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF},

#else
      {0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF,
       0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
      {0xFFFFFFFE, 0x00000001, 0x00000000, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFF,
       0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
      {0xFFFFFFFD, 0x00000002, 0x00000000, 0xFFFFFFFD, 0xFFFFFFFC, 0xFFFFFFFF,
       0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
      {0xFFFFFFFC, 0x00000003, 0x00000000, 0xFFFFFFFC, 0xFFFFFFFB, 0xFFFFFFFF,
       0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
      {0xFFFFFFFB, 0x00000004, 0x00000000, 0xFFFFFFFB, 0xFFFFFFFA, 0xFFFFFFFF,
       0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
#endif
   };

   CT::unpoison(S);
   BOTAN_ASSERT(S >= 0 && S <= 4, "Expected overflow");

   BOTAN_ASSERT_NOMSG(x.size() >= p384_limbs + 1);
   x.mask_bits(384);
   word borrow = bigint_sub2(x.mutable_data(), p384_limbs + 1, p384_mults[S], p384_limbs);
   BOTAN_DEBUG_ASSERT(borrow == 0 || borrow == 1);
   bigint_cnd_add(borrow, x.mutable_data(), p384_limbs + 1, p384_mults[0], p384_limbs);
   }

}
/*
* Number Theory Functions
* (C) 1999-2011,2016,2018,2019 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

void sub_abs(BigInt& z, const BigInt& x, const BigInt& y)
   {
   const size_t x_sw = x.sig_words();
   const size_t y_sw = y.sig_words();
   z.resize(std::max(x_sw, y_sw));

   bigint_sub_abs(z.mutable_data(),
                  x.data(), x_sw,
                  y.data(), y_sw);
   }

}

/*
* Return the number of 0 bits at the end of n
*/
size_t low_zero_bits(const BigInt& n)
   {
   size_t low_zero = 0;

   auto seen_nonempty_word = CT::Mask<word>::cleared();

   for(size_t i = 0; i != n.size(); ++i)
      {
      const word x = n.word_at(i);

      // ctz(0) will return sizeof(word)
      const size_t tz_x = ctz(x);

      // if x > 0 we want to count tz_x in total but not any
      // further words, so set the mask after the addition
      low_zero += seen_nonempty_word.if_not_set_return(tz_x);

      seen_nonempty_word |= CT::Mask<word>::expand(x);
      }

   // if we saw no words with x > 0 then n == 0 and the value we have
   // computed is meaningless. Instead return 0 in that case.
   return seen_nonempty_word.if_set_return(low_zero);
   }

namespace {

size_t safegcd_loop_bound(size_t f_bits, size_t g_bits)
   {
   const size_t d = std::max(f_bits, g_bits);

   if(d < 46)
      return (49*d + 80) / 17;
   else
      return (49*d + 57) / 17;
   }

}

/*
* Calculate the GCD
*/
BigInt gcd(const BigInt& a, const BigInt& b)
   {
   if(a.is_zero())
      return abs(b);
   if(b.is_zero())
      return abs(a);
   if(a == 1 || b == 1)
      return 1;

   // See https://gcd.cr.yp.to/safegcd-20190413.pdf fig 1.2

   BigInt f = a;
   BigInt g = b;
   f.const_time_poison();
   g.const_time_poison();

   f.set_sign(BigInt::Positive);
   g.set_sign(BigInt::Positive);

   const size_t common2s = std::min(low_zero_bits(f), low_zero_bits(g));
   CT::unpoison(common2s);

   f >>= common2s;
   g >>= common2s;

   f.ct_cond_swap(f.is_even(), g);

   int32_t delta = 1;

   const size_t loop_cnt = safegcd_loop_bound(f.bits(), g.bits());

   BigInt newg, t;
   for(size_t i = 0; i != loop_cnt; ++i)
      {
      sub_abs(newg, f, g);

      const bool need_swap = (g.is_odd() && delta > 0);

      // if(need_swap) { delta *= -1 } else { delta *= 1 }
      delta *= CT::Mask<uint8_t>::expand(need_swap).if_not_set_return(2) - 1;
      f.ct_cond_swap(need_swap, g);
      g.ct_cond_swap(need_swap, newg);

      delta += 1;

      g.ct_cond_add(g.is_odd(), f);
      g >>= 1;
      }

   f <<= common2s;

   f.const_time_unpoison();
   g.const_time_unpoison();

   BOTAN_ASSERT_NOMSG(g.is_zero());

   return f;
   }

/*
* Calculate the LCM
*/
BigInt lcm(const BigInt& a, const BigInt& b)
   {
   return ct_divide(a * b, gcd(a, b));
   }

/*
* Modular Exponentiation
*/
BigInt power_mod(const BigInt& base, const BigInt& exp, const BigInt& mod)
   {
   if(mod.is_negative() || mod == 1)
      {
      return 0;
      }

   if(base.is_zero() || mod.is_zero())
      {
      if(exp.is_zero())
         return 1;
      return 0;
      }

   Modular_Reducer reduce_mod(mod);

   const size_t exp_bits = exp.bits();

   if(mod.is_odd())
      {
      const size_t powm_window = 4;

      auto monty_mod = std::make_shared<Montgomery_Params>(mod, reduce_mod);
      auto powm_base_mod = monty_precompute(monty_mod, reduce_mod.reduce(base), powm_window);
      return monty_execute(*powm_base_mod, exp, exp_bits);
      }

   /*
   Support for even modulus is just a convenience and not considered
   cryptographically important, so this implementation is slow ...
   */
   BigInt accum = 1;
   BigInt g = reduce_mod.reduce(base);
   BigInt t;

   for(size_t i = 0; i != exp_bits; ++i)
      {
      t = reduce_mod.multiply(g, accum);
      g = reduce_mod.square(g);
      accum.ct_cond_assign(exp.get_bit(i), t);
      }
   return accum;
   }


BigInt is_perfect_square(const BigInt& C)
   {
   if(C < 1)
      throw Invalid_Argument("is_perfect_square requires C >= 1");
   if(C == 1)
      return 1;

   const size_t n = C.bits();
   const size_t m = (n + 1) / 2;
   const BigInt B = C + BigInt::power_of_2(m);

   BigInt X = BigInt::power_of_2(m) - 1;
   BigInt X2 = (X*X);

   for(;;)
      {
      X = (X2 + C) / (2*X);
      X2 = (X*X);

      if(X2 < B)
         break;
      }

   if(X2 == C)
      return X;
   else
      return 0;
   }

/*
* Test for primality using Miller-Rabin
*/
bool is_prime(const BigInt& n,
              RandomNumberGenerator& rng,
              size_t prob,
              bool is_random)
   {
   if(n == 2)
      return true;
   if(n <= 1 || n.is_even())
      return false;

   const size_t n_bits = n.bits();

   // Fast path testing for small numbers (<= 65521)
   if(n_bits <= 16)
      {
      const uint16_t num = static_cast<uint16_t>(n.word_at(0));

      return std::binary_search(PRIMES, PRIMES + PRIME_TABLE_SIZE, num);
      }

   Modular_Reducer mod_n(n);

   if(rng.is_seeded())
      {
      const size_t t = miller_rabin_test_iterations(n_bits, prob, is_random);

      if(is_miller_rabin_probable_prime(n, mod_n, rng, t) == false)
         return false;

      if(is_random)
         return true;
      else
         return is_lucas_probable_prime(n, mod_n);
      }
   else
      {
      return is_bailie_psw_probable_prime(n, mod_n);
      }
   }

}
/*
* Modular Exponentiation Proxy
* (C) 1999-2007,2012,2018,2019 Jack Lloyd
*     2016 Matthias Gierlings
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <vector>

namespace Botan {

class Modular_Exponentiator
   {
   public:
      virtual void set_base(const BigInt&) = 0;
      virtual void set_exponent(const BigInt&) = 0;
      virtual BigInt execute() const = 0;
      virtual Modular_Exponentiator* copy() const = 0;

      Modular_Exponentiator() = default;
      Modular_Exponentiator(const Modular_Exponentiator&) = default;
      Modular_Exponentiator & operator=(const Modular_Exponentiator&) = default;
      virtual ~Modular_Exponentiator() = default;
   };

namespace {

/**
* Fixed Window Exponentiator
*/
class Fixed_Window_Exponentiator final : public Modular_Exponentiator
   {
   public:
      void set_exponent(const BigInt& e) override { m_exp = e; }
      void set_base(const BigInt&) override;
      BigInt execute() const override;

      Modular_Exponentiator* copy() const override
         { return new Fixed_Window_Exponentiator(*this); }

      Fixed_Window_Exponentiator(const BigInt&, Power_Mod::Usage_Hints);
   private:
      Modular_Reducer m_reducer;
      BigInt m_exp;
      size_t m_window_bits;
      std::vector<BigInt> m_g;
      Power_Mod::Usage_Hints m_hints;
   };

void Fixed_Window_Exponentiator::set_base(const BigInt& base)
   {
   m_window_bits = Power_Mod::window_bits(m_exp.bits(), base.bits(), m_hints);

   m_g.resize(static_cast<size_t>(1) << m_window_bits);
   m_g[0] = 1;
   m_g[1] = m_reducer.reduce(base);

   for(size_t i = 2; i != m_g.size(); ++i)
      m_g[i] = m_reducer.multiply(m_g[i-1], m_g[1]);
   }

BigInt Fixed_Window_Exponentiator::execute() const
   {
   const size_t exp_nibbles = (m_exp.bits() + m_window_bits - 1) / m_window_bits;

   BigInt x = 1;

   for(size_t i = exp_nibbles; i > 0; --i)
      {
      for(size_t j = 0; j != m_window_bits; ++j)
         x = m_reducer.square(x);

      const uint32_t nibble = m_exp.get_substring(m_window_bits*(i-1), m_window_bits);

      // not const time:
      x = m_reducer.multiply(x, m_g[nibble]);
      }
   return x;
   }

/*
* Fixed_Window_Exponentiator Constructor
*/
Fixed_Window_Exponentiator::Fixed_Window_Exponentiator(const BigInt& n,
                                                       Power_Mod::Usage_Hints hints)
   : m_reducer{Modular_Reducer(n)}, m_exp{}, m_window_bits{}, m_g{}, m_hints{hints}
   {}

class Montgomery_Exponentiator final : public Modular_Exponentiator
   {
   public:
      void set_exponent(const BigInt& e) override { m_e = e; }
      void set_base(const BigInt&) override;
      BigInt execute() const override;

      Modular_Exponentiator* copy() const override
         { return new Montgomery_Exponentiator(*this); }

      Montgomery_Exponentiator(const BigInt&, Power_Mod::Usage_Hints);
   private:
      BigInt m_p;
      Modular_Reducer m_mod_p;
      std::shared_ptr<const Montgomery_Params> m_monty_params;
      std::shared_ptr<const Montgomery_Exponentation_State> m_monty;

      BigInt m_e;
      Power_Mod::Usage_Hints m_hints;
   };

void Montgomery_Exponentiator::set_base(const BigInt& base)
   {
   size_t window_bits = Power_Mod::window_bits(m_e.bits(), base.bits(), m_hints);
   m_monty = monty_precompute(m_monty_params, m_mod_p.reduce(base), window_bits);
   }

BigInt Montgomery_Exponentiator::execute() const
   {
   /*
   This leaks size of e via loop iterations, not possible to fix without
   breaking this API. Round up to avoid leaking fine details.
   */
   return monty_execute(*m_monty, m_e, round_up(m_e.bits(), 8));
   }

Montgomery_Exponentiator::Montgomery_Exponentiator(const BigInt& mod,
                                                   Power_Mod::Usage_Hints hints) :
   m_p(mod),
   m_mod_p(mod),
   m_monty_params(std::make_shared<Montgomery_Params>(m_p, m_mod_p)),
   m_hints(hints)
   {
   }

}

/*
* Power_Mod Constructor
*/
Power_Mod::Power_Mod(const BigInt& n, Usage_Hints hints, bool disable_monty)
   {
   set_modulus(n, hints, disable_monty);
   }

Power_Mod::~Power_Mod() { /* for ~unique_ptr */ }

/*
* Power_Mod Copy Constructor
*/
Power_Mod::Power_Mod(const Power_Mod& other)
   {
   if(other.m_core.get())
      m_core.reset(other.m_core->copy());
   }

/*
* Power_Mod Assignment Operator
*/
Power_Mod& Power_Mod::operator=(const Power_Mod& other)
   {
   if(this != &other)
      {
      if(other.m_core)
         m_core.reset(other.m_core->copy());
      else
         m_core.reset();
      }
   return (*this);
   }

/*
* Set the modulus
*/
void Power_Mod::set_modulus(const BigInt& n, Usage_Hints hints, bool disable_monty) const
   {
   // Allow set_modulus(0) to mean "drop old state"

   m_core.reset();

   if(n != 0)
      {
      if(n.is_odd() && disable_monty == false)
         m_core.reset(new Montgomery_Exponentiator(n, hints));
      else
         m_core.reset(new Fixed_Window_Exponentiator(n, hints));
      }
   }

/*
* Set the base
*/
void Power_Mod::set_base(const BigInt& b) const
   {
   if(b.is_negative())
      throw Invalid_Argument("Power_Mod::set_base: arg must be non-negative");

   if(!m_core)
      throw Internal_Error("Power_Mod::set_base: m_core was NULL");
   m_core->set_base(b);
   }

/*
* Set the exponent
*/
void Power_Mod::set_exponent(const BigInt& e) const
   {
   if(e.is_negative())
      throw Invalid_Argument("Power_Mod::set_exponent: arg must be > 0");

   if(!m_core)
      throw Internal_Error("Power_Mod::set_exponent: m_core was NULL");
   m_core->set_exponent(e);
   }

/*
* Compute the result
*/
BigInt Power_Mod::execute() const
   {
   if(!m_core)
      throw Internal_Error("Power_Mod::execute: m_core was NULL");
   return m_core->execute();
   }

/*
* Try to choose a good window size
*/
size_t Power_Mod::window_bits(size_t exp_bits, size_t,
                              Power_Mod::Usage_Hints hints)
   {
   static const size_t wsize[][2] = {
      { 1434, 7 },
      {  539, 6 },
      {  197, 4 },
      {   70, 3 },
      {   17, 2 },
      {    0, 0 }
   };

   size_t window_bits = 1;

   if(exp_bits)
      {
      for(size_t j = 0; wsize[j][0]; ++j)
         {
         if(exp_bits >= wsize[j][0])
            {
            window_bits += wsize[j][1];
            break;
            }
         }
      }

   if(hints & Power_Mod::BASE_IS_FIXED)
      window_bits += 2;
   if(hints & Power_Mod::EXP_IS_LARGE)
      ++window_bits;

   return window_bits;
   }

namespace {

/*
* Choose potentially useful hints
*/
Power_Mod::Usage_Hints choose_base_hints(const BigInt& b, const BigInt& n)
   {
   if(b == 2)
      return Power_Mod::Usage_Hints(Power_Mod::BASE_IS_2 |
                                    Power_Mod::BASE_IS_SMALL);

   const size_t b_bits = b.bits();
   const size_t n_bits = n.bits();

   if(b_bits < n_bits / 32)
      return Power_Mod::BASE_IS_SMALL;
   if(b_bits > n_bits / 4)
      return Power_Mod::BASE_IS_LARGE;

   return Power_Mod::NO_HINTS;
   }

/*
* Choose potentially useful hints
*/
Power_Mod::Usage_Hints choose_exp_hints(const BigInt& e, const BigInt& n)
   {
   const size_t e_bits = e.bits();
   const size_t n_bits = n.bits();

   if(e_bits < n_bits / 32)
      return Power_Mod::BASE_IS_SMALL;
   if(e_bits > n_bits / 4)
      return Power_Mod::BASE_IS_LARGE;
   return Power_Mod::NO_HINTS;
   }

}

/*
* Fixed_Exponent_Power_Mod Constructor
*/
Fixed_Exponent_Power_Mod::Fixed_Exponent_Power_Mod(const BigInt& e,
                                                   const BigInt& n,
                                                   Usage_Hints hints) :
   Power_Mod(n, Usage_Hints(hints | EXP_IS_FIXED | choose_exp_hints(e, n)))
   {
   set_exponent(e);
   }

/*
* Fixed_Base_Power_Mod Constructor
*/
Fixed_Base_Power_Mod::Fixed_Base_Power_Mod(const BigInt& b, const BigInt& n,
                                           Usage_Hints hints) :
   Power_Mod(n, Usage_Hints(hints | BASE_IS_FIXED | choose_base_hints(b, n)))
   {
   set_base(b);
   }

}
/*
* (C) 2016,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

bool is_lucas_probable_prime(const BigInt& C, const Modular_Reducer& mod_C)
   {
   if(C <= 1)
      return false;
   else if(C == 2)
      return true;
   else if(C.is_even())
      return false;
   else if(C == 3 || C == 5 || C == 7 || C == 11 || C == 13)
      return true;

   BigInt D = 5;

   for(;;)
      {
      int32_t j = jacobi(D, C);
      if(j == 0)
         return false;

      if(j == -1)
         break;

      // Check 5, -7, 9, -11, 13, -15, 17, ...
      if(D.is_negative())
         {
         D.flip_sign();
         D += 2;
         }
      else
         {
         D += 2;
         D.flip_sign();
         }

      if(D == 17 && is_perfect_square(C).is_nonzero())
         return false;
      }

   const BigInt K = C + 1;
   const size_t K_bits = K.bits() - 1;

   BigInt U = 1;
   BigInt V = 1;

   BigInt Ut, Vt, U2, V2;

   for(size_t i = 0; i != K_bits; ++i)
      {
      const bool k_bit = K.get_bit(K_bits - 1 - i);

      Ut = mod_C.multiply(U, V);

      Vt = mod_C.reduce(mod_C.square(V) + mod_C.multiply(D, mod_C.square(U)));
      Vt.ct_cond_add(Vt.is_odd(), C);
      Vt >>= 1;
      Vt = mod_C.reduce(Vt);

      U = Ut;
      V = Vt;

      U2 = mod_C.reduce(Ut + Vt);
      U2.ct_cond_add(U2.is_odd(), C);
      U2 >>= 1;

      V2 = mod_C.reduce(Vt + Ut*D);
      V2.ct_cond_add(V2.is_odd(), C);
      V2 >>= 1;

      U.ct_cond_assign(k_bit, U2);
      V.ct_cond_assign(k_bit, V2);
      }

   return (U == 0);
   }

bool is_bailie_psw_probable_prime(const BigInt& n, const Modular_Reducer& mod_n)
   {
   auto monty_n = std::make_shared<Montgomery_Params>(n, mod_n);
   return passes_miller_rabin_test(n, mod_n, monty_n, 2) && is_lucas_probable_prime(n, mod_n);
   }

bool is_bailie_psw_probable_prime(const BigInt& n)
   {
   Modular_Reducer mod_n(n);
   return is_bailie_psw_probable_prime(n, mod_n);
   }

bool passes_miller_rabin_test(const BigInt& n,
                              const Modular_Reducer& mod_n,
                              const std::shared_ptr<Montgomery_Params>& monty_n,
                              const BigInt& a)
   {
   BOTAN_ASSERT_NOMSG(n > 1);

   const BigInt n_minus_1 = n - 1;
   const size_t s = low_zero_bits(n_minus_1);
   const BigInt nm1_s = n_minus_1 >> s;
   const size_t n_bits = n.bits();

   const size_t powm_window = 4;

   auto powm_a_n = monty_precompute(monty_n, a, powm_window);

   BigInt y = monty_execute(*powm_a_n, nm1_s, n_bits);

   if(y == 1 || y == n_minus_1)
      return true;

   for(size_t i = 1; i != s; ++i)
      {
      y = mod_n.square(y);

      if(y == 1) // found a non-trivial square root
         return false;

      /*
      -1 is the trivial square root of unity, so ``a`` is not a
      witness for this number - give up
      */
      if(y == n_minus_1)
         return true;
      }

   return false;
   }

bool is_miller_rabin_probable_prime(const BigInt& n,
                                    const Modular_Reducer& mod_n,
                                    RandomNumberGenerator& rng,
                                    size_t test_iterations)
   {
   BOTAN_ASSERT_NOMSG(n > 1);

   auto monty_n = std::make_shared<Montgomery_Params>(n, mod_n);

   for(size_t i = 0; i != test_iterations; ++i)
      {
      const BigInt a = BigInt::random_integer(rng, 2, n);

      if(!passes_miller_rabin_test(n, mod_n, monty_n, a))
         return false;
      }

   // Failed to find a counterexample
   return true;
   }


size_t miller_rabin_test_iterations(size_t n_bits, size_t prob, bool random)
   {
   const size_t base = (prob + 2) / 2; // worst case 4^-t error rate

   /*
   * If the candidate prime was maliciously constructed, we can't rely
   * on arguments based on p being random.
   */
   if(random == false)
      return base;

   /*
   * For randomly chosen numbers we can use the estimates from
   * http://www.math.dartmouth.edu/~carlp/PDF/paper88.pdf
   *
   * These values are derived from the inequality for p(k,t) given on
   * the second page.
   */
   if(prob <= 128)
      {
      if(n_bits >= 1536)
         return 4; // < 2^-133
      if(n_bits >= 1024)
         return 6; // < 2^-133
      if(n_bits >= 512)
         return 12; // < 2^-129
      if(n_bits >= 256)
         return 29; // < 2^-128
      }

   /*
   If the user desires a smaller error probability than we have
   precomputed error estimates for, just fall back to using the worst
   case error rate.
   */
   return base;
   }

}
/*
* Small Primes Table
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

const uint16_t PRIMES[PRIME_TABLE_SIZE+1] = {
    3,     5,     7,    11,    13,    17,    19,    23,    29,    31,    37,
   41,    43,    47,    53,    59,    61,    67,    71,    73,    79,    83,
   89,    97,   101,   103,   107,   109,   113,   127,   131,   137,   139,
  149,   151,   157,   163,   167,   173,   179,   181,   191,   193,   197,
  199,   211,   223,   227,   229,   233,   239,   241,   251,   257,   263,
  269,   271,   277,   281,   283,   293,   307,   311,   313,   317,   331,
  337,   347,   349,   353,   359,   367,   373,   379,   383,   389,   397,
  401,   409,   419,   421,   431,   433,   439,   443,   449,   457,   461,
  463,   467,   479,   487,   491,   499,   503,   509,   521,   523,   541,
  547,   557,   563,   569,   571,   577,   587,   593,   599,   601,   607,
  613,   617,   619,   631,   641,   643,   647,   653,   659,   661,   673,
  677,   683,   691,   701,   709,   719,   727,   733,   739,   743,   751,
  757,   761,   769,   773,   787,   797,   809,   811,   821,   823,   827,
  829,   839,   853,   857,   859,   863,   877,   881,   883,   887,   907,
  911,   919,   929,   937,   941,   947,   953,   967,   971,   977,   983,
  991,   997,  1009,  1013,  1019,  1021,  1031,  1033,  1039,  1049,  1051,
 1061,  1063,  1069,  1087,  1091,  1093,  1097,  1103,  1109,  1117,  1123,
 1129,  1151,  1153,  1163,  1171,  1181,  1187,  1193,  1201,  1213,  1217,
 1223,  1229,  1231,  1237,  1249,  1259,  1277,  1279,  1283,  1289,  1291,
 1297,  1301,  1303,  1307,  1319,  1321,  1327,  1361,  1367,  1373,  1381,
 1399,  1409,  1423,  1427,  1429,  1433,  1439,  1447,  1451,  1453,  1459,
 1471,  1481,  1483,  1487,  1489,  1493,  1499,  1511,  1523,  1531,  1543,
 1549,  1553,  1559,  1567,  1571,  1579,  1583,  1597,  1601,  1607,  1609,
 1613,  1619,  1621,  1627,  1637,  1657,  1663,  1667,  1669,  1693,  1697,
 1699,  1709,  1721,  1723,  1733,  1741,  1747,  1753,  1759,  1777,  1783,
 1787,  1789,  1801,  1811,  1823,  1831,  1847,  1861,  1867,  1871,  1873,
 1877,  1879,  1889,  1901,  1907,  1913,  1931,  1933,  1949,  1951,  1973,
 1979,  1987,  1993,  1997,  1999,  2003,  2011,  2017,  2027,  2029,  2039,
 2053,  2063,  2069,  2081,  2083,  2087,  2089,  2099,  2111,  2113,  2129,
 2131,  2137,  2141,  2143,  2153,  2161,  2179,  2203,  2207,  2213,  2221,
 2237,  2239,  2243,  2251,  2267,  2269,  2273,  2281,  2287,  2293,  2297,
 2309,  2311,  2333,  2339,  2341,  2347,  2351,  2357,  2371,  2377,  2381,
 2383,  2389,  2393,  2399,  2411,  2417,  2423,  2437,  2441,  2447,  2459,
 2467,  2473,  2477,  2503,  2521,  2531,  2539,  2543,  2549,  2551,  2557,
 2579,  2591,  2593,  2609,  2617,  2621,  2633,  2647,  2657,  2659,  2663,
 2671,  2677,  2683,  2687,  2689,  2693,  2699,  2707,  2711,  2713,  2719,
 2729,  2731,  2741,  2749,  2753,  2767,  2777,  2789,  2791,  2797,  2801,
 2803,  2819,  2833,  2837,  2843,  2851,  2857,  2861,  2879,  2887,  2897,
 2903,  2909,  2917,  2927,  2939,  2953,  2957,  2963,  2969,  2971,  2999,
 3001,  3011,  3019,  3023,  3037,  3041,  3049,  3061,  3067,  3079,  3083,
 3089,  3109,  3119,  3121,  3137,  3163,  3167,  3169,  3181,  3187,  3191,
 3203,  3209,  3217,  3221,  3229,  3251,  3253,  3257,  3259,  3271,  3299,
 3301,  3307,  3313,  3319,  3323,  3329,  3331,  3343,  3347,  3359,  3361,
 3371,  3373,  3389,  3391,  3407,  3413,  3433,  3449,  3457,  3461,  3463,
 3467,  3469,  3491,  3499,  3511,  3517,  3527,  3529,  3533,  3539,  3541,
 3547,  3557,  3559,  3571,  3581,  3583,  3593,  3607,  3613,  3617,  3623,
 3631,  3637,  3643,  3659,  3671,  3673,  3677,  3691,  3697,  3701,  3709,
 3719,  3727,  3733,  3739,  3761,  3767,  3769,  3779,  3793,  3797,  3803,
 3821,  3823,  3833,  3847,  3851,  3853,  3863,  3877,  3881,  3889,  3907,
 3911,  3917,  3919,  3923,  3929,  3931,  3943,  3947,  3967,  3989,  4001,
 4003,  4007,  4013,  4019,  4021,  4027,  4049,  4051,  4057,  4073,  4079,
 4091,  4093,  4099,  4111,  4127,  4129,  4133,  4139,  4153,  4157,  4159,
 4177,  4201,  4211,  4217,  4219,  4229,  4231,  4241,  4243,  4253,  4259,
 4261,  4271,  4273,  4283,  4289,  4297,  4327,  4337,  4339,  4349,  4357,
 4363,  4373,  4391,  4397,  4409,  4421,  4423,  4441,  4447,  4451,  4457,
 4463,  4481,  4483,  4493,  4507,  4513,  4517,  4519,  4523,  4547,  4549,
 4561,  4567,  4583,  4591,  4597,  4603,  4621,  4637,  4639,  4643,  4649,
 4651,  4657,  4663,  4673,  4679,  4691,  4703,  4721,  4723,  4729,  4733,
 4751,  4759,  4783,  4787,  4789,  4793,  4799,  4801,  4813,  4817,  4831,
 4861,  4871,  4877,  4889,  4903,  4909,  4919,  4931,  4933,  4937,  4943,
 4951,  4957,  4967,  4969,  4973,  4987,  4993,  4999,  5003,  5009,  5011,
 5021,  5023,  5039,  5051,  5059,  5077,  5081,  5087,  5099,  5101,  5107,
 5113,  5119,  5147,  5153,  5167,  5171,  5179,  5189,  5197,  5209,  5227,
 5231,  5233,  5237,  5261,  5273,  5279,  5281,  5297,  5303,  5309,  5323,
 5333,  5347,  5351,  5381,  5387,  5393,  5399,  5407,  5413,  5417,  5419,
 5431,  5437,  5441,  5443,  5449,  5471,  5477,  5479,  5483,  5501,  5503,
 5507,  5519,  5521,  5527,  5531,  5557,  5563,  5569,  5573,  5581,  5591,
 5623,  5639,  5641,  5647,  5651,  5653,  5657,  5659,  5669,  5683,  5689,
 5693,  5701,  5711,  5717,  5737,  5741,  5743,  5749,  5779,  5783,  5791,
 5801,  5807,  5813,  5821,  5827,  5839,  5843,  5849,  5851,  5857,  5861,
 5867,  5869,  5879,  5881,  5897,  5903,  5923,  5927,  5939,  5953,  5981,
 5987,  6007,  6011,  6029,  6037,  6043,  6047,  6053,  6067,  6073,  6079,
 6089,  6091,  6101,  6113,  6121,  6131,  6133,  6143,  6151,  6163,  6173,
 6197,  6199,  6203,  6211,  6217,  6221,  6229,  6247,  6257,  6263,  6269,
 6271,  6277,  6287,  6299,  6301,  6311,  6317,  6323,  6329,  6337,  6343,
 6353,  6359,  6361,  6367,  6373,  6379,  6389,  6397,  6421,  6427,  6449,
 6451,  6469,  6473,  6481,  6491,  6521,  6529,  6547,  6551,  6553,  6563,
 6569,  6571,  6577,  6581,  6599,  6607,  6619,  6637,  6653,  6659,  6661,
 6673,  6679,  6689,  6691,  6701,  6703,  6709,  6719,  6733,  6737,  6761,
 6763,  6779,  6781,  6791,  6793,  6803,  6823,  6827,  6829,  6833,  6841,
 6857,  6863,  6869,  6871,  6883,  6899,  6907,  6911,  6917,  6947,  6949,
 6959,  6961,  6967,  6971,  6977,  6983,  6991,  6997,  7001,  7013,  7019,
 7027,  7039,  7043,  7057,  7069,  7079,  7103,  7109,  7121,  7127,  7129,
 7151,  7159,  7177,  7187,  7193,  7207,  7211,  7213,  7219,  7229,  7237,
 7243,  7247,  7253,  7283,  7297,  7307,  7309,  7321,  7331,  7333,  7349,
 7351,  7369,  7393,  7411,  7417,  7433,  7451,  7457,  7459,  7477,  7481,
 7487,  7489,  7499,  7507,  7517,  7523,  7529,  7537,  7541,  7547,  7549,
 7559,  7561,  7573,  7577,  7583,  7589,  7591,  7603,  7607,  7621,  7639,
 7643,  7649,  7669,  7673,  7681,  7687,  7691,  7699,  7703,  7717,  7723,
 7727,  7741,  7753,  7757,  7759,  7789,  7793,  7817,  7823,  7829,  7841,
 7853,  7867,  7873,  7877,  7879,  7883,  7901,  7907,  7919,  7927,  7933,
 7937,  7949,  7951,  7963,  7993,  8009,  8011,  8017,  8039,  8053,  8059,
 8069,  8081,  8087,  8089,  8093,  8101,  8111,  8117,  8123,  8147,  8161,
 8167,  8171,  8179,  8191,  8209,  8219,  8221,  8231,  8233,  8237,  8243,
 8263,  8269,  8273,  8287,  8291,  8293,  8297,  8311,  8317,  8329,  8353,
 8363,  8369,  8377,  8387,  8389,  8419,  8423,  8429,  8431,  8443,  8447,
 8461,  8467,  8501,  8513,  8521,  8527,  8537,  8539,  8543,  8563,  8573,
 8581,  8597,  8599,  8609,  8623,  8627,  8629,  8641,  8647,  8663,  8669,
 8677,  8681,  8689,  8693,  8699,  8707,  8713,  8719,  8731,  8737,  8741,
 8747,  8753,  8761,  8779,  8783,  8803,  8807,  8819,  8821,  8831,  8837,
 8839,  8849,  8861,  8863,  8867,  8887,  8893,  8923,  8929,  8933,  8941,
 8951,  8963,  8969,  8971,  8999,  9001,  9007,  9011,  9013,  9029,  9041,
 9043,  9049,  9059,  9067,  9091,  9103,  9109,  9127,  9133,  9137,  9151,
 9157,  9161,  9173,  9181,  9187,  9199,  9203,  9209,  9221,  9227,  9239,
 9241,  9257,  9277,  9281,  9283,  9293,  9311,  9319,  9323,  9337,  9341,
 9343,  9349,  9371,  9377,  9391,  9397,  9403,  9413,  9419,  9421,  9431,
 9433,  9437,  9439,  9461,  9463,  9467,  9473,  9479,  9491,  9497,  9511,
 9521,  9533,  9539,  9547,  9551,  9587,  9601,  9613,  9619,  9623,  9629,
 9631,  9643,  9649,  9661,  9677,  9679,  9689,  9697,  9719,  9721,  9733,
 9739,  9743,  9749,  9767,  9769,  9781,  9787,  9791,  9803,  9811,  9817,
 9829,  9833,  9839,  9851,  9857,  9859,  9871,  9883,  9887,  9901,  9907,
 9923,  9929,  9931,  9941,  9949,  9967,  9973, 10007, 10009, 10037, 10039,
10061, 10067, 10069, 10079, 10091, 10093, 10099, 10103, 10111, 10133, 10139,
10141, 10151, 10159, 10163, 10169, 10177, 10181, 10193, 10211, 10223, 10243,
10247, 10253, 10259, 10267, 10271, 10273, 10289, 10301, 10303, 10313, 10321,
10331, 10333, 10337, 10343, 10357, 10369, 10391, 10399, 10427, 10429, 10433,
10453, 10457, 10459, 10463, 10477, 10487, 10499, 10501, 10513, 10529, 10531,
10559, 10567, 10589, 10597, 10601, 10607, 10613, 10627, 10631, 10639, 10651,
10657, 10663, 10667, 10687, 10691, 10709, 10711, 10723, 10729, 10733, 10739,
10753, 10771, 10781, 10789, 10799, 10831, 10837, 10847, 10853, 10859, 10861,
10867, 10883, 10889, 10891, 10903, 10909, 10937, 10939, 10949, 10957, 10973,
10979, 10987, 10993, 11003, 11027, 11047, 11057, 11059, 11069, 11071, 11083,
11087, 11093, 11113, 11117, 11119, 11131, 11149, 11159, 11161, 11171, 11173,
11177, 11197, 11213, 11239, 11243, 11251, 11257, 11261, 11273, 11279, 11287,
11299, 11311, 11317, 11321, 11329, 11351, 11353, 11369, 11383, 11393, 11399,
11411, 11423, 11437, 11443, 11447, 11467, 11471, 11483, 11489, 11491, 11497,
11503, 11519, 11527, 11549, 11551, 11579, 11587, 11593, 11597, 11617, 11621,
11633, 11657, 11677, 11681, 11689, 11699, 11701, 11717, 11719, 11731, 11743,
11777, 11779, 11783, 11789, 11801, 11807, 11813, 11821, 11827, 11831, 11833,
11839, 11863, 11867, 11887, 11897, 11903, 11909, 11923, 11927, 11933, 11939,
11941, 11953, 11959, 11969, 11971, 11981, 11987, 12007, 12011, 12037, 12041,
12043, 12049, 12071, 12073, 12097, 12101, 12107, 12109, 12113, 12119, 12143,
12149, 12157, 12161, 12163, 12197, 12203, 12211, 12227, 12239, 12241, 12251,
12253, 12263, 12269, 12277, 12281, 12289, 12301, 12323, 12329, 12343, 12347,
12373, 12377, 12379, 12391, 12401, 12409, 12413, 12421, 12433, 12437, 12451,
12457, 12473, 12479, 12487, 12491, 12497, 12503, 12511, 12517, 12527, 12539,
12541, 12547, 12553, 12569, 12577, 12583, 12589, 12601, 12611, 12613, 12619,
12637, 12641, 12647, 12653, 12659, 12671, 12689, 12697, 12703, 12713, 12721,
12739, 12743, 12757, 12763, 12781, 12791, 12799, 12809, 12821, 12823, 12829,
12841, 12853, 12889, 12893, 12899, 12907, 12911, 12917, 12919, 12923, 12941,
12953, 12959, 12967, 12973, 12979, 12983, 13001, 13003, 13007, 13009, 13033,
13037, 13043, 13049, 13063, 13093, 13099, 13103, 13109, 13121, 13127, 13147,
13151, 13159, 13163, 13171, 13177, 13183, 13187, 13217, 13219, 13229, 13241,
13249, 13259, 13267, 13291, 13297, 13309, 13313, 13327, 13331, 13337, 13339,
13367, 13381, 13397, 13399, 13411, 13417, 13421, 13441, 13451, 13457, 13463,
13469, 13477, 13487, 13499, 13513, 13523, 13537, 13553, 13567, 13577, 13591,
13597, 13613, 13619, 13627, 13633, 13649, 13669, 13679, 13681, 13687, 13691,
13693, 13697, 13709, 13711, 13721, 13723, 13729, 13751, 13757, 13759, 13763,
13781, 13789, 13799, 13807, 13829, 13831, 13841, 13859, 13873, 13877, 13879,
13883, 13901, 13903, 13907, 13913, 13921, 13931, 13933, 13963, 13967, 13997,
13999, 14009, 14011, 14029, 14033, 14051, 14057, 14071, 14081, 14083, 14087,
14107, 14143, 14149, 14153, 14159, 14173, 14177, 14197, 14207, 14221, 14243,
14249, 14251, 14281, 14293, 14303, 14321, 14323, 14327, 14341, 14347, 14369,
14387, 14389, 14401, 14407, 14411, 14419, 14423, 14431, 14437, 14447, 14449,
14461, 14479, 14489, 14503, 14519, 14533, 14537, 14543, 14549, 14551, 14557,
14561, 14563, 14591, 14593, 14621, 14627, 14629, 14633, 14639, 14653, 14657,
14669, 14683, 14699, 14713, 14717, 14723, 14731, 14737, 14741, 14747, 14753,
14759, 14767, 14771, 14779, 14783, 14797, 14813, 14821, 14827, 14831, 14843,
14851, 14867, 14869, 14879, 14887, 14891, 14897, 14923, 14929, 14939, 14947,
14951, 14957, 14969, 14983, 15013, 15017, 15031, 15053, 15061, 15073, 15077,
15083, 15091, 15101, 15107, 15121, 15131, 15137, 15139, 15149, 15161, 15173,
15187, 15193, 15199, 15217, 15227, 15233, 15241, 15259, 15263, 15269, 15271,
15277, 15287, 15289, 15299, 15307, 15313, 15319, 15329, 15331, 15349, 15359,
15361, 15373, 15377, 15383, 15391, 15401, 15413, 15427, 15439, 15443, 15451,
15461, 15467, 15473, 15493, 15497, 15511, 15527, 15541, 15551, 15559, 15569,
15581, 15583, 15601, 15607, 15619, 15629, 15641, 15643, 15647, 15649, 15661,
15667, 15671, 15679, 15683, 15727, 15731, 15733, 15737, 15739, 15749, 15761,
15767, 15773, 15787, 15791, 15797, 15803, 15809, 15817, 15823, 15859, 15877,
15881, 15887, 15889, 15901, 15907, 15913, 15919, 15923, 15937, 15959, 15971,
15973, 15991, 16001, 16007, 16033, 16057, 16061, 16063, 16067, 16069, 16073,
16087, 16091, 16097, 16103, 16111, 16127, 16139, 16141, 16183, 16187, 16189,
16193, 16217, 16223, 16229, 16231, 16249, 16253, 16267, 16273, 16301, 16319,
16333, 16339, 16349, 16361, 16363, 16369, 16381, 16411, 16417, 16421, 16427,
16433, 16447, 16451, 16453, 16477, 16481, 16487, 16493, 16519, 16529, 16547,
16553, 16561, 16567, 16573, 16603, 16607, 16619, 16631, 16633, 16649, 16651,
16657, 16661, 16673, 16691, 16693, 16699, 16703, 16729, 16741, 16747, 16759,
16763, 16787, 16811, 16823, 16829, 16831, 16843, 16871, 16879, 16883, 16889,
16901, 16903, 16921, 16927, 16931, 16937, 16943, 16963, 16979, 16981, 16987,
16993, 17011, 17021, 17027, 17029, 17033, 17041, 17047, 17053, 17077, 17093,
17099, 17107, 17117, 17123, 17137, 17159, 17167, 17183, 17189, 17191, 17203,
17207, 17209, 17231, 17239, 17257, 17291, 17293, 17299, 17317, 17321, 17327,
17333, 17341, 17351, 17359, 17377, 17383, 17387, 17389, 17393, 17401, 17417,
17419, 17431, 17443, 17449, 17467, 17471, 17477, 17483, 17489, 17491, 17497,
17509, 17519, 17539, 17551, 17569, 17573, 17579, 17581, 17597, 17599, 17609,
17623, 17627, 17657, 17659, 17669, 17681, 17683, 17707, 17713, 17729, 17737,
17747, 17749, 17761, 17783, 17789, 17791, 17807, 17827, 17837, 17839, 17851,
17863, 17881, 17891, 17903, 17909, 17911, 17921, 17923, 17929, 17939, 17957,
17959, 17971, 17977, 17981, 17987, 17989, 18013, 18041, 18043, 18047, 18049,
18059, 18061, 18077, 18089, 18097, 18119, 18121, 18127, 18131, 18133, 18143,
18149, 18169, 18181, 18191, 18199, 18211, 18217, 18223, 18229, 18233, 18251,
18253, 18257, 18269, 18287, 18289, 18301, 18307, 18311, 18313, 18329, 18341,
18353, 18367, 18371, 18379, 18397, 18401, 18413, 18427, 18433, 18439, 18443,
18451, 18457, 18461, 18481, 18493, 18503, 18517, 18521, 18523, 18539, 18541,
18553, 18583, 18587, 18593, 18617, 18637, 18661, 18671, 18679, 18691, 18701,
18713, 18719, 18731, 18743, 18749, 18757, 18773, 18787, 18793, 18797, 18803,
18839, 18859, 18869, 18899, 18911, 18913, 18917, 18919, 18947, 18959, 18973,
18979, 19001, 19009, 19013, 19031, 19037, 19051, 19069, 19073, 19079, 19081,
19087, 19121, 19139, 19141, 19157, 19163, 19181, 19183, 19207, 19211, 19213,
19219, 19231, 19237, 19249, 19259, 19267, 19273, 19289, 19301, 19309, 19319,
19333, 19373, 19379, 19381, 19387, 19391, 19403, 19417, 19421, 19423, 19427,
19429, 19433, 19441, 19447, 19457, 19463, 19469, 19471, 19477, 19483, 19489,
19501, 19507, 19531, 19541, 19543, 19553, 19559, 19571, 19577, 19583, 19597,
19603, 19609, 19661, 19681, 19687, 19697, 19699, 19709, 19717, 19727, 19739,
19751, 19753, 19759, 19763, 19777, 19793, 19801, 19813, 19819, 19841, 19843,
19853, 19861, 19867, 19889, 19891, 19913, 19919, 19927, 19937, 19949, 19961,
19963, 19973, 19979, 19991, 19993, 19997, 20011, 20021, 20023, 20029, 20047,
20051, 20063, 20071, 20089, 20101, 20107, 20113, 20117, 20123, 20129, 20143,
20147, 20149, 20161, 20173, 20177, 20183, 20201, 20219, 20231, 20233, 20249,
20261, 20269, 20287, 20297, 20323, 20327, 20333, 20341, 20347, 20353, 20357,
20359, 20369, 20389, 20393, 20399, 20407, 20411, 20431, 20441, 20443, 20477,
20479, 20483, 20507, 20509, 20521, 20533, 20543, 20549, 20551, 20563, 20593,
20599, 20611, 20627, 20639, 20641, 20663, 20681, 20693, 20707, 20717, 20719,
20731, 20743, 20747, 20749, 20753, 20759, 20771, 20773, 20789, 20807, 20809,
20849, 20857, 20873, 20879, 20887, 20897, 20899, 20903, 20921, 20929, 20939,
20947, 20959, 20963, 20981, 20983, 21001, 21011, 21013, 21017, 21019, 21023,
21031, 21059, 21061, 21067, 21089, 21101, 21107, 21121, 21139, 21143, 21149,
21157, 21163, 21169, 21179, 21187, 21191, 21193, 21211, 21221, 21227, 21247,
21269, 21277, 21283, 21313, 21317, 21319, 21323, 21341, 21347, 21377, 21379,
21383, 21391, 21397, 21401, 21407, 21419, 21433, 21467, 21481, 21487, 21491,
21493, 21499, 21503, 21517, 21521, 21523, 21529, 21557, 21559, 21563, 21569,
21577, 21587, 21589, 21599, 21601, 21611, 21613, 21617, 21647, 21649, 21661,
21673, 21683, 21701, 21713, 21727, 21737, 21739, 21751, 21757, 21767, 21773,
21787, 21799, 21803, 21817, 21821, 21839, 21841, 21851, 21859, 21863, 21871,
21881, 21893, 21911, 21929, 21937, 21943, 21961, 21977, 21991, 21997, 22003,
22013, 22027, 22031, 22037, 22039, 22051, 22063, 22067, 22073, 22079, 22091,
22093, 22109, 22111, 22123, 22129, 22133, 22147, 22153, 22157, 22159, 22171,
22189, 22193, 22229, 22247, 22259, 22271, 22273, 22277, 22279, 22283, 22291,
22303, 22307, 22343, 22349, 22367, 22369, 22381, 22391, 22397, 22409, 22433,
22441, 22447, 22453, 22469, 22481, 22483, 22501, 22511, 22531, 22541, 22543,
22549, 22567, 22571, 22573, 22613, 22619, 22621, 22637, 22639, 22643, 22651,
22669, 22679, 22691, 22697, 22699, 22709, 22717, 22721, 22727, 22739, 22741,
22751, 22769, 22777, 22783, 22787, 22807, 22811, 22817, 22853, 22859, 22861,
22871, 22877, 22901, 22907, 22921, 22937, 22943, 22961, 22963, 22973, 22993,
23003, 23011, 23017, 23021, 23027, 23029, 23039, 23041, 23053, 23057, 23059,
23063, 23071, 23081, 23087, 23099, 23117, 23131, 23143, 23159, 23167, 23173,
23189, 23197, 23201, 23203, 23209, 23227, 23251, 23269, 23279, 23291, 23293,
23297, 23311, 23321, 23327, 23333, 23339, 23357, 23369, 23371, 23399, 23417,
23431, 23447, 23459, 23473, 23497, 23509, 23531, 23537, 23539, 23549, 23557,
23561, 23563, 23567, 23581, 23593, 23599, 23603, 23609, 23623, 23627, 23629,
23633, 23663, 23669, 23671, 23677, 23687, 23689, 23719, 23741, 23743, 23747,
23753, 23761, 23767, 23773, 23789, 23801, 23813, 23819, 23827, 23831, 23833,
23857, 23869, 23873, 23879, 23887, 23893, 23899, 23909, 23911, 23917, 23929,
23957, 23971, 23977, 23981, 23993, 24001, 24007, 24019, 24023, 24029, 24043,
24049, 24061, 24071, 24077, 24083, 24091, 24097, 24103, 24107, 24109, 24113,
24121, 24133, 24137, 24151, 24169, 24179, 24181, 24197, 24203, 24223, 24229,
24239, 24247, 24251, 24281, 24317, 24329, 24337, 24359, 24371, 24373, 24379,
24391, 24407, 24413, 24419, 24421, 24439, 24443, 24469, 24473, 24481, 24499,
24509, 24517, 24527, 24533, 24547, 24551, 24571, 24593, 24611, 24623, 24631,
24659, 24671, 24677, 24683, 24691, 24697, 24709, 24733, 24749, 24763, 24767,
24781, 24793, 24799, 24809, 24821, 24841, 24847, 24851, 24859, 24877, 24889,
24907, 24917, 24919, 24923, 24943, 24953, 24967, 24971, 24977, 24979, 24989,
25013, 25031, 25033, 25037, 25057, 25073, 25087, 25097, 25111, 25117, 25121,
25127, 25147, 25153, 25163, 25169, 25171, 25183, 25189, 25219, 25229, 25237,
25243, 25247, 25253, 25261, 25301, 25303, 25307, 25309, 25321, 25339, 25343,
25349, 25357, 25367, 25373, 25391, 25409, 25411, 25423, 25439, 25447, 25453,
25457, 25463, 25469, 25471, 25523, 25537, 25541, 25561, 25577, 25579, 25583,
25589, 25601, 25603, 25609, 25621, 25633, 25639, 25643, 25657, 25667, 25673,
25679, 25693, 25703, 25717, 25733, 25741, 25747, 25759, 25763, 25771, 25793,
25799, 25801, 25819, 25841, 25847, 25849, 25867, 25873, 25889, 25903, 25913,
25919, 25931, 25933, 25939, 25943, 25951, 25969, 25981, 25997, 25999, 26003,
26017, 26021, 26029, 26041, 26053, 26083, 26099, 26107, 26111, 26113, 26119,
26141, 26153, 26161, 26171, 26177, 26183, 26189, 26203, 26209, 26227, 26237,
26249, 26251, 26261, 26263, 26267, 26293, 26297, 26309, 26317, 26321, 26339,
26347, 26357, 26371, 26387, 26393, 26399, 26407, 26417, 26423, 26431, 26437,
26449, 26459, 26479, 26489, 26497, 26501, 26513, 26539, 26557, 26561, 26573,
26591, 26597, 26627, 26633, 26641, 26647, 26669, 26681, 26683, 26687, 26693,
26699, 26701, 26711, 26713, 26717, 26723, 26729, 26731, 26737, 26759, 26777,
26783, 26801, 26813, 26821, 26833, 26839, 26849, 26861, 26863, 26879, 26881,
26891, 26893, 26903, 26921, 26927, 26947, 26951, 26953, 26959, 26981, 26987,
26993, 27011, 27017, 27031, 27043, 27059, 27061, 27067, 27073, 27077, 27091,
27103, 27107, 27109, 27127, 27143, 27179, 27191, 27197, 27211, 27239, 27241,
27253, 27259, 27271, 27277, 27281, 27283, 27299, 27329, 27337, 27361, 27367,
27397, 27407, 27409, 27427, 27431, 27437, 27449, 27457, 27479, 27481, 27487,
27509, 27527, 27529, 27539, 27541, 27551, 27581, 27583, 27611, 27617, 27631,
27647, 27653, 27673, 27689, 27691, 27697, 27701, 27733, 27737, 27739, 27743,
27749, 27751, 27763, 27767, 27773, 27779, 27791, 27793, 27799, 27803, 27809,
27817, 27823, 27827, 27847, 27851, 27883, 27893, 27901, 27917, 27919, 27941,
27943, 27947, 27953, 27961, 27967, 27983, 27997, 28001, 28019, 28027, 28031,
28051, 28057, 28069, 28081, 28087, 28097, 28099, 28109, 28111, 28123, 28151,
28163, 28181, 28183, 28201, 28211, 28219, 28229, 28277, 28279, 28283, 28289,
28297, 28307, 28309, 28319, 28349, 28351, 28387, 28393, 28403, 28409, 28411,
28429, 28433, 28439, 28447, 28463, 28477, 28493, 28499, 28513, 28517, 28537,
28541, 28547, 28549, 28559, 28571, 28573, 28579, 28591, 28597, 28603, 28607,
28619, 28621, 28627, 28631, 28643, 28649, 28657, 28661, 28663, 28669, 28687,
28697, 28703, 28711, 28723, 28729, 28751, 28753, 28759, 28771, 28789, 28793,
28807, 28813, 28817, 28837, 28843, 28859, 28867, 28871, 28879, 28901, 28909,
28921, 28927, 28933, 28949, 28961, 28979, 29009, 29017, 29021, 29023, 29027,
29033, 29059, 29063, 29077, 29101, 29123, 29129, 29131, 29137, 29147, 29153,
29167, 29173, 29179, 29191, 29201, 29207, 29209, 29221, 29231, 29243, 29251,
29269, 29287, 29297, 29303, 29311, 29327, 29333, 29339, 29347, 29363, 29383,
29387, 29389, 29399, 29401, 29411, 29423, 29429, 29437, 29443, 29453, 29473,
29483, 29501, 29527, 29531, 29537, 29567, 29569, 29573, 29581, 29587, 29599,
29611, 29629, 29633, 29641, 29663, 29669, 29671, 29683, 29717, 29723, 29741,
29753, 29759, 29761, 29789, 29803, 29819, 29833, 29837, 29851, 29863, 29867,
29873, 29879, 29881, 29917, 29921, 29927, 29947, 29959, 29983, 29989, 30011,
30013, 30029, 30047, 30059, 30071, 30089, 30091, 30097, 30103, 30109, 30113,
30119, 30133, 30137, 30139, 30161, 30169, 30181, 30187, 30197, 30203, 30211,
30223, 30241, 30253, 30259, 30269, 30271, 30293, 30307, 30313, 30319, 30323,
30341, 30347, 30367, 30389, 30391, 30403, 30427, 30431, 30449, 30467, 30469,
30491, 30493, 30497, 30509, 30517, 30529, 30539, 30553, 30557, 30559, 30577,
30593, 30631, 30637, 30643, 30649, 30661, 30671, 30677, 30689, 30697, 30703,
30707, 30713, 30727, 30757, 30763, 30773, 30781, 30803, 30809, 30817, 30829,
30839, 30841, 30851, 30853, 30859, 30869, 30871, 30881, 30893, 30911, 30931,
30937, 30941, 30949, 30971, 30977, 30983, 31013, 31019, 31033, 31039, 31051,
31063, 31069, 31079, 31081, 31091, 31121, 31123, 31139, 31147, 31151, 31153,
31159, 31177, 31181, 31183, 31189, 31193, 31219, 31223, 31231, 31237, 31247,
31249, 31253, 31259, 31267, 31271, 31277, 31307, 31319, 31321, 31327, 31333,
31337, 31357, 31379, 31387, 31391, 31393, 31397, 31469, 31477, 31481, 31489,
31511, 31513, 31517, 31531, 31541, 31543, 31547, 31567, 31573, 31583, 31601,
31607, 31627, 31643, 31649, 31657, 31663, 31667, 31687, 31699, 31721, 31723,
31727, 31729, 31741, 31751, 31769, 31771, 31793, 31799, 31817, 31847, 31849,
31859, 31873, 31883, 31891, 31907, 31957, 31963, 31973, 31981, 31991, 32003,
32009, 32027, 32029, 32051, 32057, 32059, 32063, 32069, 32077, 32083, 32089,
32099, 32117, 32119, 32141, 32143, 32159, 32173, 32183, 32189, 32191, 32203,
32213, 32233, 32237, 32251, 32257, 32261, 32297, 32299, 32303, 32309, 32321,
32323, 32327, 32341, 32353, 32359, 32363, 32369, 32371, 32377, 32381, 32401,
32411, 32413, 32423, 32429, 32441, 32443, 32467, 32479, 32491, 32497, 32503,
32507, 32531, 32533, 32537, 32561, 32563, 32569, 32573, 32579, 32587, 32603,
32609, 32611, 32621, 32633, 32647, 32653, 32687, 32693, 32707, 32713, 32717,
32719, 32749, 32771, 32779, 32783, 32789, 32797, 32801, 32803, 32831, 32833,
32839, 32843, 32869, 32887, 32909, 32911, 32917, 32933, 32939, 32941, 32957,
32969, 32971, 32983, 32987, 32993, 32999, 33013, 33023, 33029, 33037, 33049,
33053, 33071, 33073, 33083, 33091, 33107, 33113, 33119, 33149, 33151, 33161,
33179, 33181, 33191, 33199, 33203, 33211, 33223, 33247, 33287, 33289, 33301,
33311, 33317, 33329, 33331, 33343, 33347, 33349, 33353, 33359, 33377, 33391,
33403, 33409, 33413, 33427, 33457, 33461, 33469, 33479, 33487, 33493, 33503,
33521, 33529, 33533, 33547, 33563, 33569, 33577, 33581, 33587, 33589, 33599,
33601, 33613, 33617, 33619, 33623, 33629, 33637, 33641, 33647, 33679, 33703,
33713, 33721, 33739, 33749, 33751, 33757, 33767, 33769, 33773, 33791, 33797,
33809, 33811, 33827, 33829, 33851, 33857, 33863, 33871, 33889, 33893, 33911,
33923, 33931, 33937, 33941, 33961, 33967, 33997, 34019, 34031, 34033, 34039,
34057, 34061, 34123, 34127, 34129, 34141, 34147, 34157, 34159, 34171, 34183,
34211, 34213, 34217, 34231, 34253, 34259, 34261, 34267, 34273, 34283, 34297,
34301, 34303, 34313, 34319, 34327, 34337, 34351, 34361, 34367, 34369, 34381,
34403, 34421, 34429, 34439, 34457, 34469, 34471, 34483, 34487, 34499, 34501,
34511, 34513, 34519, 34537, 34543, 34549, 34583, 34589, 34591, 34603, 34607,
34613, 34631, 34649, 34651, 34667, 34673, 34679, 34687, 34693, 34703, 34721,
34729, 34739, 34747, 34757, 34759, 34763, 34781, 34807, 34819, 34841, 34843,
34847, 34849, 34871, 34877, 34883, 34897, 34913, 34919, 34939, 34949, 34961,
34963, 34981, 35023, 35027, 35051, 35053, 35059, 35069, 35081, 35083, 35089,
35099, 35107, 35111, 35117, 35129, 35141, 35149, 35153, 35159, 35171, 35201,
35221, 35227, 35251, 35257, 35267, 35279, 35281, 35291, 35311, 35317, 35323,
35327, 35339, 35353, 35363, 35381, 35393, 35401, 35407, 35419, 35423, 35437,
35447, 35449, 35461, 35491, 35507, 35509, 35521, 35527, 35531, 35533, 35537,
35543, 35569, 35573, 35591, 35593, 35597, 35603, 35617, 35671, 35677, 35729,
35731, 35747, 35753, 35759, 35771, 35797, 35801, 35803, 35809, 35831, 35837,
35839, 35851, 35863, 35869, 35879, 35897, 35899, 35911, 35923, 35933, 35951,
35963, 35969, 35977, 35983, 35993, 35999, 36007, 36011, 36013, 36017, 36037,
36061, 36067, 36073, 36083, 36097, 36107, 36109, 36131, 36137, 36151, 36161,
36187, 36191, 36209, 36217, 36229, 36241, 36251, 36263, 36269, 36277, 36293,
36299, 36307, 36313, 36319, 36341, 36343, 36353, 36373, 36383, 36389, 36433,
36451, 36457, 36467, 36469, 36473, 36479, 36493, 36497, 36523, 36527, 36529,
36541, 36551, 36559, 36563, 36571, 36583, 36587, 36599, 36607, 36629, 36637,
36643, 36653, 36671, 36677, 36683, 36691, 36697, 36709, 36713, 36721, 36739,
36749, 36761, 36767, 36779, 36781, 36787, 36791, 36793, 36809, 36821, 36833,
36847, 36857, 36871, 36877, 36887, 36899, 36901, 36913, 36919, 36923, 36929,
36931, 36943, 36947, 36973, 36979, 36997, 37003, 37013, 37019, 37021, 37039,
37049, 37057, 37061, 37087, 37097, 37117, 37123, 37139, 37159, 37171, 37181,
37189, 37199, 37201, 37217, 37223, 37243, 37253, 37273, 37277, 37307, 37309,
37313, 37321, 37337, 37339, 37357, 37361, 37363, 37369, 37379, 37397, 37409,
37423, 37441, 37447, 37463, 37483, 37489, 37493, 37501, 37507, 37511, 37517,
37529, 37537, 37547, 37549, 37561, 37567, 37571, 37573, 37579, 37589, 37591,
37607, 37619, 37633, 37643, 37649, 37657, 37663, 37691, 37693, 37699, 37717,
37747, 37781, 37783, 37799, 37811, 37813, 37831, 37847, 37853, 37861, 37871,
37879, 37889, 37897, 37907, 37951, 37957, 37963, 37967, 37987, 37991, 37993,
37997, 38011, 38039, 38047, 38053, 38069, 38083, 38113, 38119, 38149, 38153,
38167, 38177, 38183, 38189, 38197, 38201, 38219, 38231, 38237, 38239, 38261,
38273, 38281, 38287, 38299, 38303, 38317, 38321, 38327, 38329, 38333, 38351,
38371, 38377, 38393, 38431, 38447, 38449, 38453, 38459, 38461, 38501, 38543,
38557, 38561, 38567, 38569, 38593, 38603, 38609, 38611, 38629, 38639, 38651,
38653, 38669, 38671, 38677, 38693, 38699, 38707, 38711, 38713, 38723, 38729,
38737, 38747, 38749, 38767, 38783, 38791, 38803, 38821, 38833, 38839, 38851,
38861, 38867, 38873, 38891, 38903, 38917, 38921, 38923, 38933, 38953, 38959,
38971, 38977, 38993, 39019, 39023, 39041, 39043, 39047, 39079, 39089, 39097,
39103, 39107, 39113, 39119, 39133, 39139, 39157, 39161, 39163, 39181, 39191,
39199, 39209, 39217, 39227, 39229, 39233, 39239, 39241, 39251, 39293, 39301,
39313, 39317, 39323, 39341, 39343, 39359, 39367, 39371, 39373, 39383, 39397,
39409, 39419, 39439, 39443, 39451, 39461, 39499, 39503, 39509, 39511, 39521,
39541, 39551, 39563, 39569, 39581, 39607, 39619, 39623, 39631, 39659, 39667,
39671, 39679, 39703, 39709, 39719, 39727, 39733, 39749, 39761, 39769, 39779,
39791, 39799, 39821, 39827, 39829, 39839, 39841, 39847, 39857, 39863, 39869,
39877, 39883, 39887, 39901, 39929, 39937, 39953, 39971, 39979, 39983, 39989,
40009, 40013, 40031, 40037, 40039, 40063, 40087, 40093, 40099, 40111, 40123,
40127, 40129, 40151, 40153, 40163, 40169, 40177, 40189, 40193, 40213, 40231,
40237, 40241, 40253, 40277, 40283, 40289, 40343, 40351, 40357, 40361, 40387,
40423, 40427, 40429, 40433, 40459, 40471, 40483, 40487, 40493, 40499, 40507,
40519, 40529, 40531, 40543, 40559, 40577, 40583, 40591, 40597, 40609, 40627,
40637, 40639, 40693, 40697, 40699, 40709, 40739, 40751, 40759, 40763, 40771,
40787, 40801, 40813, 40819, 40823, 40829, 40841, 40847, 40849, 40853, 40867,
40879, 40883, 40897, 40903, 40927, 40933, 40939, 40949, 40961, 40973, 40993,
41011, 41017, 41023, 41039, 41047, 41051, 41057, 41077, 41081, 41113, 41117,
41131, 41141, 41143, 41149, 41161, 41177, 41179, 41183, 41189, 41201, 41203,
41213, 41221, 41227, 41231, 41233, 41243, 41257, 41263, 41269, 41281, 41299,
41333, 41341, 41351, 41357, 41381, 41387, 41389, 41399, 41411, 41413, 41443,
41453, 41467, 41479, 41491, 41507, 41513, 41519, 41521, 41539, 41543, 41549,
41579, 41593, 41597, 41603, 41609, 41611, 41617, 41621, 41627, 41641, 41647,
41651, 41659, 41669, 41681, 41687, 41719, 41729, 41737, 41759, 41761, 41771,
41777, 41801, 41809, 41813, 41843, 41849, 41851, 41863, 41879, 41887, 41893,
41897, 41903, 41911, 41927, 41941, 41947, 41953, 41957, 41959, 41969, 41981,
41983, 41999, 42013, 42017, 42019, 42023, 42043, 42061, 42071, 42073, 42083,
42089, 42101, 42131, 42139, 42157, 42169, 42179, 42181, 42187, 42193, 42197,
42209, 42221, 42223, 42227, 42239, 42257, 42281, 42283, 42293, 42299, 42307,
42323, 42331, 42337, 42349, 42359, 42373, 42379, 42391, 42397, 42403, 42407,
42409, 42433, 42437, 42443, 42451, 42457, 42461, 42463, 42467, 42473, 42487,
42491, 42499, 42509, 42533, 42557, 42569, 42571, 42577, 42589, 42611, 42641,
42643, 42649, 42667, 42677, 42683, 42689, 42697, 42701, 42703, 42709, 42719,
42727, 42737, 42743, 42751, 42767, 42773, 42787, 42793, 42797, 42821, 42829,
42839, 42841, 42853, 42859, 42863, 42899, 42901, 42923, 42929, 42937, 42943,
42953, 42961, 42967, 42979, 42989, 43003, 43013, 43019, 43037, 43049, 43051,
43063, 43067, 43093, 43103, 43117, 43133, 43151, 43159, 43177, 43189, 43201,
43207, 43223, 43237, 43261, 43271, 43283, 43291, 43313, 43319, 43321, 43331,
43391, 43397, 43399, 43403, 43411, 43427, 43441, 43451, 43457, 43481, 43487,
43499, 43517, 43541, 43543, 43573, 43577, 43579, 43591, 43597, 43607, 43609,
43613, 43627, 43633, 43649, 43651, 43661, 43669, 43691, 43711, 43717, 43721,
43753, 43759, 43777, 43781, 43783, 43787, 43789, 43793, 43801, 43853, 43867,
43889, 43891, 43913, 43933, 43943, 43951, 43961, 43963, 43969, 43973, 43987,
43991, 43997, 44017, 44021, 44027, 44029, 44041, 44053, 44059, 44071, 44087,
44089, 44101, 44111, 44119, 44123, 44129, 44131, 44159, 44171, 44179, 44189,
44201, 44203, 44207, 44221, 44249, 44257, 44263, 44267, 44269, 44273, 44279,
44281, 44293, 44351, 44357, 44371, 44381, 44383, 44389, 44417, 44449, 44453,
44483, 44491, 44497, 44501, 44507, 44519, 44531, 44533, 44537, 44543, 44549,
44563, 44579, 44587, 44617, 44621, 44623, 44633, 44641, 44647, 44651, 44657,
44683, 44687, 44699, 44701, 44711, 44729, 44741, 44753, 44771, 44773, 44777,
44789, 44797, 44809, 44819, 44839, 44843, 44851, 44867, 44879, 44887, 44893,
44909, 44917, 44927, 44939, 44953, 44959, 44963, 44971, 44983, 44987, 45007,
45013, 45053, 45061, 45077, 45083, 45119, 45121, 45127, 45131, 45137, 45139,
45161, 45179, 45181, 45191, 45197, 45233, 45247, 45259, 45263, 45281, 45289,
45293, 45307, 45317, 45319, 45329, 45337, 45341, 45343, 45361, 45377, 45389,
45403, 45413, 45427, 45433, 45439, 45481, 45491, 45497, 45503, 45523, 45533,
45541, 45553, 45557, 45569, 45587, 45589, 45599, 45613, 45631, 45641, 45659,
45667, 45673, 45677, 45691, 45697, 45707, 45737, 45751, 45757, 45763, 45767,
45779, 45817, 45821, 45823, 45827, 45833, 45841, 45853, 45863, 45869, 45887,
45893, 45943, 45949, 45953, 45959, 45971, 45979, 45989, 46021, 46027, 46049,
46051, 46061, 46073, 46091, 46093, 46099, 46103, 46133, 46141, 46147, 46153,
46171, 46181, 46183, 46187, 46199, 46219, 46229, 46237, 46261, 46271, 46273,
46279, 46301, 46307, 46309, 46327, 46337, 46349, 46351, 46381, 46399, 46411,
46439, 46441, 46447, 46451, 46457, 46471, 46477, 46489, 46499, 46507, 46511,
46523, 46549, 46559, 46567, 46573, 46589, 46591, 46601, 46619, 46633, 46639,
46643, 46649, 46663, 46679, 46681, 46687, 46691, 46703, 46723, 46727, 46747,
46751, 46757, 46769, 46771, 46807, 46811, 46817, 46819, 46829, 46831, 46853,
46861, 46867, 46877, 46889, 46901, 46919, 46933, 46957, 46993, 46997, 47017,
47041, 47051, 47057, 47059, 47087, 47093, 47111, 47119, 47123, 47129, 47137,
47143, 47147, 47149, 47161, 47189, 47207, 47221, 47237, 47251, 47269, 47279,
47287, 47293, 47297, 47303, 47309, 47317, 47339, 47351, 47353, 47363, 47381,
47387, 47389, 47407, 47417, 47419, 47431, 47441, 47459, 47491, 47497, 47501,
47507, 47513, 47521, 47527, 47533, 47543, 47563, 47569, 47581, 47591, 47599,
47609, 47623, 47629, 47639, 47653, 47657, 47659, 47681, 47699, 47701, 47711,
47713, 47717, 47737, 47741, 47743, 47777, 47779, 47791, 47797, 47807, 47809,
47819, 47837, 47843, 47857, 47869, 47881, 47903, 47911, 47917, 47933, 47939,
47947, 47951, 47963, 47969, 47977, 47981, 48017, 48023, 48029, 48049, 48073,
48079, 48091, 48109, 48119, 48121, 48131, 48157, 48163, 48179, 48187, 48193,
48197, 48221, 48239, 48247, 48259, 48271, 48281, 48299, 48311, 48313, 48337,
48341, 48353, 48371, 48383, 48397, 48407, 48409, 48413, 48437, 48449, 48463,
48473, 48479, 48481, 48487, 48491, 48497, 48523, 48527, 48533, 48539, 48541,
48563, 48571, 48589, 48593, 48611, 48619, 48623, 48647, 48649, 48661, 48673,
48677, 48679, 48731, 48733, 48751, 48757, 48761, 48767, 48779, 48781, 48787,
48799, 48809, 48817, 48821, 48823, 48847, 48857, 48859, 48869, 48871, 48883,
48889, 48907, 48947, 48953, 48973, 48989, 48991, 49003, 49009, 49019, 49031,
49033, 49037, 49043, 49057, 49069, 49081, 49103, 49109, 49117, 49121, 49123,
49139, 49157, 49169, 49171, 49177, 49193, 49199, 49201, 49207, 49211, 49223,
49253, 49261, 49277, 49279, 49297, 49307, 49331, 49333, 49339, 49363, 49367,
49369, 49391, 49393, 49409, 49411, 49417, 49429, 49433, 49451, 49459, 49463,
49477, 49481, 49499, 49523, 49529, 49531, 49537, 49547, 49549, 49559, 49597,
49603, 49613, 49627, 49633, 49639, 49663, 49667, 49669, 49681, 49697, 49711,
49727, 49739, 49741, 49747, 49757, 49783, 49787, 49789, 49801, 49807, 49811,
49823, 49831, 49843, 49853, 49871, 49877, 49891, 49919, 49921, 49927, 49937,
49939, 49943, 49957, 49991, 49993, 49999, 50021, 50023, 50033, 50047, 50051,
50053, 50069, 50077, 50087, 50093, 50101, 50111, 50119, 50123, 50129, 50131,
50147, 50153, 50159, 50177, 50207, 50221, 50227, 50231, 50261, 50263, 50273,
50287, 50291, 50311, 50321, 50329, 50333, 50341, 50359, 50363, 50377, 50383,
50387, 50411, 50417, 50423, 50441, 50459, 50461, 50497, 50503, 50513, 50527,
50539, 50543, 50549, 50551, 50581, 50587, 50591, 50593, 50599, 50627, 50647,
50651, 50671, 50683, 50707, 50723, 50741, 50753, 50767, 50773, 50777, 50789,
50821, 50833, 50839, 50849, 50857, 50867, 50873, 50891, 50893, 50909, 50923,
50929, 50951, 50957, 50969, 50971, 50989, 50993, 51001, 51031, 51043, 51047,
51059, 51061, 51071, 51109, 51131, 51133, 51137, 51151, 51157, 51169, 51193,
51197, 51199, 51203, 51217, 51229, 51239, 51241, 51257, 51263, 51283, 51287,
51307, 51329, 51341, 51343, 51347, 51349, 51361, 51383, 51407, 51413, 51419,
51421, 51427, 51431, 51437, 51439, 51449, 51461, 51473, 51479, 51481, 51487,
51503, 51511, 51517, 51521, 51539, 51551, 51563, 51577, 51581, 51593, 51599,
51607, 51613, 51631, 51637, 51647, 51659, 51673, 51679, 51683, 51691, 51713,
51719, 51721, 51749, 51767, 51769, 51787, 51797, 51803, 51817, 51827, 51829,
51839, 51853, 51859, 51869, 51871, 51893, 51899, 51907, 51913, 51929, 51941,
51949, 51971, 51973, 51977, 51991, 52009, 52021, 52027, 52051, 52057, 52067,
52069, 52081, 52103, 52121, 52127, 52147, 52153, 52163, 52177, 52181, 52183,
52189, 52201, 52223, 52237, 52249, 52253, 52259, 52267, 52289, 52291, 52301,
52313, 52321, 52361, 52363, 52369, 52379, 52387, 52391, 52433, 52453, 52457,
52489, 52501, 52511, 52517, 52529, 52541, 52543, 52553, 52561, 52567, 52571,
52579, 52583, 52609, 52627, 52631, 52639, 52667, 52673, 52691, 52697, 52709,
52711, 52721, 52727, 52733, 52747, 52757, 52769, 52783, 52807, 52813, 52817,
52837, 52859, 52861, 52879, 52883, 52889, 52901, 52903, 52919, 52937, 52951,
52957, 52963, 52967, 52973, 52981, 52999, 53003, 53017, 53047, 53051, 53069,
53077, 53087, 53089, 53093, 53101, 53113, 53117, 53129, 53147, 53149, 53161,
53171, 53173, 53189, 53197, 53201, 53231, 53233, 53239, 53267, 53269, 53279,
53281, 53299, 53309, 53323, 53327, 53353, 53359, 53377, 53381, 53401, 53407,
53411, 53419, 53437, 53441, 53453, 53479, 53503, 53507, 53527, 53549, 53551,
53569, 53591, 53593, 53597, 53609, 53611, 53617, 53623, 53629, 53633, 53639,
53653, 53657, 53681, 53693, 53699, 53717, 53719, 53731, 53759, 53773, 53777,
53783, 53791, 53813, 53819, 53831, 53849, 53857, 53861, 53881, 53887, 53891,
53897, 53899, 53917, 53923, 53927, 53939, 53951, 53959, 53987, 53993, 54001,
54011, 54013, 54037, 54049, 54059, 54083, 54091, 54101, 54121, 54133, 54139,
54151, 54163, 54167, 54181, 54193, 54217, 54251, 54269, 54277, 54287, 54293,
54311, 54319, 54323, 54331, 54347, 54361, 54367, 54371, 54377, 54401, 54403,
54409, 54413, 54419, 54421, 54437, 54443, 54449, 54469, 54493, 54497, 54499,
54503, 54517, 54521, 54539, 54541, 54547, 54559, 54563, 54577, 54581, 54583,
54601, 54617, 54623, 54629, 54631, 54647, 54667, 54673, 54679, 54709, 54713,
54721, 54727, 54751, 54767, 54773, 54779, 54787, 54799, 54829, 54833, 54851,
54869, 54877, 54881, 54907, 54917, 54919, 54941, 54949, 54959, 54973, 54979,
54983, 55001, 55009, 55021, 55049, 55051, 55057, 55061, 55073, 55079, 55103,
55109, 55117, 55127, 55147, 55163, 55171, 55201, 55207, 55213, 55217, 55219,
55229, 55243, 55249, 55259, 55291, 55313, 55331, 55333, 55337, 55339, 55343,
55351, 55373, 55381, 55399, 55411, 55439, 55441, 55457, 55469, 55487, 55501,
55511, 55529, 55541, 55547, 55579, 55589, 55603, 55609, 55619, 55621, 55631,
55633, 55639, 55661, 55663, 55667, 55673, 55681, 55691, 55697, 55711, 55717,
55721, 55733, 55763, 55787, 55793, 55799, 55807, 55813, 55817, 55819, 55823,
55829, 55837, 55843, 55849, 55871, 55889, 55897, 55901, 55903, 55921, 55927,
55931, 55933, 55949, 55967, 55987, 55997, 56003, 56009, 56039, 56041, 56053,
56081, 56087, 56093, 56099, 56101, 56113, 56123, 56131, 56149, 56167, 56171,
56179, 56197, 56207, 56209, 56237, 56239, 56249, 56263, 56267, 56269, 56299,
56311, 56333, 56359, 56369, 56377, 56383, 56393, 56401, 56417, 56431, 56437,
56443, 56453, 56467, 56473, 56477, 56479, 56489, 56501, 56503, 56509, 56519,
56527, 56531, 56533, 56543, 56569, 56591, 56597, 56599, 56611, 56629, 56633,
56659, 56663, 56671, 56681, 56687, 56701, 56711, 56713, 56731, 56737, 56747,
56767, 56773, 56779, 56783, 56807, 56809, 56813, 56821, 56827, 56843, 56857,
56873, 56891, 56893, 56897, 56909, 56911, 56921, 56923, 56929, 56941, 56951,
56957, 56963, 56983, 56989, 56993, 56999, 57037, 57041, 57047, 57059, 57073,
57077, 57089, 57097, 57107, 57119, 57131, 57139, 57143, 57149, 57163, 57173,
57179, 57191, 57193, 57203, 57221, 57223, 57241, 57251, 57259, 57269, 57271,
57283, 57287, 57301, 57329, 57331, 57347, 57349, 57367, 57373, 57383, 57389,
57397, 57413, 57427, 57457, 57467, 57487, 57493, 57503, 57527, 57529, 57557,
57559, 57571, 57587, 57593, 57601, 57637, 57641, 57649, 57653, 57667, 57679,
57689, 57697, 57709, 57713, 57719, 57727, 57731, 57737, 57751, 57773, 57781,
57787, 57791, 57793, 57803, 57809, 57829, 57839, 57847, 57853, 57859, 57881,
57899, 57901, 57917, 57923, 57943, 57947, 57973, 57977, 57991, 58013, 58027,
58031, 58043, 58049, 58057, 58061, 58067, 58073, 58099, 58109, 58111, 58129,
58147, 58151, 58153, 58169, 58171, 58189, 58193, 58199, 58207, 58211, 58217,
58229, 58231, 58237, 58243, 58271, 58309, 58313, 58321, 58337, 58363, 58367,
58369, 58379, 58391, 58393, 58403, 58411, 58417, 58427, 58439, 58441, 58451,
58453, 58477, 58481, 58511, 58537, 58543, 58549, 58567, 58573, 58579, 58601,
58603, 58613, 58631, 58657, 58661, 58679, 58687, 58693, 58699, 58711, 58727,
58733, 58741, 58757, 58763, 58771, 58787, 58789, 58831, 58889, 58897, 58901,
58907, 58909, 58913, 58921, 58937, 58943, 58963, 58967, 58979, 58991, 58997,
59009, 59011, 59021, 59023, 59029, 59051, 59053, 59063, 59069, 59077, 59083,
59093, 59107, 59113, 59119, 59123, 59141, 59149, 59159, 59167, 59183, 59197,
59207, 59209, 59219, 59221, 59233, 59239, 59243, 59263, 59273, 59281, 59333,
59341, 59351, 59357, 59359, 59369, 59377, 59387, 59393, 59399, 59407, 59417,
59419, 59441, 59443, 59447, 59453, 59467, 59471, 59473, 59497, 59509, 59513,
59539, 59557, 59561, 59567, 59581, 59611, 59617, 59621, 59627, 59629, 59651,
59659, 59663, 59669, 59671, 59693, 59699, 59707, 59723, 59729, 59743, 59747,
59753, 59771, 59779, 59791, 59797, 59809, 59833, 59863, 59879, 59887, 59921,
59929, 59951, 59957, 59971, 59981, 59999, 60013, 60017, 60029, 60037, 60041,
60077, 60083, 60089, 60091, 60101, 60103, 60107, 60127, 60133, 60139, 60149,
60161, 60167, 60169, 60209, 60217, 60223, 60251, 60257, 60259, 60271, 60289,
60293, 60317, 60331, 60337, 60343, 60353, 60373, 60383, 60397, 60413, 60427,
60443, 60449, 60457, 60493, 60497, 60509, 60521, 60527, 60539, 60589, 60601,
60607, 60611, 60617, 60623, 60631, 60637, 60647, 60649, 60659, 60661, 60679,
60689, 60703, 60719, 60727, 60733, 60737, 60757, 60761, 60763, 60773, 60779,
60793, 60811, 60821, 60859, 60869, 60887, 60889, 60899, 60901, 60913, 60917,
60919, 60923, 60937, 60943, 60953, 60961, 61001, 61007, 61027, 61031, 61043,
61051, 61057, 61091, 61099, 61121, 61129, 61141, 61151, 61153, 61169, 61211,
61223, 61231, 61253, 61261, 61283, 61291, 61297, 61331, 61333, 61339, 61343,
61357, 61363, 61379, 61381, 61403, 61409, 61417, 61441, 61463, 61469, 61471,
61483, 61487, 61493, 61507, 61511, 61519, 61543, 61547, 61553, 61559, 61561,
61583, 61603, 61609, 61613, 61627, 61631, 61637, 61643, 61651, 61657, 61667,
61673, 61681, 61687, 61703, 61717, 61723, 61729, 61751, 61757, 61781, 61813,
61819, 61837, 61843, 61861, 61871, 61879, 61909, 61927, 61933, 61949, 61961,
61967, 61979, 61981, 61987, 61991, 62003, 62011, 62017, 62039, 62047, 62053,
62057, 62071, 62081, 62099, 62119, 62129, 62131, 62137, 62141, 62143, 62171,
62189, 62191, 62201, 62207, 62213, 62219, 62233, 62273, 62297, 62299, 62303,
62311, 62323, 62327, 62347, 62351, 62383, 62401, 62417, 62423, 62459, 62467,
62473, 62477, 62483, 62497, 62501, 62507, 62533, 62539, 62549, 62563, 62581,
62591, 62597, 62603, 62617, 62627, 62633, 62639, 62653, 62659, 62683, 62687,
62701, 62723, 62731, 62743, 62753, 62761, 62773, 62791, 62801, 62819, 62827,
62851, 62861, 62869, 62873, 62897, 62903, 62921, 62927, 62929, 62939, 62969,
62971, 62981, 62983, 62987, 62989, 63029, 63031, 63059, 63067, 63073, 63079,
63097, 63103, 63113, 63127, 63131, 63149, 63179, 63197, 63199, 63211, 63241,
63247, 63277, 63281, 63299, 63311, 63313, 63317, 63331, 63337, 63347, 63353,
63361, 63367, 63377, 63389, 63391, 63397, 63409, 63419, 63421, 63439, 63443,
63463, 63467, 63473, 63487, 63493, 63499, 63521, 63527, 63533, 63541, 63559,
63577, 63587, 63589, 63599, 63601, 63607, 63611, 63617, 63629, 63647, 63649,
63659, 63667, 63671, 63689, 63691, 63697, 63703, 63709, 63719, 63727, 63737,
63743, 63761, 63773, 63781, 63793, 63799, 63803, 63809, 63823, 63839, 63841,
63853, 63857, 63863, 63901, 63907, 63913, 63929, 63949, 63977, 63997, 64007,
64013, 64019, 64033, 64037, 64063, 64067, 64081, 64091, 64109, 64123, 64151,
64153, 64157, 64171, 64187, 64189, 64217, 64223, 64231, 64237, 64271, 64279,
64283, 64301, 64303, 64319, 64327, 64333, 64373, 64381, 64399, 64403, 64433,
64439, 64451, 64453, 64483, 64489, 64499, 64513, 64553, 64567, 64577, 64579,
64591, 64601, 64609, 64613, 64621, 64627, 64633, 64661, 64663, 64667, 64679,
64693, 64709, 64717, 64747, 64763, 64781, 64783, 64793, 64811, 64817, 64849,
64853, 64871, 64877, 64879, 64891, 64901, 64919, 64921, 64927, 64937, 64951,
64969, 64997, 65003, 65011, 65027, 65029, 65033, 65053, 65063, 65071, 65089,
65099, 65101, 65111, 65119, 65123, 65129, 65141, 65147, 65167, 65171, 65173,
65179, 65183, 65203, 65213, 65239, 65257, 65267, 65269, 65287, 65293, 65309,
65323, 65327, 65353, 65357, 65371, 65381, 65393, 65407, 65413, 65419, 65423,
65437, 65447, 65449, 65479, 65497, 65519, 65521, 0 };

}
/*
* Modular Reducer
* (C) 1999-2011,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/*
* Modular_Reducer Constructor
*/
Modular_Reducer::Modular_Reducer(const BigInt& mod)
   {
   if(mod < 0)
      throw Invalid_Argument("Modular_Reducer: modulus must be positive");

   // Left uninitialized if mod == 0
   m_mod_words = 0;

   if(mod > 0)
      {
      m_modulus = mod;
      m_mod_words = m_modulus.sig_words();

      // Compute mu = floor(2^{2k} / m)
      m_mu.set_bit(2 * BOTAN_MP_WORD_BITS * m_mod_words);
      m_mu = ct_divide(m_mu, m_modulus);
      }
   }

BigInt Modular_Reducer::reduce(const BigInt& x) const
   {
   BigInt r;
   secure_vector<word> ws;
   reduce(r, x, ws);
   return r;
   }

namespace {

/*
* Like if(cnd) x.rev_sub(...) but in const time
*/
void cnd_rev_sub(bool cnd, BigInt& x, const word y[], size_t y_sw, secure_vector<word>& ws)
   {
   if(x.sign() != BigInt::Positive)
      throw Invalid_State("BigInt::sub_rev requires this is positive");

   const size_t x_sw = x.sig_words();

   const size_t max_words = std::max(x_sw, y_sw);
   ws.resize(std::max(x_sw, y_sw));
   clear_mem(ws.data(), ws.size());
   x.grow_to(max_words);

   const int32_t relative_size = bigint_sub_abs(ws.data(), x.data(), x_sw, y, y_sw);

   x.cond_flip_sign((relative_size > 0) && cnd);
   bigint_cnd_swap(cnd, x.mutable_data(), ws.data(), max_words);
   }

}

void Modular_Reducer::reduce(BigInt& t1, const BigInt& x, secure_vector<word>& ws) const
   {
   if(&t1 == &x)
      throw Invalid_State("Modular_Reducer arguments cannot alias");
   if(m_mod_words == 0)
      throw Invalid_State("Modular_Reducer: Never initalized");

   const size_t x_sw = x.sig_words();

   if(x_sw > 2*m_mod_words)
      {
      // too big, fall back to slow boat division
      t1 = ct_modulo(x, m_modulus);
      return;
      }

   t1 = x;
   t1.set_sign(BigInt::Positive);
   t1 >>= (BOTAN_MP_WORD_BITS * (m_mod_words - 1));

   t1.mul(m_mu, ws);
   t1 >>= (BOTAN_MP_WORD_BITS * (m_mod_words + 1));

   // TODO add masked mul to avoid computing high bits
   t1.mul(m_modulus, ws);
   t1.mask_bits(BOTAN_MP_WORD_BITS * (m_mod_words + 1));

   t1.rev_sub(x.data(), std::min(x_sw, m_mod_words + 1), ws);

   /*
   * If t1 < 0 then we must add b^(k+1) where b = 2^w. To avoid a
   * side channel perform the addition unconditionally, with ws set
   * to either b^(k+1) or else 0.
   */
   const word t1_neg = t1.is_negative();

   if(ws.size() < m_mod_words + 2)
      ws.resize(m_mod_words + 2);
   clear_mem(ws.data(), ws.size());
   ws[m_mod_words + 1] = t1_neg;

   t1.add(ws.data(), m_mod_words + 2, BigInt::Positive);

   // Per HAC this step requires at most 2 subtractions
   t1.ct_reduce_below(m_modulus, ws, 2);

   cnd_rev_sub(t1.is_nonzero() && x.is_negative(), t1, m_modulus.data(), m_modulus.size(), ws);
   }

}
/*
* (C) 2007,2008 Falko Strenzke, FlexSecure GmbH
* (C) 2008 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/*
* Tonelli-Shanks algorithm
*/
BigInt ressol(const BigInt& a, const BigInt& p)
   {
   if(p <= 1 || p.is_even())
      throw Invalid_Argument("ressol: invalid prime");

   if(a == 0)
      return 0;
   else if(a < 0)
      throw Invalid_Argument("ressol: value to solve for must be positive");
   else if(a >= p)
      throw Invalid_Argument("ressol: value to solve for must be less than p");

   if(p == 2)
      return a;

   if(jacobi(a, p) != 1) // not a quadratic residue
      return -BigInt(1);

   if(p % 4 == 3) // The easy case
      {
      return power_mod(a, ((p+1) >> 2), p);
      }

   size_t s = low_zero_bits(p - 1);
   BigInt q = p >> s;

   q -= 1;
   q >>= 1;

   Modular_Reducer mod_p(p);

   BigInt r = power_mod(a, q, p);
   BigInt n = mod_p.multiply(a, mod_p.square(r));
   r = mod_p.multiply(r, a);

   if(n == 1)
      return r;

   // find random quadratic nonresidue z
   word z = 2;
   for(;;)
      {
      if(jacobi(z, p) == -1) // found one
         break;

      z += 1; // try next z

      /*
      * The expected number of tests to find a non-residue modulo a
      * prime is 2. If we have not found one after 256 then almost
      * certainly we have been given a non-prime p.
      */
      if(z >= 256)
         return -BigInt(1);
      }

   BigInt c = power_mod(z, (q << 1) + 1, p);

   while(n > 1)
      {
      q = n;

      size_t i = 0;
      while(q != 1)
         {
         q = mod_p.square(q);
         ++i;

         if(i >= s)
            {
            return -BigInt(1);
            }
         }

      c = power_mod(c, BigInt::power_of_2(s-i-1), p);
      r = mod_p.multiply(r, c);
      c = mod_p.square(c);
      n = mod_p.multiply(n, c);
      s = i;
      }

   return r;
   }

}
/*
* PEM Encoding/Decoding
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace PEM_Code {

namespace {

std::string linewrap(size_t width, const std::string& in)
   {
   std::string out;
   for(size_t i = 0; i != in.size(); ++i)
      {
      if(i > 0 && i % width == 0)
         {
         out.push_back('\n');
         }
      out.push_back(in[i]);
      }
   if(out.size() > 0 && out[out.size()-1] != '\n')
      {
      out.push_back('\n');
      }

   return out;
   }

}

/*
* PEM encode BER/DER-encoded objects
*/
std::string encode(const uint8_t der[], size_t length, const std::string& label, size_t width)
   {
   const std::string PEM_HEADER = "-----BEGIN " + label + "-----\n";
   const std::string PEM_TRAILER = "-----END " + label + "-----\n";

   return (PEM_HEADER + linewrap(width, base64_encode(der, length)) + PEM_TRAILER);
   }

/*
* Decode PEM down to raw BER/DER
*/
secure_vector<uint8_t> decode_check_label(DataSource& source,
                                          const std::string& label_want)
   {
   std::string label_got;
   secure_vector<uint8_t> ber = decode(source, label_got);
   if(label_got != label_want)
      throw Decoding_Error("PEM: Label mismatch, wanted " + label_want +
                           ", got " + label_got);
   return ber;
   }

/*
* Decode PEM down to raw BER/DER
*/
secure_vector<uint8_t> decode(DataSource& source, std::string& label)
   {
   const size_t RANDOM_CHAR_LIMIT = 8;

   label.clear();

   const std::string PEM_HEADER1 = "-----BEGIN ";
   const std::string PEM_HEADER2 = "-----";
   size_t position = 0;

   while(position != PEM_HEADER1.length())
      {
      uint8_t b;
      if(!source.read_byte(b))
         throw Decoding_Error("PEM: No PEM header found");
      if(b == PEM_HEADER1[position])
         ++position;
      else if(position >= RANDOM_CHAR_LIMIT)
         throw Decoding_Error("PEM: Malformed PEM header");
      else
         position = 0;
      }
   position = 0;
   while(position != PEM_HEADER2.length())
      {
      uint8_t b;
      if(!source.read_byte(b))
         throw Decoding_Error("PEM: No PEM header found");
      if(b == PEM_HEADER2[position])
         ++position;
      else if(position)
         throw Decoding_Error("PEM: Malformed PEM header");

      if(position == 0)
         label += static_cast<char>(b);
      }

   std::vector<char> b64;

   const std::string PEM_TRAILER = "-----END " + label + "-----";
   position = 0;
   while(position != PEM_TRAILER.length())
      {
      uint8_t b;
      if(!source.read_byte(b))
         throw Decoding_Error("PEM: No PEM trailer found");
      if(b == PEM_TRAILER[position])
         ++position;
      else if(position)
         throw Decoding_Error("PEM: Malformed PEM trailer");

      if(position == 0)
         b64.push_back(b);
      }

   return base64_decode(b64.data(), b64.size());
   }

secure_vector<uint8_t> decode_check_label(const std::string& pem,
                                          const std::string& label_want)
   {
   DataSource_Memory src(pem);
   return decode_check_label(src, label_want);
   }

secure_vector<uint8_t> decode(const std::string& pem, std::string& label)
   {
   DataSource_Memory src(pem);
   return decode(src, label);
   }

/*
* Search for a PEM signature
*/
bool matches(DataSource& source, const std::string& extra,
             size_t search_range)
   {
   const std::string PEM_HEADER = "-----BEGIN " + extra;

   secure_vector<uint8_t> search_buf(search_range);
   size_t got = source.peek(search_buf.data(), search_buf.size(), 0);

   if(got < PEM_HEADER.length())
      return false;

   size_t index = 0;

   for(size_t j = 0; j != got; ++j)
      {
      if(search_buf[j] == PEM_HEADER[index])
         ++index;
      else
         index = 0;
      if(index == PEM_HEADER.size())
         return true;
      }
   return false;
   }

}

}
/*
* EME Base Class
* (C) 1999-2008 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_EME_OAEP)
#endif

#if defined(BOTAN_HAS_EME_PKCS1)
#endif

#if defined(BOTAN_HAS_EME_RAW)
#endif

namespace Botan {

EME* get_eme(const std::string& algo_spec)
   {
#if defined(BOTAN_HAS_EME_RAW)
   if(algo_spec == "Raw")
      return new EME_Raw;
#endif

#if defined(BOTAN_HAS_EME_PKCS1)
   if(algo_spec == "PKCS1v15" || algo_spec == "EME-PKCS1-v1_5")
      return new EME_PKCS1v15;
#endif

#if defined(BOTAN_HAS_EME_OAEP)
   SCAN_Name req(algo_spec);

   if(req.algo_name() == "OAEP" ||
      req.algo_name() == "EME-OAEP" ||
      req.algo_name() == "EME1")
      {
      if(req.arg_count() == 1 ||
         ((req.arg_count() == 2 || req.arg_count() == 3) && req.arg(1) == "MGF1"))
         {
         if(auto hash = HashFunction::create(req.arg(0)))
            return new OAEP(hash.release(), req.arg(2, ""));
         }
      else if(req.arg_count() == 2 || req.arg_count() == 3)
         {
         auto mgf_params = parse_algorithm_name(req.arg(1));

         if(mgf_params.size() == 2 && mgf_params[0] == "MGF1")
            {
            auto hash = HashFunction::create(req.arg(0));
            auto mgf1_hash = HashFunction::create(mgf_params[1]);

            if(hash && mgf1_hash)
               {
               return new OAEP(hash.release(), mgf1_hash.release(), req.arg(2, ""));
               }
            }
         }
      }
#endif

   throw Algorithm_Not_Found(algo_spec);
   }

/*
* Encode a message
*/
secure_vector<uint8_t> EME::encode(const uint8_t msg[], size_t msg_len,
                                size_t key_bits,
                                RandomNumberGenerator& rng) const
   {
   return pad(msg, msg_len, key_bits, rng);
   }

/*
* Encode a message
*/
secure_vector<uint8_t> EME::encode(const secure_vector<uint8_t>& msg,
                                size_t key_bits,
                                RandomNumberGenerator& rng) const
   {
   return pad(msg.data(), msg.size(), key_bits, rng);
   }


}
/*
* (C) 2015 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_EMSA1)
#endif

#if defined(BOTAN_HAS_EMSA_X931)
#endif

#if defined(BOTAN_HAS_EMSA_PKCS1)
#endif

#if defined(BOTAN_HAS_EMSA_PSSR)
#endif

#if defined(BOTAN_HAS_EMSA_RAW)
#endif

#if defined(BOTAN_HAS_ISO_9796)
#endif

namespace Botan {

AlgorithmIdentifier EMSA::config_for_x509(const Private_Key&,
                                          const std::string&) const
   {
   throw Not_Implemented("Encoding " + name() + " not supported for signing X509 objects");
   }

EMSA* get_emsa(const std::string& algo_spec)
   {
   SCAN_Name req(algo_spec);

#if defined(BOTAN_HAS_EMSA1)
   if(req.algo_name() == "EMSA1" && req.arg_count() == 1)
      {
      if(auto hash = HashFunction::create(req.arg(0)))
         return new EMSA1(hash.release());
      }
#endif

#if defined(BOTAN_HAS_EMSA_PKCS1)
   if(req.algo_name() == "EMSA_PKCS1" ||
      req.algo_name() == "PKCS1v15" ||
      req.algo_name() == "EMSA-PKCS1-v1_5" ||
      req.algo_name() == "EMSA3")
      {
      if(req.arg_count() == 2 && req.arg(0) == "Raw")
         {
         return new EMSA_PKCS1v15_Raw(req.arg(1));
         }
      else if(req.arg_count() == 1)
         {
         if(req.arg(0) == "Raw")
            {
            return new EMSA_PKCS1v15_Raw;
            }
         else
            {
            if(auto hash = HashFunction::create(req.arg(0)))
               {
               return new EMSA_PKCS1v15(hash.release());
               }
            }
         }
      }
#endif

#if defined(BOTAN_HAS_EMSA_PSSR)
   if(req.algo_name() == "PSS_Raw" ||
      req.algo_name() == "PSSR_Raw")
      {
      if(req.arg_count_between(1, 3) && req.arg(1, "MGF1") == "MGF1")
         {
         if(auto h = HashFunction::create(req.arg(0)))
            {
            if(req.arg_count() == 3)
               {
               const size_t salt_size = req.arg_as_integer(2, 0);
               return new PSSR_Raw(h.release(), salt_size);
               }
            else
               {
               return new PSSR_Raw(h.release());
               }
            }
         }
      }

   if(req.algo_name() == "PSS" ||
      req.algo_name() == "PSSR" ||
      req.algo_name() == "EMSA-PSS" ||
      req.algo_name() == "PSS-MGF1" ||
      req.algo_name() == "EMSA4")
      {
      if(req.arg_count_between(1, 3) && req.arg(1, "MGF1") == "MGF1")
         {
         if(auto h = HashFunction::create(req.arg(0)))
            {
            if(req.arg_count() == 3)
               {
               const size_t salt_size = req.arg_as_integer(2, 0);
               return new PSSR(h.release(), salt_size);
               }
            else
               {
               return new PSSR(h.release());
               }
            }
         }
      }
#endif

#if defined(BOTAN_HAS_ISO_9796)
   if(req.algo_name() == "ISO_9796_DS2")
      {
      if(req.arg_count_between(1, 3))
         {
         if(auto h = HashFunction::create(req.arg(0)))
            {
            const size_t salt_size = req.arg_as_integer(2, h->output_length());
            const bool implicit = req.arg(1, "exp") == "imp";
            return new ISO_9796_DS2(h.release(), implicit, salt_size);
            }
         }
      }
   //ISO-9796-2 DS 3 is deterministic and DS2 without a salt
   if(req.algo_name() == "ISO_9796_DS3")
      {
      if(req.arg_count_between(1, 2))
         {
         if(auto h = HashFunction::create(req.arg(0)))
            {
            const bool implicit = req.arg(1, "exp") == "imp";
            return new ISO_9796_DS3(h.release(), implicit);
            }
         }
      }
#endif

#if defined(BOTAN_HAS_EMSA_X931)
   if(req.algo_name() == "EMSA_X931" ||
         req.algo_name() == "EMSA2" ||
         req.algo_name() == "X9.31")
      {
      if(req.arg_count() == 1)
         {
         if(auto hash = HashFunction::create(req.arg(0)))
            {
            return new EMSA_X931(hash.release());
            }
         }
      }
#endif

#if defined(BOTAN_HAS_EMSA_RAW)
   if(req.algo_name() == "Raw")
      {
      if(req.arg_count() == 0)
         {
         return new EMSA_Raw;
         }
      else
         {
         auto hash = HashFunction::create(req.arg(0));
         if(hash)
            return new EMSA_Raw(hash->output_length());
         }
      }
#endif

   throw Algorithm_Not_Found(algo_spec);
   }

std::string hash_for_emsa(const std::string& algo_spec)
   {
   SCAN_Name emsa_name(algo_spec);

   if(emsa_name.arg_count() > 0)
      {
      const std::string pos_hash = emsa_name.arg(0);
      return pos_hash;
      }

   // If we don't understand what this is return a safe default
#if defined(BOTAN_HAS_SHA2_64)
   return "SHA-512";
#else
   return "SHA-256";
#endif
   }

}
/*
* Sets of allowed padding schemes for public key types
*
* This file was automatically generated by ./src/scripts/oids.py on 2017-12-20
*
* All manual edits to this file will be lost. Edit the script
* then regenerate this source file.
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <map>
#include <string>

namespace Botan {

const std::map<const std::string, std::vector<std::string>> allowed_signature_paddings =
   {
   { "DSA", {"EMSA1"} },
   { "ECDSA", {"EMSA1"} },
   { "ECGDSA", {"EMSA1"} },
   { "ECKCDSA", {"EMSA1"} },
   { "GOST-34.10", {"EMSA1"} },
   { "GOST-34.10-2012-256", {"EMSA1"} },
   { "GOST-34.10-2012-512", {"EMSA1"} },
   { "RSA", {"EMSA4", "EMSA3"} },
   };

const std::vector<std::string> get_sig_paddings(const std::string algo)
   {
   if(allowed_signature_paddings.count(algo) > 0)
      return allowed_signature_paddings.at(algo);
   return {};
   }

bool sig_algo_and_pad_ok(const std::string algo, const std::string padding)
   {
   std::vector<std::string> pads = get_sig_paddings(algo);
   return std::find(pads.begin(), pads.end(), padding) != pads.end();
   }
}
/*
* Blinding for public key operations
* (C) 1999-2010,2015 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

Blinder::Blinder(const BigInt& modulus,
                 RandomNumberGenerator& rng,
                 std::function<BigInt (const BigInt&)> fwd,
                 std::function<BigInt (const BigInt&)> inv) :
      m_reducer(modulus),
      m_rng(rng),
      m_fwd_fn(fwd),
      m_inv_fn(inv),
      m_modulus_bits(modulus.bits()),
      m_e{},
      m_d{},
      m_counter{}
   {
   const BigInt k = blinding_nonce();
   m_e = m_fwd_fn(k);
   m_d = m_inv_fn(k);
   }

BigInt Blinder::blinding_nonce() const
   {
   return BigInt(m_rng, m_modulus_bits - 1);
   }

BigInt Blinder::blind(const BigInt& i) const
   {
   if(!m_reducer.initialized())
      throw Invalid_State("Blinder not initialized, cannot blind");

   ++m_counter;

   if((BOTAN_BLINDING_REINIT_INTERVAL > 0) && (m_counter > BOTAN_BLINDING_REINIT_INTERVAL))
      {
      const BigInt k = blinding_nonce();
      m_e = m_fwd_fn(k);
      m_d = m_inv_fn(k);
      m_counter = 0;
      }
   else
      {
      m_e = m_reducer.square(m_e);
      m_d = m_reducer.square(m_d);
      }

   return m_reducer.multiply(i, m_e);
   }

BigInt Blinder::unblind(const BigInt& i) const
   {
   if(!m_reducer.initialized())
      throw Invalid_State("Blinder not initialized, cannot unblind");

   return m_reducer.multiply(i, m_d);
   }

}
/*
* PK Key
* (C) 1999-2010,2016 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_RSA)
#endif

#if defined(BOTAN_HAS_DSA)
#endif

#if defined(BOTAN_HAS_DL_GROUP)
#endif

#if defined(BOTAN_HAS_DIFFIE_HELLMAN)
#endif

#if defined(BOTAN_HAS_ECC_PUBLIC_KEY_CRYPTO)
#endif

#if defined(BOTAN_HAS_ECDSA)
#endif

#if defined(BOTAN_HAS_ECGDSA)
#endif

#if defined(BOTAN_HAS_ECKCDSA)
#endif

#if defined(BOTAN_HAS_ED25519)
#endif

#if defined(BOTAN_HAS_GOST_34_10_2001)
#endif

#if defined(BOTAN_HAS_ELGAMAL)
#endif

#if defined(BOTAN_HAS_ECDH)
#endif

#if defined(BOTAN_HAS_CURVE_25519)
#endif

#if defined(BOTAN_HAS_MCELIECE)
#endif

#if defined(BOTAN_HAS_XMSS_RFC8391)
#endif

#if defined(BOTAN_HAS_SM2)
#endif

#if defined(BOTAN_HAS_OPENSSL)
#endif

namespace Botan {

std::unique_ptr<Public_Key>
load_public_key(const AlgorithmIdentifier& alg_id,
                const std::vector<uint8_t>& key_bits)
   {
   const std::string oid_str = alg_id.get_oid().to_formatted_string();
   const std::vector<std::string> alg_info = split_on(oid_str, '/');
   const std::string alg_name = alg_info[0];

#if defined(BOTAN_HAS_RSA)
   if(alg_name == "RSA")
      return std::unique_ptr<Public_Key>(new RSA_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_CURVE_25519)
   if(alg_name == "Curve25519")
      return std::unique_ptr<Public_Key>(new Curve25519_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_MCELIECE)
   if(alg_name == "McEliece")
      return std::unique_ptr<Public_Key>(new McEliece_PublicKey(key_bits));
#endif

#if defined(BOTAN_HAS_ECDSA)
   if(alg_name == "ECDSA")
      return std::unique_ptr<Public_Key>(new ECDSA_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_ECDH)
   if(alg_name == "ECDH")
      return std::unique_ptr<Public_Key>(new ECDH_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_DIFFIE_HELLMAN)
   if(alg_name == "DH")
      return std::unique_ptr<Public_Key>(new DH_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_DSA)
   if(alg_name == "DSA")
      return std::unique_ptr<Public_Key>(new DSA_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_ELGAMAL)
   if(alg_name == "ElGamal")
      return std::unique_ptr<Public_Key>(new ElGamal_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_ECGDSA)
   if(alg_name == "ECGDSA")
      return std::unique_ptr<Public_Key>(new ECGDSA_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_ECKCDSA)
   if(alg_name == "ECKCDSA")
      return std::unique_ptr<Public_Key>(new ECKCDSA_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_ED25519)
   if(alg_name == "Ed25519")
      return std::unique_ptr<Public_Key>(new Ed25519_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_GOST_34_10_2001)
   if(alg_name == "GOST-34.10" || alg_name == "GOST-34.10-2012-256" || alg_name == "GOST-34.10-2012-512")
      return std::unique_ptr<Public_Key>(new GOST_3410_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_SM2)
   if(alg_name == "SM2" || alg_name == "SM2_Sig" || alg_name == "SM2_Enc")
      return std::unique_ptr<Public_Key>(new SM2_PublicKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_XMSS_RFC8391)
   if(alg_name == "XMSS")
      return std::unique_ptr<Public_Key>(new XMSS_PublicKey(key_bits));
#endif

   throw Decoding_Error("Unknown or unavailable public key algorithm " + alg_name);
   }

std::unique_ptr<Private_Key>
load_private_key(const AlgorithmIdentifier& alg_id,
                 const secure_vector<uint8_t>& key_bits)
   {
   const std::string alg_name = alg_id.get_oid().to_formatted_string();

#if defined(BOTAN_HAS_RSA)
   if(alg_name == "RSA")
      return std::unique_ptr<Private_Key>(new RSA_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_CURVE_25519)
   if(alg_name == "Curve25519")
      return std::unique_ptr<Private_Key>(new Curve25519_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_ECDSA)
   if(alg_name == "ECDSA")
      return std::unique_ptr<Private_Key>(new ECDSA_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_ECDH)
   if(alg_name == "ECDH")
      return std::unique_ptr<Private_Key>(new ECDH_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_DIFFIE_HELLMAN)
   if(alg_name == "DH")
      return std::unique_ptr<Private_Key>(new DH_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_DSA)
   if(alg_name == "DSA")
      return std::unique_ptr<Private_Key>(new DSA_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_MCELIECE)
   if(alg_name == "McEliece")
      return std::unique_ptr<Private_Key>(new McEliece_PrivateKey(key_bits));
#endif

#if defined(BOTAN_HAS_ECGDSA)
   if(alg_name == "ECGDSA")
      return std::unique_ptr<Private_Key>(new ECGDSA_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_ECKCDSA)
   if(alg_name == "ECKCDSA")
      return std::unique_ptr<Private_Key>(new ECKCDSA_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_ED25519)
   if(alg_name == "Ed25519")
      return std::unique_ptr<Private_Key>(new Ed25519_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_GOST_34_10_2001)
   if(alg_name == "GOST-34.10" || alg_name == "GOST-34.10-2012-256" || alg_name == "GOST-34.10-2012-512")
      return std::unique_ptr<Private_Key>(new GOST_3410_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_SM2)
   if(alg_name == "SM2" || alg_name == "SM2_Sig" || alg_name == "SM2_Enc")
      return std::unique_ptr<Private_Key>(new SM2_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_ELGAMAL)
   if(alg_name == "ElGamal")
      return std::unique_ptr<Private_Key>(new ElGamal_PrivateKey(alg_id, key_bits));
#endif

#if defined(BOTAN_HAS_XMSS_RFC8391)
   if(alg_name == "XMSS")
      return std::unique_ptr<Private_Key>(new XMSS_PrivateKey(key_bits));
#endif

   throw Decoding_Error("Unknown or unavailable public key algorithm " + alg_name);
   }

#if defined(BOTAN_HAS_ECC_GROUP)

namespace {

std::string default_ec_group_for(const std::string& alg_name)
   {
   if(alg_name == "SM2" || alg_name == "SM2_Enc" || alg_name == "SM2_Sig")
      return "sm2p256v1";
   if(alg_name == "GOST-34.10" || alg_name == "GOST-34.10-2012-256")
      return "gost_256A";
   if(alg_name == "GOST-34.10-2012-512")
      return "gost_512A";
   if(alg_name == "ECGDSA")
      return "brainpool256r1";
   return "secp256r1";

   }

}

#endif

std::unique_ptr<Private_Key>
create_private_key(const std::string& alg_name,
                   RandomNumberGenerator& rng,
                   const std::string& params,
                   const std::string& provider)
   {
   /*
   * Default paramaters are chosen for work factor > 2**128 where possible
   */

#if defined(BOTAN_HAS_CURVE_25519)
   if(alg_name == "Curve25519")
      return std::unique_ptr<Private_Key>(new Curve25519_PrivateKey(rng));
#endif

#if defined(BOTAN_HAS_RSA)
   if(alg_name == "RSA")
      {
      const size_t rsa_bits = (params.empty() ? 3072 : to_u32bit(params));
#if defined(BOTAN_HAS_OPENSSL)
      if(provider.empty() || provider == "openssl")
         {
         std::unique_ptr<Botan::Private_Key> pk;
         if((pk = make_openssl_rsa_private_key(rng, rsa_bits)))
            return pk;

         if(!provider.empty())
            return nullptr;
         }
#endif
      return std::unique_ptr<Private_Key>(new RSA_PrivateKey(rng, rsa_bits));
      }
#endif

#if defined(BOTAN_HAS_MCELIECE)
   if(alg_name == "McEliece")
      {
      std::vector<std::string> mce_param =
         Botan::split_on(params.empty() ? "2960,57" : params, ',');

      if(mce_param.size() != 2)
         throw Invalid_Argument("create_private_key bad McEliece parameters " + params);

      size_t mce_n = Botan::to_u32bit(mce_param[0]);
      size_t mce_t = Botan::to_u32bit(mce_param[1]);

      return std::unique_ptr<Botan::Private_Key>(new Botan::McEliece_PrivateKey(rng, mce_n, mce_t));
      }
#endif

#if defined(BOTAN_HAS_XMSS_RFC8391)
   if(alg_name == "XMSS")
      {
      return std::unique_ptr<Private_Key>(
         new XMSS_PrivateKey(XMSS_Parameters(params.empty() ? "XMSS-SHA2_10_512" : params).oid(), rng));
      }
#endif

#if defined(BOTAN_HAS_ED25519)
   if(alg_name == "Ed25519")
      {
      return std::unique_ptr<Private_Key>(new Ed25519_PrivateKey(rng));
      }
#endif

   // ECC crypto
#if defined(BOTAN_HAS_ECC_PUBLIC_KEY_CRYPTO)

   if(alg_name == "ECDSA" ||
      alg_name == "ECDH" ||
      alg_name == "ECKCDSA" ||
      alg_name == "ECGDSA" ||
      alg_name == "SM2" ||
      alg_name == "SM2_Sig" ||
      alg_name == "SM2_Enc" ||
      alg_name == "GOST-34.10" ||
      alg_name == "GOST-34.10-2012-256" ||
      alg_name == "GOST-34.10-2012-512")
      {
      const EC_Group ec_group(params.empty() ? default_ec_group_for(alg_name) : params);

#if defined(BOTAN_HAS_ECDSA)
      if(alg_name == "ECDSA")
         return std::unique_ptr<Private_Key>(new ECDSA_PrivateKey(rng, ec_group));
#endif

#if defined(BOTAN_HAS_ECDH)
      if(alg_name == "ECDH")
         return std::unique_ptr<Private_Key>(new ECDH_PrivateKey(rng, ec_group));
#endif

#if defined(BOTAN_HAS_ECKCDSA)
      if(alg_name == "ECKCDSA")
         return std::unique_ptr<Private_Key>(new ECKCDSA_PrivateKey(rng, ec_group));
#endif

#if defined(BOTAN_HAS_GOST_34_10_2001)
      if(alg_name == "GOST-34.10" || alg_name == "GOST-34.10-2012-256" || alg_name == "GOST-34.10-2012-512")
         return std::unique_ptr<Private_Key>(new GOST_3410_PrivateKey(rng, ec_group));
#endif

#if defined(BOTAN_HAS_SM2)
      if(alg_name == "SM2" || alg_name == "SM2_Sig" || alg_name == "SM2_Enc")
         return std::unique_ptr<Private_Key>(new SM2_PrivateKey(rng, ec_group));
#endif

#if defined(BOTAN_HAS_ECGDSA)
      if(alg_name == "ECGDSA")
         return std::unique_ptr<Private_Key>(new ECGDSA_PrivateKey(rng, ec_group));
#endif
      }
#endif

   // DL crypto
#if defined(BOTAN_HAS_DL_GROUP)
   if(alg_name == "DH" || alg_name == "DSA" || alg_name == "ElGamal")
      {
      std::string default_group = (alg_name == "DSA") ? "dsa/botan/2048" : "modp/ietf/2048";
      DL_Group modp_group(params.empty() ? default_group : params);

#if defined(BOTAN_HAS_DIFFIE_HELLMAN)
      if(alg_name == "DH")
         return std::unique_ptr<Private_Key>(new DH_PrivateKey(rng, modp_group));
#endif

#if defined(BOTAN_HAS_DSA)
      if(alg_name == "DSA")
         return std::unique_ptr<Private_Key>(new DSA_PrivateKey(rng, modp_group));
#endif

#if defined(BOTAN_HAS_ELGAMAL)
      if(alg_name == "ElGamal")
         return std::unique_ptr<Private_Key>(new ElGamal_PrivateKey(rng, modp_group));
#endif
      }
#endif

   BOTAN_UNUSED(alg_name, rng, params, provider);

   return std::unique_ptr<Private_Key>();
   }

std::vector<std::string>
probe_provider_private_key(const std::string& alg_name,
                           const std::vector<std::string> possible)
   {
   std::vector<std::string> providers;
   for(auto&& prov : possible)
      {
      if(prov == "base" ||
#if defined(BOTAN_HAS_OPENSSL)
         (prov == "openssl" && alg_name == "RSA") ||
#endif
         0)
         {
         providers.push_back(prov); // available
         }
      }

   BOTAN_UNUSED(alg_name);

   return providers;
   }
}
/*
* PK Key Types
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

std::string create_hex_fingerprint(const uint8_t bits[],
                                   size_t bits_len,
                                   const std::string& hash_name)
   {
   std::unique_ptr<HashFunction> hash_fn(HashFunction::create_or_throw(hash_name));
   const std::string hex_hash = hex_encode(hash_fn->process(bits, bits_len));

   std::string fprint;

   for(size_t i = 0; i != hex_hash.size(); i += 2)
      {
      if(i != 0)
         fprint.push_back(':');

      fprint.push_back(hex_hash[i]);
      fprint.push_back(hex_hash[i+1]);
      }

   return fprint;
   }

std::vector<uint8_t> Public_Key::subject_public_key() const
   {
   std::vector<uint8_t> output;

   DER_Encoder(output).start_cons(SEQUENCE)
         .encode(algorithm_identifier())
         .encode(public_key_bits(), BIT_STRING)
      .end_cons();

   return output;
   }

/*
* Default OID access
*/
OID Public_Key::get_oid() const
   {
   const OID o = OIDS::str2oid_or_empty(algo_name());
   if(o.empty())
      throw Lookup_Error("PK algo " + algo_name() + " has no defined OIDs");
   return o;
   }

secure_vector<uint8_t> Private_Key::private_key_info() const
   {
   const size_t PKCS8_VERSION = 0;

   return DER_Encoder()
         .start_cons(SEQUENCE)
            .encode(PKCS8_VERSION)
            .encode(pkcs8_algorithm_identifier())
            .encode(private_key_bits(), OCTET_STRING)
         .end_cons()
      .get_contents();
   }

/*
* Hash of the X.509 subjectPublicKey encoding
*/
std::string Public_Key::fingerprint_public(const std::string& hash_algo) const
   {
   return create_hex_fingerprint(subject_public_key(), hash_algo);
   }

/*
* Hash of the PKCS #8 encoding for this key object
*/
std::string Private_Key::fingerprint_private(const std::string& hash_algo) const
   {
   return create_hex_fingerprint(private_key_bits(), hash_algo);
   }

std::unique_ptr<PK_Ops::Encryption>
Public_Key::create_encryption_op(RandomNumberGenerator& /*rng*/,
                                 const std::string& /*params*/,
                                 const std::string& /*provider*/) const
   {
   throw Lookup_Error(algo_name() + " does not support encryption");
   }

std::unique_ptr<PK_Ops::KEM_Encryption>
Public_Key::create_kem_encryption_op(RandomNumberGenerator& /*rng*/,
                                     const std::string& /*params*/,
                                     const std::string& /*provider*/) const
   {
   throw Lookup_Error(algo_name() + " does not support KEM encryption");
   }

std::unique_ptr<PK_Ops::Verification>
Public_Key::create_verification_op(const std::string& /*params*/,
                                   const std::string& /*provider*/) const
   {
   throw Lookup_Error(algo_name() + " does not support verification");
   }

std::unique_ptr<PK_Ops::Decryption>
Private_Key::create_decryption_op(RandomNumberGenerator& /*rng*/,
                                  const std::string& /*params*/,
                                  const std::string& /*provider*/) const
   {
   throw Lookup_Error(algo_name() + " does not support decryption");
   }

std::unique_ptr<PK_Ops::KEM_Decryption>
Private_Key::create_kem_decryption_op(RandomNumberGenerator& /*rng*/,
                                      const std::string& /*params*/,
                                      const std::string& /*provider*/) const
   {
   throw Lookup_Error(algo_name() + " does not support KEM decryption");
   }

std::unique_ptr<PK_Ops::Signature>
Private_Key::create_signature_op(RandomNumberGenerator& /*rng*/,
                                 const std::string& /*params*/,
                                 const std::string& /*provider*/) const
   {
   throw Lookup_Error(algo_name() + " does not support signatures");
   }

std::unique_ptr<PK_Ops::Key_Agreement>
Private_Key::create_key_agreement_op(RandomNumberGenerator& /*rng*/,
                                     const std::string& /*params*/,
                                     const std::string& /*provider*/) const
   {
   throw Lookup_Error(algo_name() + " does not support key agreement");
   }

}
/*
* PK Operation Types
* (C) 2010,2015 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

PK_Ops::Encryption_with_EME::Encryption_with_EME(const std::string& eme)
   {
   m_eme.reset(get_eme(eme));
   if(!m_eme.get())
      throw Algorithm_Not_Found(eme);
   }

size_t PK_Ops::Encryption_with_EME::max_input_bits() const
   {
   return 8 * m_eme->maximum_input_size(max_raw_input_bits());
   }

secure_vector<uint8_t> PK_Ops::Encryption_with_EME::encrypt(const uint8_t msg[], size_t msg_len,
                                                         RandomNumberGenerator& rng)
   {
   const size_t max_raw = max_raw_input_bits();
   const std::vector<uint8_t> encoded = unlock(m_eme->encode(msg, msg_len, max_raw, rng));
   return raw_encrypt(encoded.data(), encoded.size(), rng);
   }

PK_Ops::Decryption_with_EME::Decryption_with_EME(const std::string& eme)
   {
   m_eme.reset(get_eme(eme));
   if(!m_eme.get())
      throw Algorithm_Not_Found(eme);
   }

secure_vector<uint8_t>
PK_Ops::Decryption_with_EME::decrypt(uint8_t& valid_mask,
                                     const uint8_t ciphertext[],
                                     size_t ciphertext_len)
   {
   const secure_vector<uint8_t> raw = raw_decrypt(ciphertext, ciphertext_len);
   return m_eme->unpad(valid_mask, raw.data(), raw.size());
   }

PK_Ops::Key_Agreement_with_KDF::Key_Agreement_with_KDF(const std::string& kdf)
   {
   if(kdf != "Raw")
      m_kdf.reset(get_kdf(kdf));
   }

secure_vector<uint8_t> PK_Ops::Key_Agreement_with_KDF::agree(size_t key_len,
                                                          const uint8_t w[], size_t w_len,
                                                          const uint8_t salt[], size_t salt_len)
   {
   secure_vector<uint8_t> z = raw_agree(w, w_len);
   if(m_kdf)
      return m_kdf->derive_key(key_len, z, salt, salt_len);
   return z;
  }

PK_Ops::Signature_with_EMSA::Signature_with_EMSA(const std::string& emsa) :
   Signature(),
   m_emsa(get_emsa(emsa)),
   m_hash(hash_for_emsa(emsa)),
   m_prefix_used(false)
   {
   if(!m_emsa)
      throw Algorithm_Not_Found(emsa);
   }

void PK_Ops::Signature_with_EMSA::update(const uint8_t msg[], size_t msg_len)
   {
   if(has_prefix() && !m_prefix_used)
      {
      m_prefix_used = true;
      secure_vector<uint8_t> prefix = message_prefix();
      m_emsa->update(prefix.data(), prefix.size());
      }
   m_emsa->update(msg, msg_len);
   }

secure_vector<uint8_t> PK_Ops::Signature_with_EMSA::sign(RandomNumberGenerator& rng)
   {
   m_prefix_used = false;
   const secure_vector<uint8_t> msg = m_emsa->raw_data();
   const auto padded = m_emsa->encoding_of(msg, this->max_input_bits(), rng);
   return raw_sign(padded.data(), padded.size(), rng);
   }

PK_Ops::Verification_with_EMSA::Verification_with_EMSA(const std::string& emsa) :
   Verification(),
   m_emsa(get_emsa(emsa)),
   m_hash(hash_for_emsa(emsa)),
   m_prefix_used(false)
   {
   if(!m_emsa)
      throw Algorithm_Not_Found(emsa);
   }

void PK_Ops::Verification_with_EMSA::update(const uint8_t msg[], size_t msg_len)
   {
   if(has_prefix() && !m_prefix_used)
      {
      m_prefix_used = true;
      secure_vector<uint8_t> prefix = message_prefix();
      m_emsa->update(prefix.data(), prefix.size());
      }
   m_emsa->update(msg, msg_len);
   }

bool PK_Ops::Verification_with_EMSA::is_valid_signature(const uint8_t sig[], size_t sig_len)
   {
   m_prefix_used = false;
   const secure_vector<uint8_t> msg = m_emsa->raw_data();

   if(with_recovery())
      {
      secure_vector<uint8_t> output_of_key = verify_mr(sig, sig_len);
      return m_emsa->verify(output_of_key, msg, max_input_bits());
      }
   else
      {
      Null_RNG rng;
      secure_vector<uint8_t> encoded = m_emsa->encoding_of(msg, max_input_bits(), rng);
      return verify(encoded.data(), encoded.size(), sig, sig_len);
      }
   }

void PK_Ops::KEM_Encryption_with_KDF::kem_encrypt(secure_vector<uint8_t>& out_encapsulated_key,
                                                  secure_vector<uint8_t>& out_shared_key,
                                                  size_t desired_shared_key_len,
                                                  Botan::RandomNumberGenerator& rng,
                                                  const uint8_t salt[],
                                                  size_t salt_len)
   {
   secure_vector<uint8_t> raw_shared;
   this->raw_kem_encrypt(out_encapsulated_key, raw_shared, rng);

   out_shared_key = m_kdf->derive_key(desired_shared_key_len,
                                      raw_shared.data(), raw_shared.size(),
                                      salt, salt_len);
   }

PK_Ops::KEM_Encryption_with_KDF::KEM_Encryption_with_KDF(const std::string& kdf)
   {
   m_kdf.reset(get_kdf(kdf));
   }

secure_vector<uint8_t>
PK_Ops::KEM_Decryption_with_KDF::kem_decrypt(const uint8_t encap_key[],
                                             size_t len,
                                             size_t desired_shared_key_len,
                                             const uint8_t salt[],
                                             size_t salt_len)
   {
   secure_vector<uint8_t> raw_shared = this->raw_kem_decrypt(encap_key, len);

   return m_kdf->derive_key(desired_shared_key_len,
                            raw_shared.data(), raw_shared.size(),
                            salt, salt_len);
   }

PK_Ops::KEM_Decryption_with_KDF::KEM_Decryption_with_KDF(const std::string& kdf)
   {
   m_kdf.reset(get_kdf(kdf));
   }

}
/*
* PKCS #8
* (C) 1999-2010,2014,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_PKCS5_PBES2)
#endif

namespace Botan {

namespace PKCS8 {

namespace {

/*
* Get info from an EncryptedPrivateKeyInfo
*/
secure_vector<uint8_t> PKCS8_extract(DataSource& source,
                                  AlgorithmIdentifier& pbe_alg_id)
   {
   secure_vector<uint8_t> key_data;

   BER_Decoder(source)
      .start_cons(SEQUENCE)
         .decode(pbe_alg_id)
         .decode(key_data, OCTET_STRING)
      .verify_end();

   return key_data;
   }

/*
* PEM decode and/or decrypt a private key
*/
secure_vector<uint8_t> PKCS8_decode(
   DataSource& source,
   std::function<std::string ()> get_passphrase,
   AlgorithmIdentifier& pk_alg_id,
   bool is_encrypted)
   {
   AlgorithmIdentifier pbe_alg_id;
   secure_vector<uint8_t> key_data, key;

   try {
      if(ASN1::maybe_BER(source) && !PEM_Code::matches(source))
         {
         if(is_encrypted)
            {
            key_data = PKCS8_extract(source, pbe_alg_id);
            }
         else
            {
            // todo read more efficiently
            while(!source.end_of_data())
               {
               uint8_t b;
               size_t read = source.read_byte(b);
               if(read)
                  {
                  key_data.push_back(b);
                  }
               }
            }
         }
      else
         {
         std::string label;
         key_data = PEM_Code::decode(source, label);

         // todo remove autodetect for pem as well?
         if(label == "PRIVATE KEY")
            is_encrypted = false;
         else if(label == "ENCRYPTED PRIVATE KEY")
            {
            DataSource_Memory key_source(key_data);
            key_data = PKCS8_extract(key_source, pbe_alg_id);
            }
         else
            throw PKCS8_Exception("Unknown PEM label " + label);
         }

      if(key_data.empty())
         throw PKCS8_Exception("No key data found");
      }
   catch(Decoding_Error& e)
      {
      throw Decoding_Error("PKCS #8 private key decoding", e);
      }

   try
      {
      if(is_encrypted)
         {
         if(OIDS::oid2str_or_throw(pbe_alg_id.get_oid()) != "PBE-PKCS5v20")
            throw PKCS8_Exception("Unknown PBE type " + pbe_alg_id.get_oid().to_string());
#if defined(BOTAN_HAS_PKCS5_PBES2)
         key = pbes2_decrypt(key_data, get_passphrase(), pbe_alg_id.get_parameters());
#else
         BOTAN_UNUSED(get_passphrase);
         throw Decoding_Error("Private key is encrypted but PBES2 was disabled in build");
#endif
         }
      else
         key = key_data;

      BER_Decoder(key)
         .start_cons(SEQUENCE)
            .decode_and_check<size_t>(0, "Unknown PKCS #8 version number")
            .decode(pk_alg_id)
            .decode(key, OCTET_STRING)
            .discard_remaining()
         .end_cons();
      }
   catch(std::exception& e)
      {
      throw Decoding_Error("PKCS #8 private key decoding", e);
      }
   return key;
   }

}

/*
* BER encode a PKCS #8 private key, unencrypted
*/
secure_vector<uint8_t> BER_encode(const Private_Key& key)
   {
   // keeping around for compat
   return key.private_key_info();
   }

/*
* PEM encode a PKCS #8 private key, unencrypted
*/
std::string PEM_encode(const Private_Key& key)
   {
   return PEM_Code::encode(PKCS8::BER_encode(key), "PRIVATE KEY");
   }

#if defined(BOTAN_HAS_PKCS5_PBES2)

namespace {

std::pair<std::string, std::string>
choose_pbe_params(const std::string& pbe_algo, const std::string& key_algo)
   {
   if(pbe_algo.empty())
      {
      /*
      * For algorithms where we are using a non-RFC format anyway, default to
      * SIV or GCM. For others (RSA, ECDSA, ...) default to something widely
      * compatible.
      */
      const bool nonstandard_pk = (key_algo == "McEliece" || key_algo == "XMSS");

      if(nonstandard_pk)
         {
#if defined(BOTAN_HAS_AEAD_SIV) && defined(BOTAN_HAS_SHA2_64)
         return std::make_pair("AES-256/SIV", "SHA-512");
#elif defined(BOTAN_HAS_AEAD_GCM) && defined(BOTAN_HAS_SHA2_64)
         return std::make_pair("AES-256/GCM", "SHA-512");
#endif
         }

      // Default is something compatible with everyone else
      return std::make_pair("AES-256/CBC", "SHA-256");
      }

   SCAN_Name request(pbe_algo);

   if(request.arg_count() != 2 ||
      (request.algo_name() != "PBE-PKCS5v20" && request.algo_name() != "PBES2"))
      {
      throw Invalid_Argument("Unsupported PBE " + pbe_algo);
      }

   return std::make_pair(request.arg(0), request.arg(1));
   }

}

#endif

/*
* BER encode a PKCS #8 private key, encrypted
*/
std::vector<uint8_t> BER_encode(const Private_Key& key,
                             RandomNumberGenerator& rng,
                             const std::string& pass,
                             std::chrono::milliseconds msec,
                             const std::string& pbe_algo)
   {
#if defined(BOTAN_HAS_PKCS5_PBES2)
   const auto pbe_params = choose_pbe_params(pbe_algo, key.algo_name());

   const std::pair<AlgorithmIdentifier, std::vector<uint8_t>> pbe_info =
      pbes2_encrypt_msec(PKCS8::BER_encode(key), pass, msec, nullptr,
                         pbe_params.first, pbe_params.second, rng);

   std::vector<uint8_t> output;
   DER_Encoder der(output);
   der.start_cons(SEQUENCE)
         .encode(pbe_info.first)
         .encode(pbe_info.second, OCTET_STRING)
      .end_cons();

   return output;
#else
   BOTAN_UNUSED(key, rng, pass, msec, pbe_algo);
   throw Encoding_Error("PKCS8::BER_encode cannot encrypt because PBES2 was disabled in build");
#endif
   }

/*
* PEM encode a PKCS #8 private key, encrypted
*/
std::string PEM_encode(const Private_Key& key,
                       RandomNumberGenerator& rng,
                       const std::string& pass,
                       std::chrono::milliseconds msec,
                       const std::string& pbe_algo)
   {
   if(pass.empty())
      return PEM_encode(key);

   return PEM_Code::encode(PKCS8::BER_encode(key, rng, pass, msec, pbe_algo),
                           "ENCRYPTED PRIVATE KEY");
   }

/*
* BER encode a PKCS #8 private key, encrypted
*/
std::vector<uint8_t> BER_encode_encrypted_pbkdf_iter(const Private_Key& key,
                                                     RandomNumberGenerator& rng,
                                                     const std::string& pass,
                                                     size_t pbkdf_iterations,
                                                     const std::string& cipher,
                                                     const std::string& pbkdf_hash)
   {
#if defined(BOTAN_HAS_PKCS5_PBES2)
   const std::pair<AlgorithmIdentifier, std::vector<uint8_t>> pbe_info =
      pbes2_encrypt_iter(key.private_key_info(),
                         pass, pbkdf_iterations,
                         cipher.empty() ? "AES-256/CBC" : cipher,
                         pbkdf_hash.empty() ? "SHA-256" : pbkdf_hash,
                         rng);

   std::vector<uint8_t> output;
   DER_Encoder der(output);
   der.start_cons(SEQUENCE)
         .encode(pbe_info.first)
         .encode(pbe_info.second, OCTET_STRING)
      .end_cons();

   return output;

#else
   BOTAN_UNUSED(key, rng, pass, pbkdf_iterations, cipher, pbkdf_hash);
   throw Encoding_Error("PKCS8::BER_encode_encrypted_pbkdf_iter cannot encrypt because PBES2 disabled in build");
#endif
   }

/*
* PEM encode a PKCS #8 private key, encrypted
*/
std::string PEM_encode_encrypted_pbkdf_iter(const Private_Key& key,
                                            RandomNumberGenerator& rng,
                                            const std::string& pass,
                                            size_t pbkdf_iterations,
                                            const std::string& cipher,
                                            const std::string& pbkdf_hash)
   {
   return PEM_Code::encode(
      PKCS8::BER_encode_encrypted_pbkdf_iter(key, rng, pass, pbkdf_iterations, cipher, pbkdf_hash),
      "ENCRYPTED PRIVATE KEY");
   }

/*
* BER encode a PKCS #8 private key, encrypted
*/
std::vector<uint8_t> BER_encode_encrypted_pbkdf_msec(const Private_Key& key,
                                                     RandomNumberGenerator& rng,
                                                     const std::string& pass,
                                                     std::chrono::milliseconds pbkdf_msec,
                                                     size_t* pbkdf_iterations,
                                                     const std::string& cipher,
                                                     const std::string& pbkdf_hash)
   {
#if defined(BOTAN_HAS_PKCS5_PBES2)
   const std::pair<AlgorithmIdentifier, std::vector<uint8_t>> pbe_info =
      pbes2_encrypt_msec(key.private_key_info(), pass,
                         pbkdf_msec, pbkdf_iterations,
                         cipher.empty() ? "AES-256/CBC" : cipher,
                         pbkdf_hash.empty() ? "SHA-256" : pbkdf_hash,
                         rng);

   std::vector<uint8_t> output;
   DER_Encoder(output)
      .start_cons(SEQUENCE)
         .encode(pbe_info.first)
         .encode(pbe_info.second, OCTET_STRING)
      .end_cons();

   return output;
#else
   BOTAN_UNUSED(key, rng, pass, pbkdf_msec, pbkdf_iterations, cipher, pbkdf_hash);
   throw Encoding_Error("BER_encode_encrypted_pbkdf_msec cannot encrypt because PBES2 disabled in build");
#endif
   }

/*
* PEM encode a PKCS #8 private key, encrypted
*/
std::string PEM_encode_encrypted_pbkdf_msec(const Private_Key& key,
                                            RandomNumberGenerator& rng,
                                            const std::string& pass,
                                            std::chrono::milliseconds pbkdf_msec,
                                            size_t* pbkdf_iterations,
                                            const std::string& cipher,
                                            const std::string& pbkdf_hash)
   {
   return PEM_Code::encode(
      PKCS8::BER_encode_encrypted_pbkdf_msec(key, rng, pass, pbkdf_msec, pbkdf_iterations, cipher, pbkdf_hash),
      "ENCRYPTED PRIVATE KEY");
   }

namespace {

/*
* Extract a private key (encrypted/unencrypted) and return it
*/
std::unique_ptr<Private_Key>
load_key(DataSource& source,
         std::function<std::string ()> get_pass,
         bool is_encrypted)
   {
   AlgorithmIdentifier alg_id;
   secure_vector<uint8_t> pkcs8_key = PKCS8_decode(source, get_pass, alg_id, is_encrypted);

   const std::string alg_name = OIDS::oid2str_or_empty(alg_id.get_oid());
   if(alg_name.empty())
      throw PKCS8_Exception("Unknown algorithm OID: " +
                            alg_id.get_oid().to_string());

   return load_private_key(alg_id, pkcs8_key);
   }

}

/*
* Extract an encrypted private key and return it
*/
std::unique_ptr<Private_Key> load_key(DataSource& source,
                      std::function<std::string ()> get_pass)
   {
   return load_key(source, get_pass, true);
   }

/*
* Extract an encrypted private key and return it
*/
std::unique_ptr<Private_Key> load_key(DataSource& source,
                      const std::string& pass)
   {
   // We need to use bind rather than a lambda capturing `pass` here in order to avoid a Clang 8 bug.
   // See https://github.com/randombit/botan/issues/2255.
   return load_key(source, std::bind([](const std::string p) { return p; }, pass), true);
   }

/*
* Extract an unencrypted private key and return it
*/
std::unique_ptr<Private_Key> load_key(DataSource& source)
   {
   auto fail_fn = []() -> std::string {
      throw PKCS8_Exception("Internal error: Attempt to read password for unencrypted key");
   };

   return load_key(source, fail_fn, false);
   }

/*
* Make a copy of this private key
*/
std::unique_ptr<Private_Key> copy_key(const Private_Key& key)
   {
   DataSource_Memory source(PEM_encode(key));
   return PKCS8::load_key(source);
   }

/*
* Extract an encrypted private key and return it
*/
Private_Key* load_key(DataSource& source,
                      RandomNumberGenerator& rng,
                      std::function<std::string ()> get_pass)
   {
   BOTAN_UNUSED(rng);
   return PKCS8::load_key(source, get_pass).release();
   }

/*
* Extract an encrypted private key and return it
*/
Private_Key* load_key(DataSource& source,
                      RandomNumberGenerator& rng,
                      const std::string& pass)
   {
   BOTAN_UNUSED(rng);
   return PKCS8::load_key(source, pass).release();
   }

/*
* Extract an unencrypted private key and return it
*/
Private_Key* load_key(DataSource& source,
                      RandomNumberGenerator& rng)
   {
   BOTAN_UNUSED(rng);
   return PKCS8::load_key(source).release();
   }

#if defined(BOTAN_TARGET_OS_HAS_FILESYSTEM)

/*
* Extract an encrypted private key and return it
*/
Private_Key* load_key(const std::string& fsname,
                      RandomNumberGenerator& rng,
                      std::function<std::string ()> get_pass)
   {
   BOTAN_UNUSED(rng);
   DataSource_Stream in(fsname);
   return PKCS8::load_key(in, get_pass).release();
   }

/*
* Extract an encrypted private key and return it
*/
Private_Key* load_key(const std::string& fsname,
                      RandomNumberGenerator& rng,
                      const std::string& pass)
   {
   BOTAN_UNUSED(rng);
   DataSource_Stream in(fsname);
   // We need to use bind rather than a lambda capturing `pass` here in order to avoid a Clang 8 bug.
   // See https://github.com/randombit/botan/issues/2255.
   return PKCS8::load_key(in, std::bind([](const std::string p) { return p; }, pass)).release();
   }

/*
* Extract an unencrypted private key and return it
*/
Private_Key* load_key(const std::string& fsname,
                      RandomNumberGenerator& rng)
   {
   BOTAN_UNUSED(rng);
   DataSource_Stream in(fsname);
   return PKCS8::load_key(in).release();
   }
#endif

/*
* Make a copy of this private key
*/
Private_Key* copy_key(const Private_Key& key,
                      RandomNumberGenerator& rng)
   {
   BOTAN_UNUSED(rng);
   return PKCS8::copy_key(key).release();
   }



}

}
/*
* (C) 1999-2010,2015,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

secure_vector<uint8_t> PK_Decryptor::decrypt(const uint8_t in[], size_t length) const
   {
   uint8_t valid_mask = 0;

   secure_vector<uint8_t> decoded = do_decrypt(valid_mask, in, length);

   if(valid_mask == 0)
      throw Decoding_Error("Invalid public key ciphertext, cannot decrypt");

   return decoded;
   }

secure_vector<uint8_t>
PK_Decryptor::decrypt_or_random(const uint8_t in[],
                                size_t length,
                                size_t expected_pt_len,
                                RandomNumberGenerator& rng,
                                const uint8_t required_content_bytes[],
                                const uint8_t required_content_offsets[],
                                size_t required_contents_length) const
   {
   const secure_vector<uint8_t> fake_pms = rng.random_vec(expected_pt_len);

   uint8_t decrypt_valid = 0;
   secure_vector<uint8_t> decoded = do_decrypt(decrypt_valid, in, length);

   auto valid_mask = CT::Mask<uint8_t>::is_equal(decrypt_valid, 0xFF);
   valid_mask &= CT::Mask<uint8_t>(CT::Mask<size_t>::is_zero(decoded.size() ^ expected_pt_len));

   decoded.resize(expected_pt_len);

   for(size_t i = 0; i != required_contents_length; ++i)
      {
      /*
      These values are chosen by the application and for TLS are constants,
      so this early failure via assert is fine since we know 0,1 < 48

      If there is a protocol that has content checks on the key where
      the expected offsets are controllable by the attacker this could
      still leak.

      Alternately could always reduce the offset modulo the length?
      */

      const uint8_t exp = required_content_bytes[i];
      const uint8_t off = required_content_offsets[i];

      BOTAN_ASSERT(off < expected_pt_len, "Offset in range of plaintext");

      auto eq = CT::Mask<uint8_t>::is_equal(decoded[off], exp);

      valid_mask &= eq;
      }

   // If valid_mask is false, assign fake pre master instead
   valid_mask.select_n(decoded.data(), decoded.data(), fake_pms.data(), expected_pt_len);

   return decoded;
   }

secure_vector<uint8_t>
PK_Decryptor::decrypt_or_random(const uint8_t in[],
                                size_t length,
                                size_t expected_pt_len,
                                RandomNumberGenerator& rng) const
   {
   return decrypt_or_random(in, length, expected_pt_len, rng,
                            nullptr, nullptr, 0);
   }

PK_Encryptor_EME::PK_Encryptor_EME(const Public_Key& key,
                                   RandomNumberGenerator& rng,
                                   const std::string& padding,
                                   const std::string& provider)
   {
   m_op = key.create_encryption_op(rng, padding, provider);
   if(!m_op)
      throw Invalid_Argument("Key type " + key.algo_name() + " does not support encryption");
   }

PK_Encryptor_EME::~PK_Encryptor_EME() { /* for unique_ptr */ }

size_t PK_Encryptor_EME::ciphertext_length(size_t ptext_len) const
   {
   return m_op->ciphertext_length(ptext_len);
   }

std::vector<uint8_t>
PK_Encryptor_EME::enc(const uint8_t in[], size_t length, RandomNumberGenerator& rng) const
   {
   return unlock(m_op->encrypt(in, length, rng));
   }

size_t PK_Encryptor_EME::maximum_input_size() const
   {
   return m_op->max_input_bits() / 8;
   }

PK_Decryptor_EME::PK_Decryptor_EME(const Private_Key& key,
                                   RandomNumberGenerator& rng,
                                   const std::string& padding,
                                   const std::string& provider)
   {
   m_op = key.create_decryption_op(rng, padding, provider);
   if(!m_op)
      throw Invalid_Argument("Key type " + key.algo_name() + " does not support decryption");
   }

PK_Decryptor_EME::~PK_Decryptor_EME() { /* for unique_ptr */ }

size_t PK_Decryptor_EME::plaintext_length(size_t ctext_len) const
   {
   return m_op->plaintext_length(ctext_len);
   }

secure_vector<uint8_t> PK_Decryptor_EME::do_decrypt(uint8_t& valid_mask,
                                                 const uint8_t in[], size_t in_len) const
   {
   return m_op->decrypt(valid_mask, in, in_len);
   }

PK_KEM_Encryptor::PK_KEM_Encryptor(const Public_Key& key,
                                   RandomNumberGenerator& rng,
                                   const std::string& param,
                                   const std::string& provider)
   {
   m_op = key.create_kem_encryption_op(rng, param, provider);
   if(!m_op)
      throw Invalid_Argument("Key type " + key.algo_name() + " does not support KEM encryption");
   }

PK_KEM_Encryptor::~PK_KEM_Encryptor() { /* for unique_ptr */ }

void PK_KEM_Encryptor::encrypt(secure_vector<uint8_t>& out_encapsulated_key,
                               secure_vector<uint8_t>& out_shared_key,
                               size_t desired_shared_key_len,
                               Botan::RandomNumberGenerator& rng,
                               const uint8_t salt[],
                               size_t salt_len)
   {
   m_op->kem_encrypt(out_encapsulated_key,
                     out_shared_key,
                     desired_shared_key_len,
                     rng,
                     salt,
                     salt_len);
   }

PK_KEM_Decryptor::PK_KEM_Decryptor(const Private_Key& key,
                                   RandomNumberGenerator& rng,
                                   const std::string& param,
                                   const std::string& provider)
   {
   m_op = key.create_kem_decryption_op(rng, param, provider);
   if(!m_op)
      throw Invalid_Argument("Key type " + key.algo_name() + " does not support KEM decryption");
   }

PK_KEM_Decryptor::~PK_KEM_Decryptor() { /* for unique_ptr */ }

secure_vector<uint8_t> PK_KEM_Decryptor::decrypt(const uint8_t encap_key[],
                                              size_t encap_key_len,
                                              size_t desired_shared_key_len,
                                              const uint8_t salt[],
                                              size_t salt_len)
   {
   return m_op->kem_decrypt(encap_key, encap_key_len,
                            desired_shared_key_len,
                            salt, salt_len);
   }

PK_Key_Agreement::PK_Key_Agreement(const Private_Key& key,
                                   RandomNumberGenerator& rng,
                                   const std::string& kdf,
                                   const std::string& provider)
   {
   m_op = key.create_key_agreement_op(rng, kdf, provider);
   if(!m_op)
      throw Invalid_Argument("Key type " + key.algo_name() + " does not support key agreement");
   }

PK_Key_Agreement::~PK_Key_Agreement() { /* for unique_ptr */ }

PK_Key_Agreement& PK_Key_Agreement::operator=(PK_Key_Agreement&& other)
   {
   if(this != &other)
      {
      m_op = std::move(other.m_op);
      }
   return (*this);
   }

PK_Key_Agreement::PK_Key_Agreement(PK_Key_Agreement&& other) :
   m_op(std::move(other.m_op))
   {}

size_t PK_Key_Agreement::agreed_value_size() const
   {
   return m_op->agreed_value_size();
   }

SymmetricKey PK_Key_Agreement::derive_key(size_t key_len,
                                          const uint8_t in[], size_t in_len,
                                          const uint8_t salt[],
                                          size_t salt_len) const
   {
   return m_op->agree(key_len, in, in_len, salt, salt_len);
   }

static void check_der_format_supported(Signature_Format format, size_t parts)
   {
      if(format != IEEE_1363 && parts == 1)
         throw Invalid_Argument("PK: This algorithm does not support DER encoding");
   }

PK_Signer::PK_Signer(const Private_Key& key,
                     RandomNumberGenerator& rng,
                     const std::string& emsa,
                     Signature_Format format,
                     const std::string& provider)
   {
   m_op = key.create_signature_op(rng, emsa, provider);
   if(!m_op)
      throw Invalid_Argument("Key type " + key.algo_name() + " does not support signature generation");
   m_sig_format = format;
   m_parts = key.message_parts();
   m_part_size = key.message_part_size();
   check_der_format_supported(format, m_parts);
   }

PK_Signer::~PK_Signer() { /* for unique_ptr */ }

void PK_Signer::update(const uint8_t in[], size_t length)
   {
   m_op->update(in, length);
   }

namespace {

std::vector<uint8_t> der_encode_signature(const std::vector<uint8_t>& sig,
                                          size_t parts,
                                          size_t part_size)
   {
   if(sig.size() % parts != 0 || sig.size() != parts * part_size)
      throw Encoding_Error("Unexpected size for DER signature");

   std::vector<BigInt> sig_parts(parts);
   for(size_t i = 0; i != sig_parts.size(); ++i)
      sig_parts[i].binary_decode(&sig[part_size*i], part_size);

   std::vector<uint8_t> output;
   DER_Encoder(output)
      .start_cons(SEQUENCE)
      .encode_list(sig_parts)
      .end_cons();
   return output;
   }

}

size_t PK_Signer::signature_length() const
   {
   if(m_sig_format == IEEE_1363)
      {
      return m_op->signature_length();
      }
   else if(m_sig_format == DER_SEQUENCE)
      {
      // This is a large over-estimate but its easier than computing
      // the exact value
      return m_op->signature_length() + (8 + 4*m_parts);
      }
   else
      throw Internal_Error("PK_Signer: Invalid signature format enum");
   }

std::vector<uint8_t> PK_Signer::signature(RandomNumberGenerator& rng)
   {
   const std::vector<uint8_t> sig = unlock(m_op->sign(rng));

   if(m_sig_format == IEEE_1363)
      {
      return sig;
      }
   else if(m_sig_format == DER_SEQUENCE)
      {
      return der_encode_signature(sig, m_parts, m_part_size);
      }
   else
      throw Internal_Error("PK_Signer: Invalid signature format enum");
   }

PK_Verifier::PK_Verifier(const Public_Key& key,
                         const std::string& emsa,
                         Signature_Format format,
                         const std::string& provider)
   {
   m_op = key.create_verification_op(emsa, provider);
   if(!m_op)
      throw Invalid_Argument("Key type " + key.algo_name() + " does not support signature verification");
   m_sig_format = format;
   m_parts = key.message_parts();
   m_part_size = key.message_part_size();
   check_der_format_supported(format, m_parts);
   }

PK_Verifier::~PK_Verifier() { /* for unique_ptr */ }

void PK_Verifier::set_input_format(Signature_Format format)
   {
   check_der_format_supported(format, m_parts);
   m_sig_format = format;
   }

bool PK_Verifier::verify_message(const uint8_t msg[], size_t msg_length,
                                 const uint8_t sig[], size_t sig_length)
   {
   update(msg, msg_length);
   return check_signature(sig, sig_length);
   }

void PK_Verifier::update(const uint8_t in[], size_t length)
   {
   m_op->update(in, length);
   }

bool PK_Verifier::check_signature(const uint8_t sig[], size_t length)
   {
   try {
      if(m_sig_format == IEEE_1363)
         {
         return m_op->is_valid_signature(sig, length);
         }
      else if(m_sig_format == DER_SEQUENCE)
         {
         std::vector<uint8_t> real_sig;
         BER_Decoder decoder(sig, length);
         BER_Decoder ber_sig = decoder.start_cons(SEQUENCE);

         BOTAN_ASSERT_NOMSG(m_parts != 0 && m_part_size != 0);

         size_t count = 0;

         while(ber_sig.more_items())
            {
            BigInt sig_part;
            ber_sig.decode(sig_part);
            real_sig += BigInt::encode_1363(sig_part, m_part_size);
            ++count;
            }

         if(count != m_parts)
            throw Decoding_Error("PK_Verifier: signature size invalid");

         const std::vector<uint8_t> reencoded =
            der_encode_signature(real_sig, m_parts, m_part_size);

         if(reencoded.size() != length ||
            same_mem(reencoded.data(), sig, reencoded.size()) == false)
            {
            throw Decoding_Error("PK_Verifier: signature is not the canonical DER encoding");
            }

         return m_op->is_valid_signature(real_sig.data(), real_sig.size());
         }
      else
         throw Internal_Error("PK_Verifier: Invalid signature format enum");
      }
   catch(Invalid_Argument&) { return false; }
   }

}
/*
* Public Key Work Factor Functions
* (C) 1999-2007,2012 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <cmath>

namespace Botan {

size_t ecp_work_factor(size_t bits)
   {
   return bits / 2;
   }

namespace {

size_t nfs_workfactor(size_t bits, double log2_k)
   {
   // approximates natural logarithm of an integer of given bitsize
   const double log2_e = 1.44269504088896340736;
   const double log_p = bits / log2_e;

   const double log_log_p = std::log(log_p);

   // RFC 3766: k * e^((1.92 + o(1)) * cubrt(ln(n) * (ln(ln(n)))^2))
   const double est = 1.92 * std::pow(log_p * log_log_p * log_log_p, 1.0/3.0);

   // return log2 of the workfactor
   return static_cast<size_t>(log2_k + log2_e * est);
   }

}

size_t if_work_factor(size_t bits)
   {
   // RFC 3766 estimates k at .02 and o(1) to be effectively zero for sizes of interest

   const double log2_k = -5.6438; // log2(.02)
   return nfs_workfactor(bits, log2_k);
   }

size_t dl_work_factor(size_t bits)
   {
   // Lacking better estimates...
   return if_work_factor(bits);
   }

size_t dl_exponent_size(size_t bits)
   {
   /*
   This uses a slightly tweaked version of the standard work factor
   function above. It assumes k is 1 (thus overestimating the strength
   of the prime group by 5-6 bits), and always returns at least 128 bits
   (this only matters for very small primes).
   */
   const size_t min_workfactor = 64;
   const double log2_k = 0;

   return 2 * std::max<size_t>(min_workfactor, nfs_workfactor(bits, log2_k));
   }

}
/*
* X.509 Public Key
* (C) 1999-2010 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace X509 {

std::vector<uint8_t> BER_encode(const Public_Key& key)
   {
   // keeping it around for compat
   return key.subject_public_key();
   }

/*
* PEM encode a X.509 public key
*/
std::string PEM_encode(const Public_Key& key)
   {
   return PEM_Code::encode(key.subject_public_key(),
                           "PUBLIC KEY");
   }

/*
* Extract a public key and return it
*/
Public_Key* load_key(DataSource& source)
   {
   try {
      AlgorithmIdentifier alg_id;
      std::vector<uint8_t> key_bits;

      if(ASN1::maybe_BER(source) && !PEM_Code::matches(source))
         {
         BER_Decoder(source)
            .start_cons(SEQUENCE)
            .decode(alg_id)
            .decode(key_bits, BIT_STRING)
         .end_cons();
         }
      else
         {
         DataSource_Memory ber(
            PEM_Code::decode_check_label(source, "PUBLIC KEY")
            );

         BER_Decoder(ber)
            .start_cons(SEQUENCE)
            .decode(alg_id)
            .decode(key_bits, BIT_STRING)
         .end_cons();
         }

      if(key_bits.empty())
         throw Decoding_Error("X.509 public key decoding");

      return load_public_key(alg_id, key_bits).release();
      }
   catch(Decoding_Error& e)
      {
      throw Decoding_Error("X.509 public key decoding", e);
      }
   }

#if defined(BOTAN_TARGET_OS_HAS_FILESYSTEM)
/*
* Extract a public key and return it
*/
Public_Key* load_key(const std::string& fsname)
   {
   DataSource_Stream source(fsname, true);
   return X509::load_key(source);
   }
#endif

/*
* Extract a public key and return it
*/
Public_Key* load_key(const std::vector<uint8_t>& mem)
   {
   DataSource_Memory source(mem);
   return X509::load_key(source);
   }

/*
* Make a copy of this public key
*/
Public_Key* copy_key(const Public_Key& key)
   {
   DataSource_Memory source(PEM_encode(key));
   return X509::load_key(source);
   }

}

}
/*
* (C) 2016 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_HAS_AUTO_SEEDING_RNG)
#endif

namespace Botan {

void RandomNumberGenerator::randomize_with_ts_input(uint8_t output[], size_t output_len)
   {
   if(this->accepts_input())
      {
      /*
      Form additional input which is provided to the PRNG implementation
      to paramaterize the KDF output.
      */
      uint8_t additional_input[16] = { 0 };
      store_le(OS::get_system_timestamp_ns(), additional_input);
      store_le(OS::get_high_resolution_clock(), additional_input + 8);

      this->randomize_with_input(output, output_len, additional_input, sizeof(additional_input));
      }
   else
      {
      this->randomize(output, output_len);
      }
   }

void RandomNumberGenerator::randomize_with_input(uint8_t output[], size_t output_len,
                                                 const uint8_t input[], size_t input_len)
   {
   this->add_entropy(input, input_len);
   this->randomize(output, output_len);
   }

size_t RandomNumberGenerator::reseed(Entropy_Sources& srcs,
                                     size_t poll_bits,
                                     std::chrono::milliseconds poll_timeout)
   {
   if(this->accepts_input())
      {
      return srcs.poll(*this, poll_bits, poll_timeout);
      }
   else
      {
      return 0;
      }
   }

void RandomNumberGenerator::reseed_from_rng(RandomNumberGenerator& rng, size_t poll_bits)
   {
   if(this->accepts_input())
      {
      secure_vector<uint8_t> buf(poll_bits / 8);
      rng.randomize(buf.data(), buf.size());
      this->add_entropy(buf.data(), buf.size());
      }
   }

RandomNumberGenerator* RandomNumberGenerator::make_rng()
   {
#if defined(BOTAN_HAS_AUTO_SEEDING_RNG)
   return new AutoSeeded_RNG;
#else
   throw Not_Implemented("make_rng failed, no AutoSeeded_RNG in this build");
#endif
   }

#if defined(BOTAN_TARGET_OS_HAS_THREADS)

#if defined(BOTAN_HAS_AUTO_SEEDING_RNG)
Serialized_RNG::Serialized_RNG() : m_rng(new AutoSeeded_RNG) {}
#else
Serialized_RNG::Serialized_RNG()
   {
   throw Not_Implemented("Serialized_RNG default constructor failed: AutoSeeded_RNG disabled in build");
   }
#endif

#endif

}
/*
* SHA-160
* (C) 1999-2008,2011 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

std::unique_ptr<HashFunction> SHA_160::copy_state() const
   {
   return std::unique_ptr<HashFunction>(new SHA_160(*this));
   }

namespace SHA1_F {

namespace {

/*
* SHA-160 F1 Function
*/
inline void F1(uint32_t A, uint32_t& B, uint32_t C, uint32_t D, uint32_t& E, uint32_t msg)
   {
   E += (D ^ (B & (C ^ D))) + msg + 0x5A827999 + rotl<5>(A);
   B  = rotl<30>(B);
   }

/*
* SHA-160 F2 Function
*/
inline void F2(uint32_t A, uint32_t& B, uint32_t C, uint32_t D, uint32_t& E, uint32_t msg)
   {
   E += (B ^ C ^ D) + msg + 0x6ED9EBA1 + rotl<5>(A);
   B  = rotl<30>(B);
   }

/*
* SHA-160 F3 Function
*/
inline void F3(uint32_t A, uint32_t& B, uint32_t C, uint32_t D, uint32_t& E, uint32_t msg)
   {
   E += ((B & C) | ((B | C) & D)) + msg + 0x8F1BBCDC + rotl<5>(A);
   B  = rotl<30>(B);
   }

/*
* SHA-160 F4 Function
*/
inline void F4(uint32_t A, uint32_t& B, uint32_t C, uint32_t D, uint32_t& E, uint32_t msg)
   {
   E += (B ^ C ^ D) + msg + 0xCA62C1D6 + rotl<5>(A);
   B  = rotl<30>(B);
   }

}

}

/*
* SHA-160 Compression Function
*/
void SHA_160::compress_n(const uint8_t input[], size_t blocks)
   {
   using namespace SHA1_F;

#if defined(BOTAN_HAS_SHA1_X86_SHA_NI)
   if(CPUID::has_intel_sha())
      {
      return sha1_compress_x86(m_digest, input, blocks);
      }
#endif

#if defined(BOTAN_HAS_SHA1_ARMV8)
   if(CPUID::has_arm_sha1())
      {
      return sha1_armv8_compress_n(m_digest, input, blocks);
      }
#endif

#if defined(BOTAN_HAS_SHA1_SSE2)
   if(CPUID::has_sse2())
      {
      return sse2_compress_n(m_digest, input, blocks);
      }

#endif

   uint32_t A = m_digest[0], B = m_digest[1], C = m_digest[2],
          D = m_digest[3], E = m_digest[4];

   m_W.resize(80);

   for(size_t i = 0; i != blocks; ++i)
      {
      load_be(m_W.data(), input, 16);

      for(size_t j = 16; j != 80; j += 8)
         {
         m_W[j  ] = rotl<1>(m_W[j-3] ^ m_W[j-8] ^ m_W[j-14] ^ m_W[j-16]);
         m_W[j+1] = rotl<1>(m_W[j-2] ^ m_W[j-7] ^ m_W[j-13] ^ m_W[j-15]);
         m_W[j+2] = rotl<1>(m_W[j-1] ^ m_W[j-6] ^ m_W[j-12] ^ m_W[j-14]);
         m_W[j+3] = rotl<1>(m_W[j  ] ^ m_W[j-5] ^ m_W[j-11] ^ m_W[j-13]);
         m_W[j+4] = rotl<1>(m_W[j+1] ^ m_W[j-4] ^ m_W[j-10] ^ m_W[j-12]);
         m_W[j+5] = rotl<1>(m_W[j+2] ^ m_W[j-3] ^ m_W[j- 9] ^ m_W[j-11]);
         m_W[j+6] = rotl<1>(m_W[j+3] ^ m_W[j-2] ^ m_W[j- 8] ^ m_W[j-10]);
         m_W[j+7] = rotl<1>(m_W[j+4] ^ m_W[j-1] ^ m_W[j- 7] ^ m_W[j- 9]);
         }

      F1(A, B, C, D, E, m_W[ 0]);   F1(E, A, B, C, D, m_W[ 1]);
      F1(D, E, A, B, C, m_W[ 2]);   F1(C, D, E, A, B, m_W[ 3]);
      F1(B, C, D, E, A, m_W[ 4]);   F1(A, B, C, D, E, m_W[ 5]);
      F1(E, A, B, C, D, m_W[ 6]);   F1(D, E, A, B, C, m_W[ 7]);
      F1(C, D, E, A, B, m_W[ 8]);   F1(B, C, D, E, A, m_W[ 9]);
      F1(A, B, C, D, E, m_W[10]);   F1(E, A, B, C, D, m_W[11]);
      F1(D, E, A, B, C, m_W[12]);   F1(C, D, E, A, B, m_W[13]);
      F1(B, C, D, E, A, m_W[14]);   F1(A, B, C, D, E, m_W[15]);
      F1(E, A, B, C, D, m_W[16]);   F1(D, E, A, B, C, m_W[17]);
      F1(C, D, E, A, B, m_W[18]);   F1(B, C, D, E, A, m_W[19]);

      F2(A, B, C, D, E, m_W[20]);   F2(E, A, B, C, D, m_W[21]);
      F2(D, E, A, B, C, m_W[22]);   F2(C, D, E, A, B, m_W[23]);
      F2(B, C, D, E, A, m_W[24]);   F2(A, B, C, D, E, m_W[25]);
      F2(E, A, B, C, D, m_W[26]);   F2(D, E, A, B, C, m_W[27]);
      F2(C, D, E, A, B, m_W[28]);   F2(B, C, D, E, A, m_W[29]);
      F2(A, B, C, D, E, m_W[30]);   F2(E, A, B, C, D, m_W[31]);
      F2(D, E, A, B, C, m_W[32]);   F2(C, D, E, A, B, m_W[33]);
      F2(B, C, D, E, A, m_W[34]);   F2(A, B, C, D, E, m_W[35]);
      F2(E, A, B, C, D, m_W[36]);   F2(D, E, A, B, C, m_W[37]);
      F2(C, D, E, A, B, m_W[38]);   F2(B, C, D, E, A, m_W[39]);

      F3(A, B, C, D, E, m_W[40]);   F3(E, A, B, C, D, m_W[41]);
      F3(D, E, A, B, C, m_W[42]);   F3(C, D, E, A, B, m_W[43]);
      F3(B, C, D, E, A, m_W[44]);   F3(A, B, C, D, E, m_W[45]);
      F3(E, A, B, C, D, m_W[46]);   F3(D, E, A, B, C, m_W[47]);
      F3(C, D, E, A, B, m_W[48]);   F3(B, C, D, E, A, m_W[49]);
      F3(A, B, C, D, E, m_W[50]);   F3(E, A, B, C, D, m_W[51]);
      F3(D, E, A, B, C, m_W[52]);   F3(C, D, E, A, B, m_W[53]);
      F3(B, C, D, E, A, m_W[54]);   F3(A, B, C, D, E, m_W[55]);
      F3(E, A, B, C, D, m_W[56]);   F3(D, E, A, B, C, m_W[57]);
      F3(C, D, E, A, B, m_W[58]);   F3(B, C, D, E, A, m_W[59]);

      F4(A, B, C, D, E, m_W[60]);   F4(E, A, B, C, D, m_W[61]);
      F4(D, E, A, B, C, m_W[62]);   F4(C, D, E, A, B, m_W[63]);
      F4(B, C, D, E, A, m_W[64]);   F4(A, B, C, D, E, m_W[65]);
      F4(E, A, B, C, D, m_W[66]);   F4(D, E, A, B, C, m_W[67]);
      F4(C, D, E, A, B, m_W[68]);   F4(B, C, D, E, A, m_W[69]);
      F4(A, B, C, D, E, m_W[70]);   F4(E, A, B, C, D, m_W[71]);
      F4(D, E, A, B, C, m_W[72]);   F4(C, D, E, A, B, m_W[73]);
      F4(B, C, D, E, A, m_W[74]);   F4(A, B, C, D, E, m_W[75]);
      F4(E, A, B, C, D, m_W[76]);   F4(D, E, A, B, C, m_W[77]);
      F4(C, D, E, A, B, m_W[78]);   F4(B, C, D, E, A, m_W[79]);

      A = (m_digest[0] += A);
      B = (m_digest[1] += B);
      C = (m_digest[2] += C);
      D = (m_digest[3] += D);
      E = (m_digest[4] += E);

      input += hash_block_size();
      }
   }

/*
* Copy out the digest
*/
void SHA_160::copy_out(uint8_t output[])
   {
   copy_out_vec_be(output, output_length(), m_digest);
   }

/*
* Clear memory of sensitive data
*/
void SHA_160::clear()
   {
   MDx_HashFunction::clear();
   zeroise(m_W);
   m_digest[0] = 0x67452301;
   m_digest[1] = 0xEFCDAB89;
   m_digest[2] = 0x98BADCFE;
   m_digest[3] = 0x10325476;
   m_digest[4] = 0xC3D2E1F0;
   }

}
/*
* SHA-{224,256}
* (C) 1999-2010,2017 Jack Lloyd
*     2007 FlexSecure GmbH
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

std::string sha256_provider()
   {
#if defined(BOTAN_HAS_SHA2_32_X86)
   if(CPUID::has_intel_sha())
      {
      return "shani";
      }
#endif

#if defined(BOTAN_HAS_SHA2_32_X86_BMI2)
   if(CPUID::has_bmi2())
      {
      return "bmi2";
      }
#endif

#if defined(BOTAN_HAS_SHA2_32_ARMV8)
   if(CPUID::has_arm_sha2())
      {
      return "armv8";
      }
#endif

   return "base";
   }

}

std::unique_ptr<HashFunction> SHA_224::copy_state() const
   {
   return std::unique_ptr<HashFunction>(new SHA_224(*this));
   }

std::unique_ptr<HashFunction> SHA_256::copy_state() const
   {
   return std::unique_ptr<HashFunction>(new SHA_256(*this));
   }

/*
* SHA-256 F1 Function
*
* Use a macro as many compilers won't inline a function this big,
* even though it is much faster if inlined.
*/
#define SHA2_32_F(A, B, C, D, E, F, G, H, M1, M2, M3, M4, magic) do {               \
   uint32_t A_rho = rotr<2>(A) ^ rotr<13>(A) ^ rotr<22>(A); \
   uint32_t E_rho = rotr<6>(E) ^ rotr<11>(E) ^ rotr<25>(E); \
   uint32_t M2_sigma = rotr<17>(M2) ^ rotr<19>(M2) ^ (M2 >> 10);    \
   uint32_t M4_sigma = rotr<7>(M4) ^ rotr<18>(M4) ^ (M4 >> 3);      \
   H += magic + E_rho + ((E & F) ^ (~E & G)) + M1;                                  \
   D += H;                                                                          \
   H += A_rho + ((A & B) | ((A | B) & C));                                          \
   M1 += M2_sigma + M3 + M4_sigma;                                                  \
   } while(0);

/*
* SHA-224 / SHA-256 compression function
*/
void SHA_256::compress_digest(secure_vector<uint32_t>& digest,
                              const uint8_t input[], size_t blocks)
   {
#if defined(BOTAN_HAS_SHA2_32_X86)
   if(CPUID::has_intel_sha())
      {
      return SHA_256::compress_digest_x86(digest, input, blocks);
      }
#endif

#if defined(BOTAN_HAS_SHA2_32_X86_BMI2)
   if(CPUID::has_bmi2())
      {
      return SHA_256::compress_digest_x86_bmi2(digest, input, blocks);
      }
#endif

#if defined(BOTAN_HAS_SHA2_32_ARMV8)
   if(CPUID::has_arm_sha2())
      {
      return SHA_256::compress_digest_armv8(digest, input, blocks);
      }
#endif

   uint32_t A = digest[0], B = digest[1], C = digest[2],
            D = digest[3], E = digest[4], F = digest[5],
            G = digest[6], H = digest[7];

   for(size_t i = 0; i != blocks; ++i)
      {
      uint32_t W00 = load_be<uint32_t>(input,  0);
      uint32_t W01 = load_be<uint32_t>(input,  1);
      uint32_t W02 = load_be<uint32_t>(input,  2);
      uint32_t W03 = load_be<uint32_t>(input,  3);
      uint32_t W04 = load_be<uint32_t>(input,  4);
      uint32_t W05 = load_be<uint32_t>(input,  5);
      uint32_t W06 = load_be<uint32_t>(input,  6);
      uint32_t W07 = load_be<uint32_t>(input,  7);
      uint32_t W08 = load_be<uint32_t>(input,  8);
      uint32_t W09 = load_be<uint32_t>(input,  9);
      uint32_t W10 = load_be<uint32_t>(input, 10);
      uint32_t W11 = load_be<uint32_t>(input, 11);
      uint32_t W12 = load_be<uint32_t>(input, 12);
      uint32_t W13 = load_be<uint32_t>(input, 13);
      uint32_t W14 = load_be<uint32_t>(input, 14);
      uint32_t W15 = load_be<uint32_t>(input, 15);

      SHA2_32_F(A, B, C, D, E, F, G, H, W00, W14, W09, W01, 0x428A2F98);
      SHA2_32_F(H, A, B, C, D, E, F, G, W01, W15, W10, W02, 0x71374491);
      SHA2_32_F(G, H, A, B, C, D, E, F, W02, W00, W11, W03, 0xB5C0FBCF);
      SHA2_32_F(F, G, H, A, B, C, D, E, W03, W01, W12, W04, 0xE9B5DBA5);
      SHA2_32_F(E, F, G, H, A, B, C, D, W04, W02, W13, W05, 0x3956C25B);
      SHA2_32_F(D, E, F, G, H, A, B, C, W05, W03, W14, W06, 0x59F111F1);
      SHA2_32_F(C, D, E, F, G, H, A, B, W06, W04, W15, W07, 0x923F82A4);
      SHA2_32_F(B, C, D, E, F, G, H, A, W07, W05, W00, W08, 0xAB1C5ED5);
      SHA2_32_F(A, B, C, D, E, F, G, H, W08, W06, W01, W09, 0xD807AA98);
      SHA2_32_F(H, A, B, C, D, E, F, G, W09, W07, W02, W10, 0x12835B01);
      SHA2_32_F(G, H, A, B, C, D, E, F, W10, W08, W03, W11, 0x243185BE);
      SHA2_32_F(F, G, H, A, B, C, D, E, W11, W09, W04, W12, 0x550C7DC3);
      SHA2_32_F(E, F, G, H, A, B, C, D, W12, W10, W05, W13, 0x72BE5D74);
      SHA2_32_F(D, E, F, G, H, A, B, C, W13, W11, W06, W14, 0x80DEB1FE);
      SHA2_32_F(C, D, E, F, G, H, A, B, W14, W12, W07, W15, 0x9BDC06A7);
      SHA2_32_F(B, C, D, E, F, G, H, A, W15, W13, W08, W00, 0xC19BF174);

      SHA2_32_F(A, B, C, D, E, F, G, H, W00, W14, W09, W01, 0xE49B69C1);
      SHA2_32_F(H, A, B, C, D, E, F, G, W01, W15, W10, W02, 0xEFBE4786);
      SHA2_32_F(G, H, A, B, C, D, E, F, W02, W00, W11, W03, 0x0FC19DC6);
      SHA2_32_F(F, G, H, A, B, C, D, E, W03, W01, W12, W04, 0x240CA1CC);
      SHA2_32_F(E, F, G, H, A, B, C, D, W04, W02, W13, W05, 0x2DE92C6F);
      SHA2_32_F(D, E, F, G, H, A, B, C, W05, W03, W14, W06, 0x4A7484AA);
      SHA2_32_F(C, D, E, F, G, H, A, B, W06, W04, W15, W07, 0x5CB0A9DC);
      SHA2_32_F(B, C, D, E, F, G, H, A, W07, W05, W00, W08, 0x76F988DA);
      SHA2_32_F(A, B, C, D, E, F, G, H, W08, W06, W01, W09, 0x983E5152);
      SHA2_32_F(H, A, B, C, D, E, F, G, W09, W07, W02, W10, 0xA831C66D);
      SHA2_32_F(G, H, A, B, C, D, E, F, W10, W08, W03, W11, 0xB00327C8);
      SHA2_32_F(F, G, H, A, B, C, D, E, W11, W09, W04, W12, 0xBF597FC7);
      SHA2_32_F(E, F, G, H, A, B, C, D, W12, W10, W05, W13, 0xC6E00BF3);
      SHA2_32_F(D, E, F, G, H, A, B, C, W13, W11, W06, W14, 0xD5A79147);
      SHA2_32_F(C, D, E, F, G, H, A, B, W14, W12, W07, W15, 0x06CA6351);
      SHA2_32_F(B, C, D, E, F, G, H, A, W15, W13, W08, W00, 0x14292967);

      SHA2_32_F(A, B, C, D, E, F, G, H, W00, W14, W09, W01, 0x27B70A85);
      SHA2_32_F(H, A, B, C, D, E, F, G, W01, W15, W10, W02, 0x2E1B2138);
      SHA2_32_F(G, H, A, B, C, D, E, F, W02, W00, W11, W03, 0x4D2C6DFC);
      SHA2_32_F(F, G, H, A, B, C, D, E, W03, W01, W12, W04, 0x53380D13);
      SHA2_32_F(E, F, G, H, A, B, C, D, W04, W02, W13, W05, 0x650A7354);
      SHA2_32_F(D, E, F, G, H, A, B, C, W05, W03, W14, W06, 0x766A0ABB);
      SHA2_32_F(C, D, E, F, G, H, A, B, W06, W04, W15, W07, 0x81C2C92E);
      SHA2_32_F(B, C, D, E, F, G, H, A, W07, W05, W00, W08, 0x92722C85);
      SHA2_32_F(A, B, C, D, E, F, G, H, W08, W06, W01, W09, 0xA2BFE8A1);
      SHA2_32_F(H, A, B, C, D, E, F, G, W09, W07, W02, W10, 0xA81A664B);
      SHA2_32_F(G, H, A, B, C, D, E, F, W10, W08, W03, W11, 0xC24B8B70);
      SHA2_32_F(F, G, H, A, B, C, D, E, W11, W09, W04, W12, 0xC76C51A3);
      SHA2_32_F(E, F, G, H, A, B, C, D, W12, W10, W05, W13, 0xD192E819);
      SHA2_32_F(D, E, F, G, H, A, B, C, W13, W11, W06, W14, 0xD6990624);
      SHA2_32_F(C, D, E, F, G, H, A, B, W14, W12, W07, W15, 0xF40E3585);
      SHA2_32_F(B, C, D, E, F, G, H, A, W15, W13, W08, W00, 0x106AA070);

      SHA2_32_F(A, B, C, D, E, F, G, H, W00, W14, W09, W01, 0x19A4C116);
      SHA2_32_F(H, A, B, C, D, E, F, G, W01, W15, W10, W02, 0x1E376C08);
      SHA2_32_F(G, H, A, B, C, D, E, F, W02, W00, W11, W03, 0x2748774C);
      SHA2_32_F(F, G, H, A, B, C, D, E, W03, W01, W12, W04, 0x34B0BCB5);
      SHA2_32_F(E, F, G, H, A, B, C, D, W04, W02, W13, W05, 0x391C0CB3);
      SHA2_32_F(D, E, F, G, H, A, B, C, W05, W03, W14, W06, 0x4ED8AA4A);
      SHA2_32_F(C, D, E, F, G, H, A, B, W06, W04, W15, W07, 0x5B9CCA4F);
      SHA2_32_F(B, C, D, E, F, G, H, A, W07, W05, W00, W08, 0x682E6FF3);
      SHA2_32_F(A, B, C, D, E, F, G, H, W08, W06, W01, W09, 0x748F82EE);
      SHA2_32_F(H, A, B, C, D, E, F, G, W09, W07, W02, W10, 0x78A5636F);
      SHA2_32_F(G, H, A, B, C, D, E, F, W10, W08, W03, W11, 0x84C87814);
      SHA2_32_F(F, G, H, A, B, C, D, E, W11, W09, W04, W12, 0x8CC70208);
      SHA2_32_F(E, F, G, H, A, B, C, D, W12, W10, W05, W13, 0x90BEFFFA);
      SHA2_32_F(D, E, F, G, H, A, B, C, W13, W11, W06, W14, 0xA4506CEB);
      SHA2_32_F(C, D, E, F, G, H, A, B, W14, W12, W07, W15, 0xBEF9A3F7);
      SHA2_32_F(B, C, D, E, F, G, H, A, W15, W13, W08, W00, 0xC67178F2);

      A = (digest[0] += A);
      B = (digest[1] += B);
      C = (digest[2] += C);
      D = (digest[3] += D);
      E = (digest[4] += E);
      F = (digest[5] += F);
      G = (digest[6] += G);
      H = (digest[7] += H);

      input += 64;
      }
   }

std::string SHA_224::provider() const
   {
   return sha256_provider();
   }

std::string SHA_256::provider() const
   {
   return sha256_provider();
   }

/*
* SHA-224 compression function
*/
void SHA_224::compress_n(const uint8_t input[], size_t blocks)
   {
   SHA_256::compress_digest(m_digest, input, blocks);
   }

/*
* Copy out the digest
*/
void SHA_224::copy_out(uint8_t output[])
   {
   copy_out_vec_be(output, output_length(), m_digest);
   }

/*
* Clear memory of sensitive data
*/
void SHA_224::clear()
   {
   MDx_HashFunction::clear();
   m_digest[0] = 0xC1059ED8;
   m_digest[1] = 0x367CD507;
   m_digest[2] = 0x3070DD17;
   m_digest[3] = 0xF70E5939;
   m_digest[4] = 0xFFC00B31;
   m_digest[5] = 0x68581511;
   m_digest[6] = 0x64F98FA7;
   m_digest[7] = 0xBEFA4FA4;
   }

/*
* SHA-256 compression function
*/
void SHA_256::compress_n(const uint8_t input[], size_t blocks)
   {
   SHA_256::compress_digest(m_digest, input, blocks);
   }

/*
* Copy out the digest
*/
void SHA_256::copy_out(uint8_t output[])
   {
   copy_out_vec_be(output, output_length(), m_digest);
   }

/*
* Clear memory of sensitive data
*/
void SHA_256::clear()
   {
   MDx_HashFunction::clear();
   m_digest[0] = 0x6A09E667;
   m_digest[1] = 0xBB67AE85;
   m_digest[2] = 0x3C6EF372;
   m_digest[3] = 0xA54FF53A;
   m_digest[4] = 0x510E527F;
   m_digest[5] = 0x9B05688C;
   m_digest[6] = 0x1F83D9AB;
   m_digest[7] = 0x5BE0CD19;
   }

}
/*
* Runtime assertion checking
* (C) 2010,2012,2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

void throw_invalid_argument(const char* message,
                            const char* func,
                            const char* file)
   {
   std::ostringstream format;
   format << message << " in " << func << ":" << file;
   throw Invalid_Argument(format.str());
   }

void throw_invalid_state(const char* expr,
                         const char* func,
                         const char* file)
   {
   std::ostringstream format;
   format << "Invalid state: " << expr << " was false in " << func << ":" << file;
   throw Invalid_State(format.str());
   }

void assertion_failure(const char* expr_str,
                       const char* assertion_made,
                       const char* func,
                       const char* file,
                       int line)
   {
   std::ostringstream format;

   format << "False assertion ";

   if(assertion_made && assertion_made[0] != 0)
      format << "'" << assertion_made << "' (expression " << expr_str << ") ";
   else
      format << expr_str << " ";

   if(func)
      format << "in " << func << " ";

   format << "@" << file << ":" << line;

   throw Internal_Error(format.str());
   }

}
/*
* Calendar Functions
* (C) 1999-2010,2017 Jack Lloyd
* (C) 2015 Simon Warta (Kullo GmbH)
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <ctime>
#include <stdlib.h>

namespace Botan {

namespace {

std::tm do_gmtime(std::time_t time_val)
   {
   std::tm tm;

#if defined(BOTAN_TARGET_OS_HAS_WIN32)
   ::gmtime_s(&tm, &time_val); // Windows
#elif defined(BOTAN_TARGET_OS_HAS_POSIX1)
   ::gmtime_r(&time_val, &tm); // Unix/SUSv2
#else
   std::tm* tm_p = std::gmtime(&time_val);
   if (tm_p == nullptr)
      throw Encoding_Error("time_t_to_tm could not convert");
   tm = *tm_p;
#endif

   return tm;
   }

/*
Portable replacement for timegm, _mkgmtime, etc

Algorithm due to Howard Hinnant

See https://howardhinnant.github.io/date_algorithms.html#days_from_civil
for details and explaination. The code is slightly simplified by our assumption
that the date is at least 1970, which is sufficient for our purposes.
*/
size_t days_since_epoch(uint32_t year, uint32_t month, uint32_t day)
   {
   if(month <= 2)
      year -= 1;
   const uint32_t era = year / 400;
   const uint32_t yoe = year - era * 400;      // [0, 399]
   const uint32_t doy = (153*(month + (month > 2 ? -3 : 9)) + 2)/5 + day-1;  // [0, 365]
   const uint32_t doe = yoe * 365 + yoe/4 - yoe/100 + doy;         // [0, 146096]
   return era * 146097 + doe - 719468;
   }

}

std::chrono::system_clock::time_point calendar_point::to_std_timepoint() const
   {
   if(get_year() < 1970)
      throw Invalid_Argument("calendar_point::to_std_timepoint() does not support years before 1970");

   // 32 bit time_t ends at January 19, 2038
   // https://msdn.microsoft.com/en-us/library/2093ets1.aspx
   // Throw after 2037 if 32 bit time_t is used

   BOTAN_IF_CONSTEXPR(sizeof(std::time_t) == 4)
      {
      if(get_year() > 2037)
         {
         throw Invalid_Argument("calendar_point::to_std_timepoint() does not support years after 2037 on this system");
         }
      }

   // This upper bound is completely arbitrary
   if(get_year() >= 2400)
      {
      throw Invalid_Argument("calendar_point::to_std_timepoint() does not support years after 2400");
      }

   const uint64_t seconds_64 = (days_since_epoch(get_year(), get_month(), get_day()) * 86400) +
                                (get_hour() * 60 * 60) + (get_minutes() * 60) + get_seconds();

   const time_t seconds_time_t = static_cast<time_t>(seconds_64);

   if(seconds_64 - seconds_time_t != 0)
      {
      throw Invalid_Argument("calendar_point::to_std_timepoint time_t overflow");
      }

   return std::chrono::system_clock::from_time_t(seconds_time_t);
   }

std::string calendar_point::to_string() const
   {
   // desired format: <YYYY>-<MM>-<dd>T<HH>:<mm>:<ss>
   std::stringstream output;
   output << std::setfill('0')
          << std::setw(4) << get_year() << "-"
          << std::setw(2) << get_month() << "-"
          << std::setw(2) << get_day() << "T"
          << std::setw(2) << get_hour() << ":"
          << std::setw(2) << get_minutes() << ":"
          << std::setw(2) << get_seconds();
   return output.str();
   }


calendar_point calendar_value(
   const std::chrono::system_clock::time_point& time_point)
   {
   std::tm tm = do_gmtime(std::chrono::system_clock::to_time_t(time_point));

   return calendar_point(tm.tm_year + 1900,
                         tm.tm_mon + 1,
                         tm.tm_mday,
                         tm.tm_hour,
                         tm.tm_min,
                         tm.tm_sec);
   }

}
/*
* Character Set Handling
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace {

void append_utf8_for(std::string& s, uint32_t c)
   {
   if(c >= 0xD800 && c < 0xE000)
      throw Decoding_Error("Invalid Unicode character");

   if(c <= 0x7F)
      {
      const uint8_t b0 = static_cast<uint8_t>(c);
      s.push_back(static_cast<char>(b0));
      }
   else if(c <= 0x7FF)
      {
      const uint8_t b0 = 0xC0 | static_cast<uint8_t>(c >> 6);
      const uint8_t b1 = 0x80 | static_cast<uint8_t>(c & 0x3F);
      s.push_back(static_cast<char>(b0));
      s.push_back(static_cast<char>(b1));
      }
   else if(c <= 0xFFFF)
      {
      const uint8_t b0 = 0xE0 | static_cast<uint8_t>(c >> 12);
      const uint8_t b1 = 0x80 | static_cast<uint8_t>((c >> 6) & 0x3F);
      const uint8_t b2 = 0x80 | static_cast<uint8_t>(c & 0x3F);
      s.push_back(static_cast<char>(b0));
      s.push_back(static_cast<char>(b1));
      s.push_back(static_cast<char>(b2));
      }
   else if(c <= 0x10FFFF)
      {
      const uint8_t b0 = 0xF0 | static_cast<uint8_t>(c >> 18);
      const uint8_t b1 = 0x80 | static_cast<uint8_t>((c >> 12) & 0x3F);
      const uint8_t b2 = 0x80 | static_cast<uint8_t>((c >> 6) & 0x3F);
      const uint8_t b3 = 0x80 | static_cast<uint8_t>(c & 0x3F);
      s.push_back(static_cast<char>(b0));
      s.push_back(static_cast<char>(b1));
      s.push_back(static_cast<char>(b2));
      s.push_back(static_cast<char>(b3));
      }
   else
      throw Decoding_Error("Invalid Unicode character");

   }

}

std::string ucs2_to_utf8(const uint8_t ucs2[], size_t len)
   {
   if(len % 2 != 0)
      throw Decoding_Error("Invalid length for UCS-2 string");

   const size_t chars = len / 2;

   std::string s;
   for(size_t i = 0; i != chars; ++i)
      {
      const uint16_t c = load_be<uint16_t>(ucs2, i);
      append_utf8_for(s, c);
      }

   return s;
   }

std::string ucs4_to_utf8(const uint8_t ucs4[], size_t len)
   {
   if(len % 4 != 0)
      throw Decoding_Error("Invalid length for UCS-4 string");

   const size_t chars = len / 4;

   std::string s;
   for(size_t i = 0; i != chars; ++i)
      {
      const uint32_t c = load_be<uint32_t>(ucs4, i);
      append_utf8_for(s, c);
      }

   return s;
   }

/*
* Convert from UTF-8 to ISO 8859-1
*/
std::string utf8_to_latin1(const std::string& utf8)
   {
   std::string iso8859;

   size_t position = 0;
   while(position != utf8.size())
      {
      const uint8_t c1 = static_cast<uint8_t>(utf8[position++]);

      if(c1 <= 0x7F)
         {
         iso8859 += static_cast<char>(c1);
         }
      else if(c1 >= 0xC0 && c1 <= 0xC7)
         {
         if(position == utf8.size())
            throw Decoding_Error("UTF-8: sequence truncated");

         const uint8_t c2 = static_cast<uint8_t>(utf8[position++]);
         const uint8_t iso_char = ((c1 & 0x07) << 6) | (c2 & 0x3F);

         if(iso_char <= 0x7F)
            throw Decoding_Error("UTF-8: sequence longer than needed");

         iso8859 += static_cast<char>(iso_char);
         }
      else
         throw Decoding_Error("UTF-8: Unicode chars not in Latin1 used");
      }

   return iso8859;
   }

namespace Charset {

namespace {

/*
* Convert from UCS-2 to ISO 8859-1
*/
std::string ucs2_to_latin1(const std::string& ucs2)
   {
   if(ucs2.size() % 2 == 1)
      throw Decoding_Error("UCS-2 string has an odd number of bytes");

   std::string latin1;

   for(size_t i = 0; i != ucs2.size(); i += 2)
      {
      const uint8_t c1 = ucs2[i];
      const uint8_t c2 = ucs2[i+1];

      if(c1 != 0)
         throw Decoding_Error("UCS-2 has non-Latin1 characters");

      latin1 += static_cast<char>(c2);
      }

   return latin1;
   }

/*
* Convert from ISO 8859-1 to UTF-8
*/
std::string latin1_to_utf8(const std::string& iso8859)
   {
   std::string utf8;
   for(size_t i = 0; i != iso8859.size(); ++i)
      {
      const uint8_t c = static_cast<uint8_t>(iso8859[i]);

      if(c <= 0x7F)
         utf8 += static_cast<char>(c);
      else
         {
         utf8 += static_cast<char>((0xC0 | (c >> 6)));
         utf8 += static_cast<char>((0x80 | (c & 0x3F)));
         }
      }
   return utf8;
   }

}

/*
* Perform character set transcoding
*/
std::string transcode(const std::string& str,
                      Character_Set to, Character_Set from)
   {
   if(to == LOCAL_CHARSET)
      to = LATIN1_CHARSET;
   if(from == LOCAL_CHARSET)
      from = LATIN1_CHARSET;

   if(to == from)
      return str;

   if(from == LATIN1_CHARSET && to == UTF8_CHARSET)
      return latin1_to_utf8(str);
   if(from == UTF8_CHARSET && to == LATIN1_CHARSET)
      return utf8_to_latin1(str);
   if(from == UCS2_CHARSET && to == LATIN1_CHARSET)
      return ucs2_to_latin1(str);

   throw Invalid_Argument("Unknown transcoding operation from " +
                          std::to_string(from) + " to " + std::to_string(to));
   }

/*
* Check if a character represents a digit
*/
bool is_digit(char c)
   {
   if(c == '0' || c == '1' || c == '2' || c == '3' || c == '4' ||
      c == '5' || c == '6' || c == '7' || c == '8' || c == '9')
      return true;
   return false;
   }

/*
* Check if a character represents whitespace
*/
bool is_space(char c)
   {
   if(c == ' ' || c == '\t' || c == '\n' || c == '\r')
      return true;
   return false;
   }

/*
* Convert a character to a digit
*/
uint8_t char2digit(char c)
   {
   switch(c)
      {
      case '0': return 0;
      case '1': return 1;
      case '2': return 2;
      case '3': return 3;
      case '4': return 4;
      case '5': return 5;
      case '6': return 6;
      case '7': return 7;
      case '8': return 8;
      case '9': return 9;
      }

   throw Invalid_Argument("char2digit: Input is not a digit character");
   }

/*
* Convert a digit to a character
*/
char digit2char(uint8_t b)
   {
   switch(b)
      {
      case 0: return '0';
      case 1: return '1';
      case 2: return '2';
      case 3: return '3';
      case 4: return '4';
      case 5: return '5';
      case 6: return '6';
      case 7: return '7';
      case 8: return '8';
      case 9: return '9';
      }

   throw Invalid_Argument("digit2char: Input is not a digit");
   }

/*
* Case-insensitive character comparison
*/
bool caseless_cmp(char a, char b)
   {
   return (std::tolower(static_cast<unsigned char>(a)) ==
           std::tolower(static_cast<unsigned char>(b)));
   }

}

}
/*
* (C) 2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

namespace CT {

secure_vector<uint8_t> copy_output(CT::Mask<uint8_t> bad_input,
                                   const uint8_t input[],
                                   size_t input_length,
                                   size_t offset)
   {
   if(input_length == 0)
      return secure_vector<uint8_t>();

   /*
   * Ensure at runtime that offset <= input_length. This is an invalid input,
   * but we can't throw without using the poisoned value. Instead, if it happens,
   * set offset to be equal to the input length (so output_bytes becomes 0 and
   * the returned vector is empty)
   */
   const auto valid_offset = CT::Mask<size_t>::is_lte(offset, input_length);
   offset = valid_offset.select(offset, input_length);

   const size_t output_bytes = input_length - offset;

   secure_vector<uint8_t> output(input_length);

   /*
   Move the desired output bytes to the front using a slow (O^n)
   but constant time loop that does not leak the value of the offset
   */
   for(size_t i = 0; i != input_length; ++i)
      {
      /*
      start index from i rather than 0 since we know j must be >= i + offset
      to have any effect, and starting from i does not reveal information
      */
      for(size_t j = i; j != input_length; ++j)
         {
         const uint8_t b = input[j];
         const auto is_eq = CT::Mask<size_t>::is_equal(j, offset + i);
         output[i] |= is_eq.if_set_return(b);
         }
      }

   bad_input.if_set_zero_out(output.data(), output.size());

   CT::unpoison(output.data(), output.size());
   CT::unpoison(output_bytes);

   /*
   This is potentially not const time, depending on how std::vector is
   implemented. But since we are always reducing length, it should
   just amount to setting the member var holding the length.
   */
   output.resize(output_bytes);
   return output;
   }

secure_vector<uint8_t> strip_leading_zeros(const uint8_t in[], size_t length)
   {
   size_t leading_zeros = 0;

   auto only_zeros = Mask<uint8_t>::set();

   for(size_t i = 0; i != length; ++i)
      {
      only_zeros &= CT::Mask<uint8_t>::is_zero(in[i]);
      leading_zeros += only_zeros.if_set_return(1);
      }

   return copy_output(CT::Mask<uint8_t>::cleared(), in, length, leading_zeros);
   }

}

}
/*
* DataSource
* (C) 1999-2007 Jack Lloyd
*     2005 Matthew Gregan
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#if defined(BOTAN_TARGET_OS_HAS_FILESYSTEM)
  #include <fstream>
#endif

namespace Botan {

/*
* Read a single byte from the DataSource
*/
size_t DataSource::read_byte(uint8_t& out)
   {
   return read(&out, 1);
   }

/*
* Peek a single byte from the DataSource
*/
size_t DataSource::peek_byte(uint8_t& out) const
   {
   return peek(&out, 1, 0);
   }

/*
* Discard the next N bytes of the data
*/
size_t DataSource::discard_next(size_t n)
   {
   uint8_t buf[64] = { 0 };
   size_t discarded = 0;

   while(n)
      {
      const size_t got = this->read(buf, std::min(n, sizeof(buf)));
      discarded += got;
      n -= got;

      if(got == 0)
         break;
      }

   return discarded;
   }

/*
* Read from a memory buffer
*/
size_t DataSource_Memory::read(uint8_t out[], size_t length)
   {
   const size_t got = std::min<size_t>(m_source.size() - m_offset, length);
   copy_mem(out, m_source.data() + m_offset, got);
   m_offset += got;
   return got;
   }

bool DataSource_Memory::check_available(size_t n)
   {
   return (n <= (m_source.size() - m_offset));
   }

/*
* Peek into a memory buffer
*/
size_t DataSource_Memory::peek(uint8_t out[], size_t length,
                               size_t peek_offset) const
   {
   const size_t bytes_left = m_source.size() - m_offset;
   if(peek_offset >= bytes_left) return 0;

   const size_t got = std::min(bytes_left - peek_offset, length);
   copy_mem(out, &m_source[m_offset + peek_offset], got);
   return got;
   }

/*
* Check if the memory buffer is empty
*/
bool DataSource_Memory::end_of_data() const
   {
   return (m_offset == m_source.size());
   }

/*
* DataSource_Memory Constructor
*/
DataSource_Memory::DataSource_Memory(const std::string& in) :
   m_source(cast_char_ptr_to_uint8(in.data()),
            cast_char_ptr_to_uint8(in.data()) + in.length()),
   m_offset(0)
   {
   }

/*
* Read from a stream
*/
size_t DataSource_Stream::read(uint8_t out[], size_t length)
   {
   m_source.read(cast_uint8_ptr_to_char(out), length);
   if(m_source.bad())
      throw Stream_IO_Error("DataSource_Stream::read: Source failure");

   const size_t got = static_cast<size_t>(m_source.gcount());
   m_total_read += got;
   return got;
   }

bool DataSource_Stream::check_available(size_t n)
   {
   const std::streampos orig_pos = m_source.tellg();
   m_source.seekg(0, std::ios::end);
   const size_t avail = static_cast<size_t>(m_source.tellg() - orig_pos);
   m_source.seekg(orig_pos);
   return (avail >= n);
   }

/*
* Peek into a stream
*/
size_t DataSource_Stream::peek(uint8_t out[], size_t length, size_t offset) const
   {
   if(end_of_data())
      throw Invalid_State("DataSource_Stream: Cannot peek when out of data");

   size_t got = 0;

   if(offset)
      {
      secure_vector<uint8_t> buf(offset);
      m_source.read(cast_uint8_ptr_to_char(buf.data()), buf.size());
      if(m_source.bad())
         throw Stream_IO_Error("DataSource_Stream::peek: Source failure");
      got = static_cast<size_t>(m_source.gcount());
      }

   if(got == offset)
      {
      m_source.read(cast_uint8_ptr_to_char(out), length);
      if(m_source.bad())
         throw Stream_IO_Error("DataSource_Stream::peek: Source failure");
      got = static_cast<size_t>(m_source.gcount());
      }

   if(m_source.eof())
      m_source.clear();
   m_source.seekg(m_total_read, std::ios::beg);

   return got;
   }

/*
* Check if the stream is empty or in error
*/
bool DataSource_Stream::end_of_data() const
   {
   return (!m_source.good());
   }

/*
* Return a human-readable ID for this stream
*/
std::string DataSource_Stream::id() const
   {
   return m_identifier;
   }

#if defined(BOTAN_TARGET_OS_HAS_FILESYSTEM)

/*
* DataSource_Stream Constructor
*/
DataSource_Stream::DataSource_Stream(const std::string& path,
                                     bool use_binary) :
   m_identifier(path),
   m_source_memory(new std::ifstream(path, use_binary ? std::ios::binary : std::ios::in)),
   m_source(*m_source_memory),
   m_total_read(0)
   {
   if(!m_source.good())
      {
      throw Stream_IO_Error("DataSource: Failure opening file " + path);
      }
   }

#endif

/*
* DataSource_Stream Constructor
*/
DataSource_Stream::DataSource_Stream(std::istream& in,
                                     const std::string& name) :
   m_identifier(name),
   m_source(in),
   m_total_read(0)
   {
   }

DataSource_Stream::~DataSource_Stream()
   {
   // for ~unique_ptr
   }

}
/*
* (C) 2017 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

std::string to_string(ErrorType type)
   {
   switch(type)
      {
      case ErrorType::Unknown:
         return "Unknown";
      case ErrorType::SystemError:
         return "SystemError";
      case ErrorType::NotImplemented:
         return "NotImplemented";
      case ErrorType::OutOfMemory:
         return "OutOfMemory";
      case ErrorType::InternalError:
         return "InternalError";
      case ErrorType::IoError:
         return "IoError";
      case ErrorType::InvalidObjectState :
         return "InvalidObjectState";
      case ErrorType::KeyNotSet:
         return "KeyNotSet";
      case ErrorType::InvalidArgument:
         return "InvalidArgument";
      case ErrorType::InvalidKeyLength:
         return "InvalidKeyLength";
      case ErrorType::InvalidNonceLength:
         return "InvalidNonceLength";
      case ErrorType::LookupError:
         return "LookupError";
      case ErrorType::EncodingFailure:
         return "EncodingFailure";
      case ErrorType::DecodingFailure:
         return "DecodingFailure";
      case ErrorType::TLSError:
         return "TLSError";
      case ErrorType::HttpError:
         return "HttpError";
      case ErrorType::InvalidTag:
         return "InvalidTag";
      case ErrorType::RoughtimeError:
         return "RoughtimeError";
      case ErrorType::OpenSSLError :
         return "OpenSSLError";
      case ErrorType::CommonCryptoError:
         return "CommonCryptoError";
      case ErrorType::Pkcs11Error:
         return "Pkcs11Error";
      case ErrorType::TPMError:
         return "TPMError";
      case ErrorType::DatabaseError:
         return "DatabaseError";
      case ErrorType::ZlibError :
         return "ZlibError";
      case ErrorType::Bzip2Error:
         return "Bzip2Error" ;
      case ErrorType::LzmaError:
         return "LzmaError";
      }

   // No default case in above switch so compiler warns
   return "Unrecognized Botan error";
   }

Exception::Exception(const std::string& msg) : m_msg(msg)
   {}

Exception::Exception(const std::string& msg, const std::exception& e) :
   m_msg(msg + " failed with " + std::string(e.what()))
   {}

Exception::Exception(const char* prefix, const std::string& msg) :
   m_msg(std::string(prefix) + " " + msg)
   {}

Invalid_Argument::Invalid_Argument(const std::string& msg) :
   Exception(msg)
   {}

Invalid_Argument::Invalid_Argument(const std::string& msg, const std::string& where) :
   Exception(msg + " in " + where)
   {}

Invalid_Argument::Invalid_Argument(const std::string& msg, const std::exception& e) :
   Exception(msg, e) {}

Lookup_Error::Lookup_Error(const std::string& type,
                           const std::string& algo,
                           const std::string& provider) :
   Exception("Unavailable " + type + " " + algo +
             (provider.empty() ? std::string("") : (" for provider " + provider)))
   {}

Internal_Error::Internal_Error(const std::string& err) :
   Exception("Internal error: " + err)
   {}

Invalid_Key_Length::Invalid_Key_Length(const std::string& name, size_t length) :
   Invalid_Argument(name + " cannot accept a key of length " +
                    std::to_string(length))
   {}

Invalid_IV_Length::Invalid_IV_Length(const std::string& mode, size_t bad_len) :
   Invalid_Argument("IV length " + std::to_string(bad_len) +
                    " is invalid for " + mode)
   {}

Key_Not_Set::Key_Not_Set(const std::string& algo) :
   Invalid_State("Key not set in " + algo)
   {}

Policy_Violation::Policy_Violation(const std::string& err) :
   Invalid_State("Policy violation: " + err) {}

PRNG_Unseeded::PRNG_Unseeded(const std::string& algo) :
   Invalid_State("PRNG not seeded: " + algo)
   {}

Algorithm_Not_Found::Algorithm_Not_Found(const std::string& name) :
   Lookup_Error("Could not find any algorithm named \"" + name + "\"")
   {}

No_Provider_Found::No_Provider_Found(const std::string& name) :
   Exception("Could not find any provider for algorithm named \"" + name + "\"")
   {}

Provider_Not_Found::Provider_Not_Found(const std::string& algo, const std::string& provider) :
   Lookup_Error("Could not find provider '" + provider + "' for " + algo)
   {}

Invalid_Algorithm_Name::Invalid_Algorithm_Name(const std::string& name):
   Invalid_Argument("Invalid algorithm name: " + name)
   {}

Encoding_Error::Encoding_Error(const std::string& name) :
   Invalid_Argument("Encoding error: " + name)
   {}

Decoding_Error::Decoding_Error(const std::string& name) :
   Invalid_Argument(name)
   {}

Decoding_Error::Decoding_Error(const std::string& msg, const std::exception& e) :
   Invalid_Argument(msg, e)
   {}

Decoding_Error::Decoding_Error(const std::string& name, const char* exception_message) :
   Invalid_Argument(name + " failed with exception " + exception_message) {}

Invalid_Authentication_Tag::Invalid_Authentication_Tag(const std::string& msg) :
   Exception("Invalid authentication tag: " + msg)
   {}

Invalid_OID::Invalid_OID(const std::string& oid) :
   Decoding_Error("Invalid ASN.1 OID: " + oid)
   {}

Stream_IO_Error::Stream_IO_Error(const std::string& err) :
   Exception("I/O error: " + err)
   {}

System_Error::System_Error(const std::string& msg, int err_code) :
   Exception(msg + " error code " + std::to_string(err_code)),
   m_error_code(err_code)
   {}

Self_Test_Failure::Self_Test_Failure(const std::string& err) :
   Internal_Error("Self test failed: " + err)
   {}

Not_Implemented::Not_Implemented(const std::string& err) :
   Exception("Not implemented", err)
   {}

}
/*
* (C) 2015,2017,2019 Jack Lloyd
* (C) 2015 Simon Warta (Kullo GmbH)
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <deque>
#include <memory>

#if defined(BOTAN_TARGET_OS_HAS_POSIX1)
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <dirent.h>
  #include <functional>
#elif defined(BOTAN_TARGET_OS_HAS_WIN32)
  #define NOMINMAX 1
  #define _WINSOCKAPI_ // stop windows.h including winsock.h
  #include <windows.h>
#endif

namespace Botan {

namespace {

#if defined(BOTAN_TARGET_OS_HAS_POSIX1)

std::vector<std::string> impl_readdir(const std::string& dir_path)
   {
   std::vector<std::string> out;
   std::deque<std::string> dir_list;
   dir_list.push_back(dir_path);

   while(!dir_list.empty())
      {
      const std::string cur_path = dir_list[0];
      dir_list.pop_front();

      std::unique_ptr<DIR, std::function<int (DIR*)>> dir(::opendir(cur_path.c_str()), ::closedir);

      if(dir)
         {
         while(struct dirent* dirent = ::readdir(dir.get()))
            {
            const std::string filename = dirent->d_name;
            if(filename == "." || filename == "..")
               continue;
            const std::string full_path = cur_path + "/" + filename;

            struct stat stat_buf;

            if(::stat(full_path.c_str(), &stat_buf) == -1)
               continue;

            if(S_ISDIR(stat_buf.st_mode))
               dir_list.push_back(full_path);
            else if(S_ISREG(stat_buf.st_mode))
               out.push_back(full_path);
            }
         }
      }

   return out;
   }

#elif defined(BOTAN_TARGET_OS_HAS_WIN32)

std::vector<std::string> impl_win32(const std::string& dir_path)
   {
   std::vector<std::string> out;
   std::deque<std::string> dir_list;
   dir_list.push_back(dir_path);

   while(!dir_list.empty())
      {
      const std::string cur_path = dir_list[0];
      dir_list.pop_front();

      WIN32_FIND_DATAA find_data;
      HANDLE dir = ::FindFirstFileA((cur_path + "/*").c_str(), &find_data);

      if(dir != INVALID_HANDLE_VALUE)
         {
         do
            {
            const std::string filename = find_data.cFileName;
            if(filename == "." || filename == "..")
               continue;
            const std::string full_path = cur_path + "/" + filename;

            if(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
               {
               dir_list.push_back(full_path);
               }
            else
               {
               out.push_back(full_path);
               }
            }
         while(::FindNextFileA(dir, &find_data));
         }

      ::FindClose(dir);
      }

   return out;
}
#endif

}

bool has_filesystem_impl()
   {
#if defined(BOTAN_TARGET_OS_HAS_POSIX1)
   return true;
#elif defined(BOTAN_TARGET_OS_HAS_WIN32)
   return true;
#else
   return false;
#endif
   }

std::vector<std::string> get_files_recursive(const std::string& dir)
   {
   std::vector<std::string> files;

#if defined(BOTAN_TARGET_OS_HAS_POSIX1)
   files = impl_readdir(dir);
#elif defined(BOTAN_TARGET_OS_HAS_WIN32)
   files = impl_win32(dir);
#else
   BOTAN_UNUSED(dir);
   throw No_Filesystem_Access();
#endif

   std::sort(files.begin(), files.end());

   return files;
   }

}
/*
* (C) 2017 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <cstdlib>
#include <new>

#if defined(BOTAN_HAS_LOCKING_ALLOCATOR)
#endif

namespace Botan {

BOTAN_MALLOC_FN void* allocate_memory(size_t elems, size_t elem_size)
   {
   if(elems == 0 || elem_size == 0)
      return nullptr;

#if defined(BOTAN_HAS_LOCKING_ALLOCATOR)
   if(void* p = mlock_allocator::instance().allocate(elems, elem_size))
      return p;
#endif

   void* ptr = std::calloc(elems, elem_size);
   if(!ptr)
      throw std::bad_alloc();
   return ptr;
   }

void deallocate_memory(void* p, size_t elems, size_t elem_size)
   {
   if(p == nullptr)
      return;

   secure_scrub_memory(p, elems * elem_size);

#if defined(BOTAN_HAS_LOCKING_ALLOCATOR)
   if(mlock_allocator::instance().deallocate(p, elems, elem_size))
      return;
#endif

   std::free(p);
   }

void initialize_allocator()
   {
#if defined(BOTAN_HAS_LOCKING_ALLOCATOR)
   mlock_allocator::instance();
#endif
   }

uint8_t ct_compare_u8(const uint8_t x[],
                      const uint8_t y[],
                      size_t len)
   {
   volatile uint8_t difference = 0;

   for(size_t i = 0; i != len; ++i)
      difference |= (x[i] ^ y[i]);

   return CT::Mask<uint8_t>::is_zero(difference).value();
   }

}
/*
* OS and machine specific utility functions
* (C) 2015,2016,2017,2018 Jack Lloyd
* (C) 2016 Daniel Neus
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


#include <chrono>

#if defined(BOTAN_TARGET_OS_HAS_THREADS)
  #include <thread>
#endif

#if defined(BOTAN_TARGET_OS_HAS_EXPLICIT_BZERO)
  #include <string.h>
#endif

#if defined(BOTAN_TARGET_OS_HAS_POSIX1)
  #include <sys/types.h>
  #include <sys/resource.h>
  #include <sys/mman.h>
  #include <signal.h>
  #include <setjmp.h>
  #include <unistd.h>
  #include <errno.h>
  #include <termios.h>
  #undef B0
#endif

#if defined(BOTAN_TARGET_OS_IS_EMSCRIPTEN)
  #include <emscripten/emscripten.h>
#endif

#if defined(BOTAN_TARGET_OS_HAS_GETAUXVAL) || defined(BOTAN_TARGET_OS_IS_ANDROID) || \
  defined(BOTAN_TARGET_OS_HAS_ELF_AUX_INFO)
  #include <sys/auxv.h>
#endif

#if defined(BOTAN_TARGET_OS_HAS_WIN32)
  #define NOMINMAX 1
  #define _WINSOCKAPI_ // stop windows.h including winsock.h
  #include <windows.h>
#endif

#if defined(BOTAN_TARGET_OS_IS_ANDROID)
  #include <elf.h>
  extern "C" char **environ;
#endif

#if defined(BOTAN_TARGET_OS_IS_IOS) || defined(BOTAN_TARGET_OS_IS_MACOS)
  #include <mach/vm_statistics.h>
#endif

namespace Botan {

// Not defined in OS namespace for historical reasons
void secure_scrub_memory(void* ptr, size_t n)
   {
#if defined(BOTAN_TARGET_OS_HAS_RTLSECUREZEROMEMORY)
   ::RtlSecureZeroMemory(ptr, n);

#elif defined(BOTAN_TARGET_OS_HAS_EXPLICIT_BZERO)
   ::explicit_bzero(ptr, n);

#elif defined(BOTAN_TARGET_OS_HAS_EXPLICIT_MEMSET)
   (void)::explicit_memset(ptr, 0, n);

#elif defined(BOTAN_USE_VOLATILE_MEMSET_FOR_ZERO) && (BOTAN_USE_VOLATILE_MEMSET_FOR_ZERO == 1)
   /*
   Call memset through a static volatile pointer, which the compiler
   should not elide. This construct should be safe in conforming
   compilers, but who knows. I did confirm that on x86-64 GCC 6.1 and
   Clang 3.8 both create code that saves the memset address in the
   data segment and unconditionally loads and jumps to that address.
   */
   static void* (*const volatile memset_ptr)(void*, int, size_t) = std::memset;
   (memset_ptr)(ptr, 0, n);
#else

   volatile uint8_t* p = reinterpret_cast<volatile uint8_t*>(ptr);

   for(size_t i = 0; i != n; ++i)
      p[i] = 0;
#endif
   }

uint32_t OS::get_process_id()
   {
#if defined(BOTAN_TARGET_OS_HAS_POSIX1)
   return ::getpid();
#elif defined(BOTAN_TARGET_OS_HAS_WIN32)
   return ::GetCurrentProcessId();
#elif defined(BOTAN_TARGET_OS_IS_INCLUDEOS) || defined(BOTAN_TARGET_OS_IS_LLVM) || defined(BOTAN_TARGET_OS_IS_NONE)
   return 0; // truly no meaningful value
#else
   #error "Missing get_process_id"
#endif
   }

unsigned long OS::get_auxval(unsigned long id)
   {
#if defined(BOTAN_TARGET_OS_HAS_GETAUXVAL)
   return ::getauxval(id);
#elif defined(BOTAN_TARGET_OS_IS_ANDROID) && defined(BOTAN_TARGET_ARCH_IS_ARM32)

   if(id == 0)
      return 0;

   char **p = environ;

   while(*p++ != nullptr)
      ;

   Elf32_auxv_t *e = reinterpret_cast<Elf32_auxv_t*>(p);

   while(e != nullptr)
      {
      if(e->a_type == id)
         return e->a_un.a_val;
      e++;
      }

   return 0;
#elif defined(BOTAN_TARGET_OS_HAS_ELF_AUX_INFO)
   unsigned long auxinfo = 0;
   ::elf_aux_info(id, &auxinfo, sizeof(auxinfo));
   return auxinfo;
#else
   BOTAN_UNUSED(id);
   return 0;
#endif
   }

bool OS::running_in_privileged_state()
   {
#if defined(AT_SECURE)
   return OS::get_auxval(AT_SECURE) != 0;
#elif defined(BOTAN_TARGET_OS_HAS_POSIX1)
   return (::getuid() != ::geteuid()) || (::getgid() != ::getegid());
#else
   return false;
#endif
   }

uint64_t OS::get_cpu_cycle_counter()
   {
   uint64_t rtc = 0;

#if defined(BOTAN_TARGET_OS_HAS_WIN32)
   LARGE_INTEGER tv;
   ::QueryPerformanceCounter(&tv);
   rtc = tv.QuadPart;

#elif defined(BOTAN_USE_GCC_INLINE_ASM)

#if defined(BOTAN_TARGET_CPU_IS_X86_FAMILY)

   if(CPUID::has_rdtsc())
      {
      uint32_t rtc_low = 0, rtc_high = 0;
      asm volatile("rdtsc" : "=d" (rtc_high), "=a" (rtc_low));
      rtc = (static_cast<uint64_t>(rtc_high) << 32) | rtc_low;
      }

#elif defined(BOTAN_TARGET_ARCH_IS_PPC64)

   for(;;)
      {
      uint32_t rtc_low = 0, rtc_high = 0, rtc_high2 = 0;
      asm volatile("mftbu %0" : "=r" (rtc_high));
      asm volatile("mftb %0" : "=r" (rtc_low));
      asm volatile("mftbu %0" : "=r" (rtc_high2));

      if(rtc_high == rtc_high2)
         {
         rtc = (static_cast<uint64_t>(rtc_high) << 32) | rtc_low;
         break;
         }
      }

#elif defined(BOTAN_TARGET_ARCH_IS_ALPHA)
   asm volatile("rpcc %0" : "=r" (rtc));

   // OpenBSD does not trap access to the %tick register
#elif defined(BOTAN_TARGET_ARCH_IS_SPARC64) && !defined(BOTAN_TARGET_OS_IS_OPENBSD)
   asm volatile("rd %%tick, %0" : "=r" (rtc));

#elif defined(BOTAN_TARGET_ARCH_IS_IA64)
   asm volatile("mov %0=ar.itc" : "=r" (rtc));

#elif defined(BOTAN_TARGET_ARCH_IS_S390X)
   asm volatile("stck 0(%0)" : : "a" (&rtc) : "memory", "cc");

#elif defined(BOTAN_TARGET_ARCH_IS_HPPA)
   asm volatile("mfctl 16,%0" : "=r" (rtc)); // 64-bit only?

#else
   //#warning "OS::get_cpu_cycle_counter not implemented"
#endif

#endif

   return rtc;
   }

size_t OS::get_cpu_total()
   {
#if defined(BOTAN_TARGET_OS_HAS_POSIX1) && defined(_SC_NPROCESSORS_CONF)
   const long res = ::sysconf(_SC_NPROCESSORS_CONF);
   if(res > 0)
      return static_cast<size_t>(res);
#endif

#if defined(BOTAN_TARGET_OS_HAS_THREADS)
   return static_cast<size_t>(std::thread::hardware_concurrency());
#else
   return 1;
#endif
   }

size_t OS::get_cpu_available()
   {
#if defined(BOTAN_TARGET_OS_HAS_POSIX1) && defined(_SC_NPROCESSORS_ONLN)
   const long res = ::sysconf(_SC_NPROCESSORS_ONLN);
   if(res > 0)
      return static_cast<size_t>(res);
#endif

   return OS::get_cpu_total();
   }

uint64_t OS::get_high_resolution_clock()
   {
   if(uint64_t cpu_clock = OS::get_cpu_cycle_counter())
      return cpu_clock;

#if defined(BOTAN_TARGET_OS_IS_EMSCRIPTEN)
   return emscripten_get_now();
#endif

   /*
   If we got here either we either don't have an asm instruction
   above, or (for x86) RDTSC is not available at runtime. Try some
   clock_gettimes and return the first one that works, or otherwise
   fall back to std::chrono.
   */

#if defined(BOTAN_TARGET_OS_HAS_CLOCK_GETTIME)

   // The ordering here is somewhat arbitrary...
   const clockid_t clock_types[] = {
#if defined(CLOCK_MONOTONIC_HR)
      CLOCK_MONOTONIC_HR,
#endif
#if defined(CLOCK_MONOTONIC_RAW)
      CLOCK_MONOTONIC_RAW,
#endif
#if defined(CLOCK_MONOTONIC)
      CLOCK_MONOTONIC,
#endif
#if defined(CLOCK_PROCESS_CPUTIME_ID)
      CLOCK_PROCESS_CPUTIME_ID,
#endif
#if defined(CLOCK_THREAD_CPUTIME_ID)
      CLOCK_THREAD_CPUTIME_ID,
#endif
   };

   for(clockid_t clock : clock_types)
      {
      struct timespec ts;
      if(::clock_gettime(clock, &ts) == 0)
         {
         return (static_cast<uint64_t>(ts.tv_sec) * 1000000000) + static_cast<uint64_t>(ts.tv_nsec);
         }
      }
#endif

   // Plain C++11 fallback
   auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
   return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
   }

uint64_t OS::get_system_timestamp_ns()
   {
#if defined(BOTAN_TARGET_OS_HAS_CLOCK_GETTIME)
   struct timespec ts;
   if(::clock_gettime(CLOCK_REALTIME, &ts) == 0)
      {
      return (static_cast<uint64_t>(ts.tv_sec) * 1000000000) + static_cast<uint64_t>(ts.tv_nsec);
      }
#endif

   auto now = std::chrono::system_clock::now().time_since_epoch();
   return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
   }

size_t OS::system_page_size()
   {
   const size_t default_page_size = 4096;

#if defined(BOTAN_TARGET_OS_HAS_POSIX1)
   long p = ::sysconf(_SC_PAGESIZE);
   if(p > 1)
      return static_cast<size_t>(p);
   else
      return default_page_size;
#elif defined(BOTAN_TARGET_OS_HAS_VIRTUAL_LOCK)
   BOTAN_UNUSED(default_page_size);
   SYSTEM_INFO sys_info;
   ::GetSystemInfo(&sys_info);
   return sys_info.dwPageSize;
#else
   return default_page_size;
#endif
   }

size_t OS::get_memory_locking_limit()
   {
#if defined(BOTAN_TARGET_OS_HAS_POSIX1) && defined(BOTAN_TARGET_OS_HAS_POSIX_MLOCK) && defined(RLIMIT_MEMLOCK)
   /*
   * If RLIMIT_MEMLOCK is not defined, likely the OS does not support
   * unprivileged mlock calls.
   *
   * Linux defaults to only 64 KiB of mlockable memory per process
   * (too small) but BSDs offer a small fraction of total RAM (more
   * than we need). Bound the total mlock size to 512 KiB which is
   * enough to run the entire test suite without spilling to non-mlock
   * memory (and thus presumably also enough for many useful
   * programs), but small enough that we should not cause problems
   * even if many processes are mlocking on the same machine.
   */
   const size_t user_req = read_env_variable_sz("BOTAN_MLOCK_POOL_SIZE", BOTAN_MLOCK_ALLOCATOR_MAX_LOCKED_KB);

   const size_t mlock_requested = std::min<size_t>(user_req, BOTAN_MLOCK_ALLOCATOR_MAX_LOCKED_KB);

   if(mlock_requested > 0)
      {
      struct ::rlimit limits;

      ::getrlimit(RLIMIT_MEMLOCK, &limits);

      if(limits.rlim_cur < limits.rlim_max)
         {
         limits.rlim_cur = limits.rlim_max;
         ::setrlimit(RLIMIT_MEMLOCK, &limits);
         ::getrlimit(RLIMIT_MEMLOCK, &limits);
         }

      return std::min<size_t>(limits.rlim_cur, mlock_requested * 1024);
      }

#elif defined(BOTAN_TARGET_OS_HAS_VIRTUAL_LOCK)
   SIZE_T working_min = 0, working_max = 0;
   if(!::GetProcessWorkingSetSize(::GetCurrentProcess(), &working_min, &working_max))
      {
      return 0;
      }

   // According to Microsoft MSDN:
   // The maximum number of pages that a process can lock is equal to the number of pages in its minimum working set minus a small overhead
   // In the book "Windows Internals Part 2": the maximum lockable pages are minimum working set size - 8 pages
   // But the information in the book seems to be inaccurate/outdated
   // I've tested this on Windows 8.1 x64, Windows 10 x64 and Windows 7 x86
   // On all three OS the value is 11 instead of 8
   const size_t overhead = OS::system_page_size() * 11;
   if(working_min > overhead)
      {
      const size_t lockable_bytes = working_min - overhead;
      return std::min<size_t>(lockable_bytes, BOTAN_MLOCK_ALLOCATOR_MAX_LOCKED_KB * 1024);
      }
#endif

   // Not supported on this platform
   return 0;
   }

bool OS::read_env_variable(std::string& value_out, const std::string& name)
   {
   value_out = "";

   if(running_in_privileged_state())
      return false;

#if defined(BOTAN_TARGET_OS_HAS_WIN32) && defined(BOTAN_BUILD_COMPILER_IS_MSVC)
   char val[128] = { 0 };
   size_t req_size = 0;
   if(getenv_s(&req_size, val, sizeof(val), name.c_str()) == 0)
      {
      value_out = std::string(val, req_size);
      return true;
      }
#else
   if(const char* val = std::getenv(name.c_str()))
      {
      value_out = val;
      return true;
      }
#endif

   return false;
   }

size_t OS::read_env_variable_sz(const std::string& name, size_t def)
   {
   std::string value;
   if(read_env_variable(value, name))
      {
      try
         {
         const size_t val = std::stoul(value, nullptr);
         return val;
         }
      catch(std::exception&) { /* ignore it */ }
      }

   return def;
   }

#if defined(BOTAN_TARGET_OS_HAS_POSIX1) && defined(BOTAN_TARGET_OS_HAS_POSIX_MLOCK)

namespace {

int get_locked_fd()
   {
#if defined(BOTAN_TARGET_OS_IS_IOS) || defined(BOTAN_TARGET_OS_IS_MACOS)
   // On Darwin, tagging anonymous pages allows vmmap to track these.
   // Allowed from 240 to 255 for userland applications
   static constexpr int default_locked_fd = 255;
   int locked_fd = default_locked_fd;

   if(size_t locked_fdl = OS::read_env_variable_sz("BOTAN_LOCKED_FD", default_locked_fd))
      {
      if(locked_fdl < 240 || locked_fdl > 255)
         {
         locked_fdl = default_locked_fd;
         }
      locked_fd = static_cast<int>(locked_fdl);
      }
   return VM_MAKE_TAG(locked_fd);
#else
   return -1;
#endif
   }

}

#endif

std::vector<void*> OS::allocate_locked_pages(size_t count)
   {
   std::vector<void*> result;

#if (defined(BOTAN_TARGET_OS_HAS_POSIX1) && defined(BOTAN_TARGET_OS_HAS_POSIX_MLOCK)) || defined(BOTAN_TARGET_OS_HAS_VIRTUAL_LOCK)

   result.reserve(count);

   const size_t page_size = OS::system_page_size();

#if defined(BOTAN_TARGET_OS_HAS_POSIX1) && defined(BOTAN_TARGET_OS_HAS_POSIX_MLOCK)
   static const int locked_fd = get_locked_fd();
#endif

   for(size_t i = 0; i != count; ++i)
      {
      void* ptr = nullptr;

#if defined(BOTAN_TARGET_OS_HAS_POSIX1) && defined(BOTAN_TARGET_OS_HAS_POSIX_MLOCK)

#if !defined(MAP_ANONYMOUS)
   #define MAP_ANONYMOUS MAP_ANON
#endif

#if !defined(MAP_NOCORE)
#if defined(MAP_CONCEAL)
   #define MAP_NOCORE MAP_CONCEAL
#else
   #define MAP_NOCORE 0
#endif
#endif

#if !defined(PROT_MAX)
   #define PROT_MAX(p) 0
#endif
      const int pflags = PROT_READ | PROT_WRITE;

      ptr = ::mmap(nullptr, 3*page_size,
                   pflags | PROT_MAX(pflags),
                   MAP_ANONYMOUS | MAP_PRIVATE | MAP_NOCORE,
                   /*fd=*/locked_fd, /*offset=*/0);

      if(ptr == MAP_FAILED)
         {
         continue;
         }

      // lock the data page
      if(::mlock(static_cast<uint8_t*>(ptr) + page_size, page_size) != 0)
         {
         ::munmap(ptr, 3*page_size);
         continue;
         }

#if defined(MADV_DONTDUMP)
      // we ignore errors here, as DONTDUMP is just a bonus
      ::madvise(static_cast<uint8_t*>(ptr) + page_size, page_size, MADV_DONTDUMP);
#endif

#elif defined(BOTAN_TARGET_OS_HAS_VIRTUAL_LOCK)
      ptr = ::VirtualAlloc(nullptr, 3*page_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

      if(ptr == nullptr)
         continue;

      if(::VirtualLock(static_cast<uint8_t*>(ptr) + page_size, page_size) == 0)
         {
         ::VirtualFree(ptr, 0, MEM_RELEASE);
         continue;
         }
#endif

      std::memset(ptr, 0, 3*page_size); // zero data page and both guard pages

      // Make guard page preceeding the data page
      page_prohibit_access(static_cast<uint8_t*>(ptr));
      // Make guard page following the data page
      page_prohibit_access(static_cast<uint8_t*>(ptr) + 2*page_size);

      result.push_back(static_cast<uint8_t*>(ptr) + page_size);
      }
#else
   BOTAN_UNUSED(count);
#endif

   return result;
   }

void OS::page_allow_access(void* page)
   {
#if defined(BOTAN_TARGET_OS_HAS_POSIX1)
   const size_t page_size = OS::system_page_size();
   ::mprotect(page, page_size, PROT_READ | PROT_WRITE);
#elif defined(BOTAN_TARGET_OS_HAS_VIRTUAL_LOCK)
   const size_t page_size = OS::system_page_size();
   DWORD old_perms = 0;
   ::VirtualProtect(page, page_size, PAGE_READWRITE, &old_perms);
   BOTAN_UNUSED(old_perms);
#else
   BOTAN_UNUSED(page);
#endif
   }

void OS::page_prohibit_access(void* page)
   {
#if defined(BOTAN_TARGET_OS_HAS_POSIX1)
   const size_t page_size = OS::system_page_size();
   ::mprotect(page, page_size, PROT_NONE);
#elif defined(BOTAN_TARGET_OS_HAS_VIRTUAL_LOCK)
   const size_t page_size = OS::system_page_size();
   DWORD old_perms = 0;
   ::VirtualProtect(page, page_size, PAGE_NOACCESS, &old_perms);
   BOTAN_UNUSED(old_perms);
#else
   BOTAN_UNUSED(page);
#endif
   }

void OS::free_locked_pages(const std::vector<void*>& pages)
   {
   const size_t page_size = OS::system_page_size();

   for(size_t i = 0; i != pages.size(); ++i)
      {
      void* ptr = pages[i];

      secure_scrub_memory(ptr, page_size);

      // ptr points to the data page, guard pages are before and after
      page_allow_access(static_cast<uint8_t*>(ptr) - page_size);
      page_allow_access(static_cast<uint8_t*>(ptr) + page_size);

#if defined(BOTAN_TARGET_OS_HAS_POSIX1) && defined(BOTAN_TARGET_OS_HAS_POSIX_MLOCK)
      ::munlock(ptr, page_size);
      ::munmap(static_cast<uint8_t*>(ptr) - page_size, 3*page_size);
#elif defined(BOTAN_TARGET_OS_HAS_VIRTUAL_LOCK)
      ::VirtualUnlock(ptr, page_size);
      ::VirtualFree(static_cast<uint8_t*>(ptr) - page_size, 0, MEM_RELEASE);
#endif
      }
   }

#if defined(BOTAN_TARGET_OS_HAS_POSIX1) && !defined(BOTAN_TARGET_OS_IS_EMSCRIPTEN)

namespace {

static ::sigjmp_buf g_sigill_jmp_buf;

void botan_sigill_handler(int)
   {
   siglongjmp(g_sigill_jmp_buf, /*non-zero return value*/1);
   }

}

#endif

int OS::run_cpu_instruction_probe(std::function<int ()> probe_fn)
   {
   volatile int probe_result = -3;

#if defined(BOTAN_TARGET_OS_HAS_POSIX1) && !defined(BOTAN_TARGET_OS_IS_EMSCRIPTEN)
   struct sigaction old_sigaction;
   struct sigaction sigaction;

   sigaction.sa_handler = botan_sigill_handler;
   sigemptyset(&sigaction.sa_mask);
   sigaction.sa_flags = 0;

   int rc = ::sigaction(SIGILL, &sigaction, &old_sigaction);

   if(rc != 0)
      throw System_Error("run_cpu_instruction_probe sigaction failed", errno);

   rc = sigsetjmp(g_sigill_jmp_buf, /*save sigs*/1);

   if(rc == 0)
      {
      // first call to sigsetjmp
      probe_result = probe_fn();
      }
   else if(rc == 1)
      {
      // non-local return from siglongjmp in signal handler: return error
      probe_result = -1;
      }

   // Restore old SIGILL handler, if any
   rc = ::sigaction(SIGILL, &old_sigaction, nullptr);
   if(rc != 0)
      throw System_Error("run_cpu_instruction_probe sigaction restore failed", errno);

#else
   BOTAN_UNUSED(probe_fn);
#endif

   return probe_result;
   }

std::unique_ptr<OS::Echo_Suppression> OS::suppress_echo_on_terminal()
   {
#if defined(BOTAN_TARGET_OS_HAS_POSIX1)
   class POSIX_Echo_Suppression : public Echo_Suppression
      {
      public:
         POSIX_Echo_Suppression()
            {
            m_stdin_fd = fileno(stdin);
            if(::tcgetattr(m_stdin_fd, &m_old_termios) != 0)
               throw System_Error("Getting terminal status failed", errno);

            struct termios noecho_flags = m_old_termios;
            noecho_flags.c_lflag &= ~ECHO;
            noecho_flags.c_lflag |= ECHONL;

            if(::tcsetattr(m_stdin_fd, TCSANOW, &noecho_flags) != 0)
               throw System_Error("Clearing terminal echo bit failed", errno);
            }

         void reenable_echo() override
            {
            if(m_stdin_fd > 0)
               {
               if(::tcsetattr(m_stdin_fd, TCSANOW, &m_old_termios) != 0)
                  throw System_Error("Restoring terminal echo bit failed", errno);
               m_stdin_fd = -1;
               }
            }

         ~POSIX_Echo_Suppression()
            {
            try
               {
               reenable_echo();
               }
            catch(...)
               {
               }
            }

      private:
         int m_stdin_fd;
         struct termios m_old_termios;
      };

   return std::unique_ptr<Echo_Suppression>(new POSIX_Echo_Suppression);

#elif defined(BOTAN_TARGET_OS_HAS_WIN32)

   class Win32_Echo_Suppression : public Echo_Suppression
      {
      public:
         Win32_Echo_Suppression()
            {
            m_input_handle = ::GetStdHandle(STD_INPUT_HANDLE);
            if(::GetConsoleMode(m_input_handle, &m_console_state) == 0)
               throw System_Error("Getting console mode failed", ::GetLastError());

            DWORD new_mode = ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT;
            if(::SetConsoleMode(m_input_handle, new_mode) == 0)
               throw System_Error("Setting console mode failed", ::GetLastError());
            }

         void reenable_echo() override
            {
            if(m_input_handle != INVALID_HANDLE_VALUE)
               {
               if(::SetConsoleMode(m_input_handle, m_console_state) == 0)
                  throw System_Error("Setting console mode failed", ::GetLastError());
               m_input_handle = INVALID_HANDLE_VALUE;
               }
            }

         ~Win32_Echo_Suppression()
            {
            try
               {
               reenable_echo();
               }
            catch(...)
               {
               }
            }

      private:
         HANDLE m_input_handle;
         DWORD m_console_state;
      };

   return std::unique_ptr<Echo_Suppression>(new Win32_Echo_Suppression);

#else

   // Not supported on this platform, return null
   return std::unique_ptr<Echo_Suppression>();
#endif
   }

}
/*
* Various string utils and parsing functions
* (C) 1999-2007,2013,2014,2015,2018 Jack Lloyd
* (C) 2015 Simon Warta (Kullo GmbH)
* (C) 2017 René Korthaus, Rohde & Schwarz Cybersecurity
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <limits>
#include <set>

#if defined(BOTAN_HAS_ASN1)
#endif

namespace Botan {

uint16_t to_uint16(const std::string& str)
   {
   const uint32_t x = to_u32bit(str);

   if(x >> 16)
      throw Invalid_Argument("Integer value exceeds 16 bit range");

   return static_cast<uint16_t>(x);
   }

uint32_t to_u32bit(const std::string& str)
   {
   // std::stoul is not strict enough. Ensure that str is digit only [0-9]*
   for(const char chr : str)
      {
      if(chr < '0' || chr > '9')
         {
         std::string chrAsString(1, chr);
         throw Invalid_Argument("String contains non-digit char: " + chrAsString);
         }
      }

   const unsigned long int x = std::stoul(str);

   if(sizeof(unsigned long int) > 4)
      {
      // x might be uint64
      if (x > std::numeric_limits<uint32_t>::max())
         {
         throw Invalid_Argument("Integer value of " + str + " exceeds 32 bit range");
         }
      }

   return static_cast<uint32_t>(x);
   }

/*
* Convert a string into a time duration
*/
uint32_t timespec_to_u32bit(const std::string& timespec)
   {
   if(timespec.empty())
      return 0;

   const char suffix = timespec[timespec.size()-1];
   std::string value = timespec.substr(0, timespec.size()-1);

   uint32_t scale = 1;

   if(Charset::is_digit(suffix))
      value += suffix;
   else if(suffix == 's')
      scale = 1;
   else if(suffix == 'm')
      scale = 60;
   else if(suffix == 'h')
      scale = 60 * 60;
   else if(suffix == 'd')
      scale = 24 * 60 * 60;
   else if(suffix == 'y')
      scale = 365 * 24 * 60 * 60;
   else
      throw Decoding_Error("timespec_to_u32bit: Bad input " + timespec);

   return scale * to_u32bit(value);
   }

/*
* Parse a SCAN-style algorithm name
*/
std::vector<std::string> parse_algorithm_name(const std::string& namex)
   {
   if(namex.find('(') == std::string::npos &&
      namex.find(')') == std::string::npos)
      return std::vector<std::string>(1, namex);

   std::string name = namex, substring;
   std::vector<std::string> elems;
   size_t level = 0;

   elems.push_back(name.substr(0, name.find('(')));
   name = name.substr(name.find('('));

   for(auto i = name.begin(); i != name.end(); ++i)
      {
      char c = *i;

      if(c == '(')
         ++level;
      if(c == ')')
         {
         if(level == 1 && i == name.end() - 1)
            {
            if(elems.size() == 1)
               elems.push_back(substring.substr(1));
            else
               elems.push_back(substring);
            return elems;
            }

         if(level == 0 || (level == 1 && i != name.end() - 1))
            throw Invalid_Algorithm_Name(namex);
         --level;
         }

      if(c == ',' && level == 1)
         {
         if(elems.size() == 1)
            elems.push_back(substring.substr(1));
         else
            elems.push_back(substring);
         substring.clear();
         }
      else
         substring += c;
      }

   if(!substring.empty())
      throw Invalid_Algorithm_Name(namex);

   return elems;
   }

std::vector<std::string> split_on(const std::string& str, char delim)
   {
   return split_on_pred(str, [delim](char c) { return c == delim; });
   }

std::vector<std::string> split_on_pred(const std::string& str,
                                       std::function<bool (char)> pred)
   {
   std::vector<std::string> elems;
   if(str.empty()) return elems;

   std::string substr;
   for(auto i = str.begin(); i != str.end(); ++i)
      {
      if(pred(*i))
         {
         if(!substr.empty())
            elems.push_back(substr);
         substr.clear();
         }
      else
         substr += *i;
      }

   if(substr.empty())
      throw Invalid_Argument("Unable to split string: " + str);
   elems.push_back(substr);

   return elems;
   }

/*
* Join a string
*/
std::string string_join(const std::vector<std::string>& strs, char delim)
   {
   std::string out = "";

   for(size_t i = 0; i != strs.size(); ++i)
      {
      if(i != 0)
         out += delim;
      out += strs[i];
      }

   return out;
   }

/*
* Parse an ASN.1 OID string
*/
std::vector<uint32_t> parse_asn1_oid(const std::string& oid)
   {
#if defined(BOTAN_HAS_ASN1)
   return OID(oid).get_components();
#else
   BOTAN_UNUSED(oid);
   throw Not_Implemented("ASN1 support not available");
#endif
   }

/*
* X.500 String Comparison
*/
bool x500_name_cmp(const std::string& name1, const std::string& name2)
   {
   auto p1 = name1.begin();
   auto p2 = name2.begin();

   while((p1 != name1.end()) && Charset::is_space(*p1)) ++p1;
   while((p2 != name2.end()) && Charset::is_space(*p2)) ++p2;

   while(p1 != name1.end() && p2 != name2.end())
      {
      if(Charset::is_space(*p1))
         {
         if(!Charset::is_space(*p2))
            return false;

         while((p1 != name1.end()) && Charset::is_space(*p1)) ++p1;
         while((p2 != name2.end()) && Charset::is_space(*p2)) ++p2;

         if(p1 == name1.end() && p2 == name2.end())
            return true;
         if(p1 == name1.end() || p2 == name2.end())
            return false;
         }

      if(!Charset::caseless_cmp(*p1, *p2))
         return false;
      ++p1;
      ++p2;
      }

   while((p1 != name1.end()) && Charset::is_space(*p1)) ++p1;
   while((p2 != name2.end()) && Charset::is_space(*p2)) ++p2;

   if((p1 != name1.end()) || (p2 != name2.end()))
      return false;
   return true;
   }

/*
* Convert a decimal-dotted string to binary IP
*/
uint32_t string_to_ipv4(const std::string& str)
   {
   std::vector<std::string> parts = split_on(str, '.');

   if(parts.size() != 4)
      throw Decoding_Error("Invalid IP string " + str);

   uint32_t ip = 0;

   for(auto part = parts.begin(); part != parts.end(); ++part)
      {
      uint32_t octet = to_u32bit(*part);

      if(octet > 255)
         throw Decoding_Error("Invalid IP string " + str);

      ip = (ip << 8) | (octet & 0xFF);
      }

   return ip;
   }

/*
* Convert an IP address to decimal-dotted string
*/
std::string ipv4_to_string(uint32_t ip)
   {
   std::string str;

   for(size_t i = 0; i != sizeof(ip); ++i)
      {
      if(i)
         str += ".";
      str += std::to_string(get_byte(i, ip));
      }

   return str;
   }

std::string erase_chars(const std::string& str, const std::set<char>& chars)
   {
   std::string out;

   for(auto c: str)
      if(chars.count(c) == 0)
         out += c;

   return out;
   }

std::string replace_chars(const std::string& str,
                          const std::set<char>& chars,
                          char to_char)
   {
   std::string out = str;

   for(size_t i = 0; i != out.size(); ++i)
      if(chars.count(out[i]))
         out[i] = to_char;

   return out;
   }

std::string replace_char(const std::string& str, char from_char, char to_char)
   {
   std::string out = str;

   for(size_t i = 0; i != out.size(); ++i)
      if(out[i] == from_char)
         out[i] = to_char;

   return out;
   }

std::string tolower_string(const std::string& in)
   {
   std::string s = in;
   for(size_t i = 0; i != s.size(); ++i)
      {
      const int cu = static_cast<unsigned char>(s[i]);
      if(std::isalpha(cu))
         s[i] = static_cast<char>(std::tolower(cu));
      }
   return s;
   }

bool host_wildcard_match(const std::string& issued_, const std::string& host_)
   {
   const std::string issued = tolower_string(issued_);
   const std::string host = tolower_string(host_);

   if(host.empty() || issued.empty())
      return false;

   /*
   If there are embedded nulls in your issued name
   Well I feel bad for you son
   */
   if(std::count(issued.begin(), issued.end(), char(0)) > 0)
      return false;

   // If more than one wildcard, then issued name is invalid
   const size_t stars = std::count(issued.begin(), issued.end(), '*');
   if(stars > 1)
      return false;

   // '*' is not a valid character in DNS names so should not appear on the host side
   if(std::count(host.begin(), host.end(), '*') != 0)
      return false;

   // Similarly a DNS name can't end in .
   if(host[host.size() - 1] == '.')
      return false;

   // And a host can't have an empty name component, so reject that
   if(host.find("..") != std::string::npos)
      return false;

   // Exact match: accept
   if(issued == host)
      {
      return true;
      }

   /*
   Otherwise it might be a wildcard

   If the issued size is strictly longer than the hostname size it
   couldn't possibly be a match, even if the issued value is a
   wildcard. The only exception is when the wildcard ends up empty
   (eg www.example.com matches www*.example.com)
   */
   if(issued.size() > host.size() + 1)
      {
      return false;
      }

   // If no * at all then not a wildcard, and so not a match
   if(stars != 1)
      {
      return false;
      }

   /*
   Now walk through the issued string, making sure every character
   matches. When we come to the (singular) '*', jump forward in the
   hostname by the corresponding amount. We know exactly how much
   space the wildcard takes because it must be exactly `len(host) -
   len(issued) + 1 chars`.

   We also verify that the '*' comes in the leftmost component, and
   doesn't skip over any '.' in the hostname.
   */
   size_t dots_seen = 0;
   size_t host_idx = 0;

   for(size_t i = 0; i != issued.size(); ++i)
      {
      dots_seen += (issued[i] == '.');

      if(issued[i] == '*')
         {
         // Fail: wildcard can only come in leftmost component
         if(dots_seen > 0)
            {
            return false;
            }

         /*
         Since there is only one * we know the tail of the issued and
         hostname must be an exact match. In this case advance host_idx
         to match.
         */
         const size_t advance = (host.size() - issued.size() + 1);

         if(host_idx + advance > host.size()) // shouldn't happen
            return false;

         // Can't be any intervening .s that we would have skipped
         if(std::count(host.begin() + host_idx,
                       host.begin() + host_idx + advance, '.') != 0)
            return false;

         host_idx += advance;
         }
      else
         {
         if(issued[i] != host[host_idx])
            {
            return false;
            }

         host_idx += 1;
         }
      }

   // Wildcard issued name must have at least 3 components
   if(dots_seen < 2)
      {
      return false;
      }

   return true;
   }

}
/*
* Simple config/test file reader
* (C) 2013,2014,2015 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

std::string clean_ws(const std::string& s)
   {
   const char* ws = " \t\n";
   auto start = s.find_first_not_of(ws);
   auto end = s.find_last_not_of(ws);

   if(start == std::string::npos)
      return "";

   if(end == std::string::npos)
      return s.substr(start, end);
   else
      return s.substr(start, start + end + 1);
   }

std::map<std::string, std::string> read_cfg(std::istream& is)
   {
   std::map<std::string, std::string> kv;
   size_t line = 0;

   while(is.good())
      {
      std::string s;

      std::getline(is, s);

      ++line;

      if(s.empty() || s[0] == '#')
         continue;

      s = clean_ws(s.substr(0, s.find('#')));

      if(s.empty())
         continue;

      auto eq = s.find("=");

      if(eq == std::string::npos || eq == 0 || eq == s.size() - 1)
         throw Decoding_Error("Bad read_cfg input '" + s + "' on line " + std::to_string(line));

      const std::string key = clean_ws(s.substr(0, eq));
      const std::string val = clean_ws(s.substr(eq + 1, std::string::npos));

      kv[key] = val;
      }

   return kv;
   }

}
/*
* (C) 2018 Ribose Inc
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

std::map<std::string, std::string> read_kv(const std::string& kv)
   {
   std::map<std::string, std::string> m;
   if(kv == "")
      return m;

   std::vector<std::string> parts;

   try
      {
      parts = split_on(kv, ',');
      }
   catch(std::exception&)
      {
      throw Invalid_Argument("Bad KV spec");
      }

   bool escaped = false;
   bool reading_key = true;
   std::string cur_key;
   std::string cur_val;

   for(char c : kv)
      {
      if(c == '\\' && !escaped)
         {
         escaped = true;
         }
      else if(c == ',' && !escaped)
         {
         if(cur_key.empty())
            throw Invalid_Argument("Bad KV spec empty key");

         if(m.find(cur_key) != m.end())
            throw Invalid_Argument("Bad KV spec duplicated key");
         m[cur_key] = cur_val;
         cur_key = "";
         cur_val = "";
         reading_key = true;
         }
      else if(c == '=' && !escaped)
         {
         if(reading_key == false)
            throw Invalid_Argument("Bad KV spec unexpected equals sign");
         reading_key = false;
         }
      else
         {
         if(reading_key)
            cur_key += c;
         else
            cur_val += c;

         if(escaped)
            escaped = false;
         }
      }

   if(!cur_key.empty())
      {
      if(reading_key == false)
         {
         if(m.find(cur_key) != m.end())
            throw Invalid_Argument("Bad KV spec duplicated key");
         m[cur_key] = cur_val;
         }
      else
         throw Invalid_Argument("Bad KV spec incomplete string");
      }

   return m;
   }

}
/*
* (C) 2018 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

void Timer::start()
   {
   stop();
   m_timer_start = OS::get_system_timestamp_ns();
   m_cpu_cycles_start = OS::get_cpu_cycle_counter();
   }

void Timer::stop()
   {
   if(m_timer_start)
      {
      if(m_cpu_cycles_start != 0)
         {
         const uint64_t cycles_taken = OS::get_cpu_cycle_counter() - m_cpu_cycles_start;
         if(cycles_taken > 0)
            {
            m_cpu_cycles_used += static_cast<size_t>(cycles_taken * m_clock_cycle_ratio);
            }
         }

      const uint64_t now = OS::get_system_timestamp_ns();

      if(now > m_timer_start)
         {
         const uint64_t dur = now - m_timer_start;

         m_time_used += dur;

         if(m_event_count == 0)
            {
            m_min_time = m_max_time = dur;
            }
         else
            {
            m_max_time = std::max(m_max_time, dur);
            m_min_time = std::min(m_min_time, dur);
            }
         }

      m_timer_start = 0;
      ++m_event_count;
      }
   }

bool Timer::operator<(const Timer& other) const
   {
   if(this->doing() != other.doing())
      return (this->doing() < other.doing());

   return (this->get_name() < other.get_name());
   }

std::string Timer::to_string() const
   {
   if(m_custom_msg.size() > 0)
      {
      return m_custom_msg;
      }
   else if(this->buf_size() == 0)
      {
      return result_string_ops();
      }
   else
      {
      return result_string_bps();
      }
   }

std::string Timer::result_string_bps() const
   {
   const size_t MiB = 1024 * 1024;

   const double MiB_total = static_cast<double>(events()) / MiB;
   const double MiB_per_sec = MiB_total / seconds();

   std::ostringstream oss;
   oss << get_name();

   if(!doing().empty())
      {
      oss << " " << doing();
      }

   if(buf_size() > 0)
      {
      oss << " buffer size " << buf_size() << " bytes:";
      }

   if(events() == 0)
      oss << " " << "N/A";
   else
      oss << " " << std::fixed << std::setprecision(3) << MiB_per_sec << " MiB/sec";

   if(cycles_consumed() != 0)
      {
      const double cycles_per_byte = static_cast<double>(cycles_consumed()) / events();
      oss << " " << std::fixed << std::setprecision(2) << cycles_per_byte << " cycles/byte";
      }

   oss << " (" << MiB_total << " MiB in " << milliseconds() << " ms)\n";

   return oss.str();
   }

std::string Timer::result_string_ops() const
   {
   std::ostringstream oss;

   oss << get_name() << " ";

   if(events() == 0)
      {
      oss << "no events\n";
      }
   else
      {
      oss << static_cast<uint64_t>(events_per_second())
          << ' ' << doing() << "/sec; "
          << std::setprecision(2) << std::fixed
          << ms_per_event() << " ms/op";

      if(cycles_consumed() != 0)
         {
         const double cycles_per_op = static_cast<double>(cycles_consumed()) / events();
         const int precision = (cycles_per_op < 10000) ? 2 : 0;
         oss << " " << std::fixed << std::setprecision(precision) << cycles_per_op << " cycles/op";
         }

      oss << " (" << events() << " " << (events() == 1 ? "op" : "ops")
          << " in " << milliseconds() << " ms)\n";
      }

   return oss.str();
   }

}
/*
* Version Information
* (C) 1999-2013,2015 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/


namespace Botan {

/*
  These are intentionally compiled rather than inlined, so an
  application running against a shared library can test the true
  version they are running against.
*/

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

const char* short_version_cstr()
   {
   return STR(BOTAN_VERSION_MAJOR) "."
          STR(BOTAN_VERSION_MINOR) "."
          STR(BOTAN_VERSION_PATCH)
#if defined(BOTAN_VERSION_SUFFIX)
          STR(BOTAN_VERSION_SUFFIX)
#endif
      ;
   }

const char* version_cstr()
   {

   /*
   It is intentional that this string is a compile-time constant;
   it makes it much easier to find in binaries.
   */

   return "Botan " STR(BOTAN_VERSION_MAJOR) "."
                   STR(BOTAN_VERSION_MINOR) "."
                   STR(BOTAN_VERSION_PATCH)
#if defined(BOTAN_VERSION_SUFFIX)
                   STR(BOTAN_VERSION_SUFFIX)
#endif
                   " ("
#if defined(BOTAN_UNSAFE_FUZZER_MODE)
                   "UNSAFE FUZZER MODE BUILD "
#endif
                   BOTAN_VERSION_RELEASE_TYPE
#if (BOTAN_VERSION_DATESTAMP != 0)
                   ", dated " STR(BOTAN_VERSION_DATESTAMP)
#endif
                   ", revision " BOTAN_VERSION_VC_REVISION
                   ", distribution " BOTAN_DISTRIBUTION_INFO ")";
   }

#undef STR
#undef QUOTE

/*
* Return the version as a string
*/
std::string version_string()
   {
   return std::string(version_cstr());
   }

std::string short_version_string()
   {
   return std::string(short_version_cstr());
   }

uint32_t version_datestamp() { return BOTAN_VERSION_DATESTAMP; }

/*
* Return parts of the version as integers
*/
uint32_t version_major() { return BOTAN_VERSION_MAJOR; }
uint32_t version_minor() { return BOTAN_VERSION_MINOR; }
uint32_t version_patch() { return BOTAN_VERSION_PATCH; }

std::string runtime_version_check(uint32_t major,
                                  uint32_t minor,
                                  uint32_t patch)
   {
   if(major != version_major() || minor != version_minor() || patch != version_patch())
      {
      std::ostringstream oss;
      oss << "Warning: linked version (" << short_version_string() << ")"
          << " does not match version built against "
          << "(" << major << '.' << minor << '.' << patch << ")\n";
      return oss.str();
      }

   return "";
   }

}
