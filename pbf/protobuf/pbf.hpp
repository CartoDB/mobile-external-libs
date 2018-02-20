#ifndef _PROTOBUF_PBF_HPP
#define _PROTOBUF_PBF_HPP

/*
 * Some parts are from upb - a minimalist implementation of protocol buffers.
 *
 * Copyright (c) 2009-2011, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Google Inc. nor the names of any other
 *       contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY GOOGLE INC. ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL GOOGLE INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Josh Haberman <jhaberman@gmail.com>
 * Author: Dane Springmeyer
 * Author: Mark Tehver
 */

#include <cstdint>
#include <stdexcept>
#include <string>

#ifndef _PROTOBUF_USE_RVALUE_REFS
#if __cplusplus >= 201103L || _MSC_VER >= 1900
#define _PROTOBUF_USE_RVALUE_REFS 1
#else
#define _PROTOBUF_USE_RVALUE_REFS 0
#endif
#endif

namespace protobuf {

	class parse_error : public std::runtime_error {
	public:
		explicit parse_error(const char * msg) : std::runtime_error(msg) { }
	};

	class message {
	public:
		std::uint64_t value;
		std::uint32_t tag;

		inline message(const void * ptr, std::size_t len);

		inline bool valid() const;
		inline std::size_t length() const;
		inline bool next();
		inline void skip();
		inline const void * data() const;

		inline message read_message();
		inline std::string read_string();
		inline std::string read_bytes();
		inline const char * read_raw_string(std::size_t & size);
		inline std::int32_t read_int32();
		inline std::int64_t read_int64();
		inline std::int32_t read_sint32();
		inline std::int64_t read_sint64();
		inline std::uint32_t read_uint32();
		inline std::uint64_t read_uint64();
		inline std::int32_t read_sfixed32();
		inline std::int64_t read_sfixed64();
		inline std::uint32_t read_fixed32();
		inline std::uint64_t read_fixed64();
		inline float read_float();
		inline double read_double();
		inline bool read_bool();

	private:
		typedef const char * value_type;

		value_type data_;
		value_type end_;

		inline void skip_value(std::uint64_t val);
		inline void skip_bytes(std::uint64_t bytes);

		inline std::uint32_t read_varint32();
		inline std::uint64_t read_varint64();
	};

	inline message::message(const void * ptr, std::size_t len) : data_(static_cast<value_type>(ptr)), end_(static_cast<value_type>(ptr) + len)
	{
	}

	inline bool message::valid() const
	{
		return data_ < end_;
	}

	inline std::size_t message::length() const
	{
		return end_ - data_;
	}

	inline bool message::next()
	{
		if (data_ < end_) {
			value = read_varint64();
			tag = static_cast<std::uint32_t>(value >> 3);
			return true;
		}
		return false;
	}

	inline const void * message::data() const
	{
		return data_;
	}

	inline void message::skip()
	{
		skip_value(value);
	}

	inline message message::read_message()
	{
		std::size_t len = static_cast<std::size_t>(read_varint64());
		skip_bytes(len);
		return message(data_ - len, len);
	}

	inline std::string message::read_string()
	{
		std::size_t len = static_cast<std::size_t>(read_varint64());
		skip_bytes(len);
		return std::string(data_ - len, len);
	}

	inline std::string message::read_bytes()
	{
		return read_string();
	}

	inline const char * message::read_raw_string(std::size_t & size)
	{
		size = static_cast<std::size_t>(read_varint64());
		skip_bytes(size);
                return data_ - size;
	}

	inline std::int32_t message::read_int32()
	{
		return static_cast<std::int32_t>(read_varint32());
	}

	inline std::int64_t message::read_int64()
	{
		return static_cast<std::int64_t>(read_varint64());
	}

	inline std::int32_t message::read_sint32()
	{
		std::uint32_t n = read_varint32();
		return static_cast<std::int32_t>(n >> 1) ^ -static_cast<std::int32_t>((n & 1));
	}

	inline std::int64_t message::read_sint64()
	{
		std::uint64_t n = read_varint64();
		return static_cast<std::int64_t>(n >> 1) ^ -static_cast<std::int64_t>((n & 1));
	}

	inline std::uint32_t message::read_uint32()
	{
		return read_varint32();
	}

	inline std::uint64_t message::read_uint64()
	{
		return read_varint64();
	}

	inline std::int32_t message::read_sfixed32()
	{
		return static_cast<std::int32_t>(read_fixed32());
	}

	inline std::int64_t message::read_sfixed64()
	{
		return static_cast<std::int64_t>(read_fixed64());
	}

	inline std::uint32_t message::read_fixed32()
	{
		skip_bytes(4);
		std::uint32_t result = static_cast<std::uint8_t>(*(data_ - 1));
		result = (result << 8) | static_cast<std::uint8_t>(*(data_ - 2));
		result = (result << 8) | static_cast<std::uint8_t>(*(data_ - 3));
		result = (result << 8) | static_cast<std::uint8_t>(*(data_ - 4));
		return result;
	}

	inline std::uint64_t message::read_fixed64()
	{
		skip_bytes(8);
		std::uint64_t result = 0;
		for (int i = 1; i <= 8; i++) {
			result = (result << 8) | static_cast<std::uint8_t>(*(data_ - i));
		}
		return result;
	}

	inline float message::read_float()
	{
		std::uint32_t n = read_fixed32();
		return *reinterpret_cast<float *>(&n);
	}

	inline double message::read_double()
	{
		std::uint64_t n = read_fixed64();
		return *reinterpret_cast<double *>(&n);
	}

	inline bool message::read_bool()
	{
		skip_bytes(1);
		return *(data_ - 1) != 0;
	}

	inline void message::skip_value(std::uint64_t val)
	{
		switch (val & 0x7) {
		case 0: // varint
			read_varint64();
			break;
		case 1: // 64 bit
			skip_bytes(8);
			break;
		case 2: // string/message
			skip_bytes(read_varint64());
			break;
		case 5: // 32 bit
			skip_bytes(4);
			break;
		default:
			throw parse_error("unknown pbf type");
		}
	}

	inline void message::skip_bytes(std::uint64_t bytes)
	{
		data_ += static_cast<std::size_t>(bytes);
		if (data_ > end_) {
			throw parse_error("unexpected end of buffer");
		}
	}

	inline std::uint32_t message::read_varint32()
	{
		return static_cast<std::uint32_t>(read_varint64());
	}

	inline std::uint64_t message::read_varint64()
	{
		std::uint64_t result = 0;
		int bitpos = 0;

		if (data_ + 4 <= end_) {
			// fast path, for up to 28-bit numbers
			std::uint8_t byte;
			result = static_cast<std::uint32_t>(byte = *data_++) & 0x7F;
			if (!(byte & 0x80)) return result;
			result |= (static_cast<std::uint32_t>(byte = *data_++) & 0x7F) << 7;
			if (!(byte & 0x80)) return result;
			result |= (static_cast<std::uint32_t>(byte = *data_++) & 0x7F) << 14;
			if (!(byte & 0x80)) return result;
			result |= (static_cast<std::uint32_t>(byte = *data_++) & 0x7F) << 21;
			if (!(byte & 0x80)) return result;
			bitpos = 28;
		}

		while (bitpos < 70) {
			if (data_ >= end_) {
				throw parse_error("unterminated varint, unexpected end of buffer");
			}

			std::uint8_t byte;
			result |= (static_cast<std::uint64_t>(byte = *data_++) & 0x7F) << bitpos;
			if (!(byte & 0x80)) return result;
			bitpos += 7;
		}
		throw parse_error("unterminated varint (too long)");
	}
}

#endif // _PROTOBUF_PBF_HPP
