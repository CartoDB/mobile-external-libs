#ifndef _PROTOBUF_ENCODEDPBF_HPP
#define _PROTOBUF_ENCODEDPBF_HPP

/*
 * Copyright (c) 2017, CARTO Inc.
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
 * Author: Mark Tehver
 */

#include <cstdint>
#include <cstddef>
#include <string>

namespace protobuf {

	class encoded_message {
	public:
		enum field_type { varint_type = 0, fixed64_type = 1, length_type = 2, fixed32_type = 5 };

		inline encoded_message();

		inline void clear();

		inline bool empty() const;
		inline const std::string & data() const;

		inline void write_tag(std::uint32_t tag, field_type field);
		inline void write_message(const encoded_message & msg);
		inline void write_string(const std::string & str);
		inline void write_bytes(const void * data, std::size_t size);
		inline void write_int32(std::int32_t val);
		inline void write_int64(std::int64_t val);
		inline void write_sint32(std::int32_t val);
		inline void write_sint64(std::int64_t val);
		inline void write_uint32(std::uint32_t val);
		inline void write_uint64(std::uint64_t val);
		inline void write_sfixed32(std::int32_t val);
		inline void write_sfixed64(std::int64_t);
		inline void write_fixed32(std::uint32_t);
		inline void write_fixed64(std::uint64_t);
		inline void write_float(float val);
		inline void write_double(double val);
		inline void write_bool(bool val);

	private:
		std::string message_;

		inline void write_varint32(std::uint32_t val);
		inline void write_varint64(std::uint64_t val);
	};

	inline encoded_message::encoded_message() : message_()
	{
	}

	inline void encoded_message::clear()
	{
		message_.clear();
	}

	inline bool encoded_message::empty() const
	{
		return message_.empty();
	}

	inline const std::string & encoded_message::data() const
	{
		return message_;
	}

	inline void encoded_message::write_tag(std::uint32_t tag, field_type field)
	{
		write_varint64((static_cast<std::uint64_t>(tag) << 3) | static_cast<std::uint64_t>(field));
	}

	inline void encoded_message::write_message(const encoded_message & msg)
	{
		write_varint64(static_cast<std::int64_t>(msg.message_.size()));
		message_.append(msg.message_);
	}

	inline void encoded_message::write_string(const std::string & str)
	{
		write_varint64(static_cast<std::int64_t>(str.size()));
		message_.append(str);
	}

	inline void encoded_message::write_bytes(const void * data, std::size_t size)
	{
		write_varint64(static_cast<std::int64_t>(size));
		message_.append(static_cast<const char *>(data), size);
	}

	inline void encoded_message::write_int32(std::int32_t val)
	{
		write_varint32(static_cast<std::uint32_t>(val));
	}

	inline void encoded_message::write_int64(std::int64_t val)
	{
		write_varint64(static_cast<std::uint64_t>(val));
	}

	inline void encoded_message::write_sint32(std::int32_t val)
	{
		std::int32_t s = (val < 0 ? 1 : 0);
		std::uint32_t n = static_cast<std::uint32_t>((val << 1) ^ (-s));
		write_varint32(n);
	}

	inline void encoded_message::write_sint64(std::int64_t val)
	{
		std::int64_t s = (val < 0 ? 1 : 0);
		std::uint64_t n = static_cast<std::uint64_t>((val << 1) ^ (-s));
		write_varint64(n);
	}

	inline void encoded_message::write_uint32(std::uint32_t val)
	{
		write_varint32(val);
	}

	inline void encoded_message::write_uint64(std::uint64_t val)
	{
		write_varint64(val);
	}

	inline void encoded_message::write_sfixed32(std::int32_t val)
	{
		write_fixed32(static_cast<std::uint32_t>(val));
	}

	inline void encoded_message::write_sfixed64(std::int64_t val)
	{
		write_fixed64(static_cast<std::uint64_t>(val));
	}

	inline void encoded_message::write_fixed32(std::uint32_t val)
	{
		message_.append(1, static_cast<std::uint8_t>((val >>  0) & 0xFF));
		message_.append(1, static_cast<std::uint8_t>((val >>  8) & 0xFF));
		message_.append(1, static_cast<std::uint8_t>((val >> 16) & 0xFF));
		message_.append(1, static_cast<std::uint8_t>((val >> 24) & 0xFF));
	}

	inline void encoded_message::write_fixed64(std::uint64_t val)
	{
		for (int i = 0; i < 8; i++) {
			message_.append(1, static_cast<std::uint8_t>((val >> (i * 8)) & 0xFF));
		}
	}

	inline void encoded_message::write_float(float val)
	{
		std::uint32_t n = 0;
		*reinterpret_cast<float *>(&n) = val;
		write_fixed32(n);
	}

	inline void encoded_message::write_double(double val)
	{
		std::uint64_t n = 0;
		*reinterpret_cast<double *>(&n) = val;
		write_fixed64(n);
	}

	inline void encoded_message::write_bool(bool val)
	{
		message_.append(1, val ? 1 : 0);
	}

	inline void encoded_message::write_varint32(std::uint32_t val)
	{
		write_varint64(val);
	}

	inline void encoded_message::write_varint64(std::uint64_t val)
	{
		do {
			message_.append(1, (val > 0x7F ? 0x80 : 0) | (val & 0x7F));
			val = val >> 7;
		} while (val != 0);
	}
}

#endif // _PROTOBUF_ENCODEDPBF_HPP
