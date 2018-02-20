#ifndef _ZLIB_H_INCLUDED_
#define _ZLIB_H_INCLUDED_

#include <vector>
#include <cstdlib>
#include <cstring>

#include <zlib.h>

namespace zlib {

    template <typename T>
    inline bool inflate_raw(const void* in_data, std::size_t in_size, const void* dict, std::size_t dict_size, std::vector<T>& out) {
        const unsigned char* in = reinterpret_cast<const unsigned char*>(in_data);

        out.reserve(in_size);

        unsigned char buf[4096];
        ::z_stream infstream;
        std::memset(&infstream, 0, sizeof(infstream));
        infstream.zalloc = NULL;
        infstream.zfree = NULL;
        infstream.opaque = NULL;
        int err = Z_OK;
        infstream.avail_in = static_cast<unsigned int>(in_size); // size of input
        infstream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(in_data)); // input char array
        infstream.avail_out = sizeof(buf); // size of output
        infstream.next_out = buf; // output char array
        ::inflateInit2(&infstream, -MAX_WBITS);
        if (dict) {
            ::inflateSetDictionary(&infstream, reinterpret_cast<const Bytef*>(dict), static_cast<unsigned int>(dict_size));
        }
        do {
            infstream.avail_out = sizeof(buf); // size of output
            infstream.next_out = buf; // output char array
            err = ::inflate(&infstream, infstream.avail_in > 0 ? Z_NO_FLUSH : Z_FINISH);
            if (err != Z_OK && err != Z_STREAM_END) {
                break;
            }
            out.insert(out.end(), reinterpret_cast<T*>(&buf[0]), reinterpret_cast<T*>(&buf[0]) + sizeof(buf) - infstream.avail_out);
        } while (err != Z_STREAM_END);
        ::inflateEnd(&infstream);
        return err == Z_OK || err == Z_STREAM_END;
    }

    template <typename T>
    inline bool inflate_raw(const void* in_data, std::size_t in_size, std::vector<T>& out) {
        return inflate_raw(in_data, in_size, nullptr, 0, out);
    }

    template <typename T>
    inline bool inflate_zlib(const void* in_data, std::size_t in_size, const void* dict, std::size_t dict_size, std::vector<T>& out) {
        const unsigned char* in = reinterpret_cast<const unsigned char*>(in_data);

        out.reserve(in_size);

        unsigned char buf[4096];
        ::z_stream infstream;
        std::memset(&infstream, 0, sizeof(infstream));
        infstream.zalloc = NULL;
        infstream.zfree = NULL;
        infstream.opaque = NULL;
        int err = Z_OK;
        infstream.avail_in = static_cast<unsigned int>(in_size); // size of input
        infstream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(in_data)); // input char array
        infstream.avail_out = sizeof(buf); // size of output
        infstream.next_out = buf; // output char array
        ::inflateInit2(&infstream, MAX_WBITS);
        if (dict) {
            ::inflateSetDictionary(&infstream, reinterpret_cast<const Bytef*>(dict), static_cast<unsigned int>(dict_size));
        }
        do {
            infstream.avail_out = sizeof(buf); // size of output
            infstream.next_out = buf; // output char array
            err = ::inflate(&infstream, infstream.avail_in > 0 ? Z_NO_FLUSH : Z_FINISH);
            if (err != Z_OK && err != Z_STREAM_END) {
                break;
            }
            out.insert(out.end(), reinterpret_cast<T*>(&buf[0]), reinterpret_cast<T*>(&buf[0]) + sizeof(buf) - infstream.avail_out);
        } while (err != Z_STREAM_END);
        ::inflateEnd(&infstream);
        return err == Z_OK || err == Z_STREAM_END;
    }

    template <typename T>
    inline bool inflate_zlib(const void* in_data, std::size_t in_size, std::vector<T>& out) {
        return inflate_zlib(in_data, in_size, nullptr, 0, out);
    }

    template <typename T>
    inline bool inflate_gzip(const void* in_data, std::size_t in_size, const void* dict, std::size_t dict_size, std::vector<T>& out) {
        if (in_size < 14) {
            return false;
        }

        const unsigned char* in = reinterpret_cast<const unsigned char*>(in_data);
        if (in[0] != 0x1f || in[1] != 0x8b) {
            return false;
        }
        if (in[2] != 8) {
            return false;
        }

        std::size_t out_size = in[in_size - 4];
        out_size += static_cast<std::size_t>(in[in_size - 3]) << 8;
        out_size += static_cast<std::size_t>(in[in_size - 2]) << 16;
        out_size += static_cast<std::size_t>(in[in_size - 1]) << 24;
        if (out_size < (1 << 24)) { // ignore size if too over 16MB, could be broken data
            out.reserve(out_size);
        }

        unsigned char buf[4096];
        ::z_stream infstream;
        std::memset(&infstream, 0, sizeof(infstream));
        infstream.zalloc = NULL;
        infstream.zfree = NULL;
        infstream.opaque = NULL;
        int err = Z_OK;
        infstream.avail_in = static_cast<unsigned int>(in_size); // size of input
        infstream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(in_data)); // input char array
        infstream.avail_out = sizeof(buf); // size of output
        infstream.next_out = buf; // output char array
        ::inflateInit2(&infstream, MAX_WBITS + 16);
        if (dict) {
            ::inflateSetDictionary(&infstream, reinterpret_cast<const Bytef*>(dict), static_cast<unsigned int>(dict_size));
        }
        do {
            infstream.avail_out = sizeof(buf); // size of output
            infstream.next_out = buf; // output char array
            err = ::inflate(&infstream, infstream.avail_in > 0 ? Z_NO_FLUSH : Z_FINISH);
            if (err != Z_OK && err != Z_STREAM_END) {
                break;
            }
            out.insert(out.end(), reinterpret_cast<T*>(&buf[0]), reinterpret_cast<T*>(&buf[0]) + sizeof(buf) - infstream.avail_out);
        } while (err != Z_STREAM_END);
        ::inflateEnd(&infstream);
        return err == Z_OK || err == Z_STREAM_END;
    }

    template <typename T>
    inline bool inflate_gzip(const void* in_data, std::size_t in_size, std::vector<T>& out) {
        return inflate_gzip(in_data, in_size, nullptr, 0, out);
    }

} // namespace zlib

#endif // _ZLIB_H_INCLUDED_
