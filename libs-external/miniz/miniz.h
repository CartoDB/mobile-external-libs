#pragma once

#define MINIZ_HEADER_FILE_ONLY
#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES
#include <miniz.c>
#include <vector>
#include <cstdlib>

namespace miniz {

    template <typename T>
    inline bool inflate(const void* in_data, std::size_t in_size, std::vector<T>& out) {
        if (in_size < 14) {
            return false;
        }

        const unsigned char* in = reinterpret_cast<const unsigned char*>(in_data);
        std::size_t offset = 0;
        if (in[0] != 0x1f || in[1] != 0x8b) {
            return false;
        }
        if (in[2] != 8) {
            return false;
        }
        int flags = in[3];
        offset += 10;
        if (flags & (1 << 2)) { // FEXTRA
            int n = static_cast<int>(in[offset + 0]) | (static_cast<int>(in[offset + 1]) << 8);
            offset += n + 2;
        }
        if (flags & (1 << 3)) { // FNAME
            while (offset < in_size) {
                if (in[offset++] == 0) {
                    break;
                }
            }
        }
        if (flags & (1 << 4)) { // FCOMMENT
            while (offset < in_size) {
                if (in[offset++] == 0) {
                    break;
                }
            }
        }
        if (flags & (1 << 1)) { // FCRC
            offset += 2;
        }

        unsigned char buf[4096];
        ::mz_stream infstream;
        std::memset(&infstream, 0, sizeof(infstream));
        infstream.zalloc = NULL;
        infstream.zfree = NULL;
        infstream.opaque = NULL;
        int err = MZ_OK;
        infstream.avail_in = static_cast<unsigned int>(in_size - offset - 4); // size of input
        infstream.next_in = &in[offset]; // input char array
        infstream.avail_out = sizeof(buf); // size of output
        infstream.next_out = buf; // output char array
        ::mz_inflateInit2(&infstream, -MZ_DEFAULT_WINDOW_BITS);
        do {
            infstream.avail_out = sizeof(buf); // size of output
            infstream.next_out = buf; // output char array
            err = ::mz_inflate(&infstream, infstream.avail_in > 0 ? MZ_NO_FLUSH : MZ_FINISH);
            if (err != MZ_OK && err != MZ_STREAM_END) {
                break;
            }
            out.insert(out.end(), reinterpret_cast<T*>(&buf[0]), reinterpret_cast<T*>(&buf[0]) + sizeof(buf) - infstream.avail_out);
        } while (err != MZ_STREAM_END);
        ::mz_inflateEnd(&infstream);
        return err == MZ_OK || err == MZ_STREAM_END;
    }

}
