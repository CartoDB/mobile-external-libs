#ifndef _UNGZIP_H_INCLUDED_
#define _UNGZIP_H_INCLUDED_

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>

#include <zlib.h>

#include "utf8_filesystem.h"

namespace zlib {

    inline bool ungzip_file(FILE* in_file, FILE* out_file) {
        std::int64_t pos = utf8_filesystem::ftell64(in_file);
        unsigned char header[14];
        if (fread(header, 1, sizeof(header), in_file) < sizeof(header)) {
            utf8_filesystem::fseek64(in_file, pos, SEEK_SET);
            return false;
        }
        utf8_filesystem::fseek64(in_file, pos, SEEK_SET);

        if (header[0] != 0x1f || header[1] != 0x8b) {
            return false;
        }
        if (header[2] != 8) {
            return false;
        }

        std::vector<unsigned char> in_buf(4096);
        std::vector<unsigned char> out_buf(16384);
        ::z_stream infstream;
        std::memset(&infstream, 0, sizeof(infstream));
        infstream.zalloc = NULL;
        infstream.zfree = NULL;
        infstream.opaque = NULL;
        int err = Z_OK;
        infstream.avail_in = static_cast<unsigned int>(in_buf.size()); // size of input
        infstream.next_in = in_buf.data();
        infstream.avail_out = static_cast<unsigned int>(out_buf.size()); // size of output
        infstream.next_out = out_buf.data(); // output char array
        ::inflateInit2(&infstream, MAX_WBITS + 16);
        do {
            infstream.avail_in = static_cast<unsigned int>(fread(in_buf.data(), 1, in_buf.size(), in_file));
            if (ferror(in_file)) {
                err = Z_ERRNO;
                break;
            }
            if (infstream.avail_in == 0) {
                break;
            }
            infstream.next_in = in_buf.data();

            do {
                infstream.avail_out = static_cast<unsigned int>(out_buf.size()); // size of output
                infstream.next_out = out_buf.data(); // output char array
                err = ::inflate(&infstream, Z_NO_FLUSH);
                if (err == Z_BUF_ERROR) {
                    err = Z_OK;
                }
                if (err != Z_OK && err != Z_STREAM_END) {
                    break;
                }

                fwrite(out_buf.data(), 1, out_buf.size() - infstream.avail_out, out_file);
                if (ferror(out_file)) {
                    err = Z_ERRNO;
                    break;
                }
            } while (infstream.avail_out == 0);
        } while (err == Z_OK);
        ::inflateEnd(&infstream);

        utf8_filesystem::ftruncate64(out_file, utf8_filesystem::ftell64(out_file));

        return err == Z_OK || err == Z_STREAM_END;
    }

} // namespace zlib

#endif // _ZLIB_H_INCLUDED_
