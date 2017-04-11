#ifndef _BASE64_H_INCLUDED_
#define _BASE64_H_INCLUDED_

#include <cstddef>
#include <string>
#include <vector>
#include <cctype>

namespace base64 {

    inline std::string encode_base64(const void* data, std::size_t in_len) {
        const std::string chars ="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        const unsigned char* bytes_to_encode = static_cast<const unsigned char*>(data);
        unsigned int i = 0;
        unsigned int j = 0;
        unsigned char char_array_3[3];
        unsigned char char_array_4[4];

        std::string out;
        out.reserve(in_len / 5 * 8 + 4);
        while (in_len--) {
            char_array_3[i++] = *(bytes_to_encode++);
            if (i == 3) {
                char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
                char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
                char_array_4[3] = char_array_3[2] & 0x3f;

                for (i = 0; (i < 4); i++)
                    out += chars[char_array_4[i]];
                i = 0;
            }
        }

        if (i) {
            for (j = i; j < 3; j++)
                char_array_3[j] = '\0';

            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (j = 0; (j < i + 1); j++)
                out += chars[char_array_4[j]];

            while ((i++ < 3))
                out += '=';
        }

        return out;
    }

    template <typename T>
    inline std::vector<T> decode_base64(const char* encoded, std::size_t in_len) {
        const std::string chars ="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        unsigned int i = 0;
        unsigned int j = 0;
        unsigned int in_ = 0;
        unsigned char char_array_4[4], char_array_3[3];

        std::vector<T> out;
        out.reserve(in_len / 8 * 5 + 2);
        while (in_len-- && (encoded[in_] != '=') && (isalnum(encoded[in_]) || encoded[in_] == '+' || encoded[in_] == '/')) {
            char_array_4[i++] = encoded[in_]; in_++;
            if (i == 4) {
                for (i = 0; i < 4; i++)
                    char_array_4[i] = static_cast<unsigned char>(chars.find(char_array_4[i]));

                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

                for (i = 0; (i < 3); i++)
                    out.push_back(char_array_3[i]);
                i = 0;
            }
        }

        if (i) {
            for (j = i; j < 4; j++)
                char_array_4[j] = 0;

            for (j = 0; j < 4; j++)
                char_array_4[j] = static_cast<unsigned char>(chars.find(char_array_4[j]));

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (j = 0; (j < i - 1); j++) out.push_back(char_array_3[j]);
        }

        return out;
    }

} // namespace base64

#endif // _BASE64_H_INCLUDED_
