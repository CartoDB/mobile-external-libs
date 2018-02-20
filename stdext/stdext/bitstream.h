#ifndef _BITSTREAM_H_INCLUDED_
#define _BITSTREAM_H_INCLUDED_

#include <vector>
#include <cstddef>
#include <cassert>

namespace bitstreams {

    // Get the minimum number of bits required to encode specified value
    template <typename T>
    int get_required_bits(T val) {
        int bits = 0;
        for (; bits < 64; bits++) {
            if (val == 0) {
                break;
            }
            val = val >> 1;
        }
        return bits;
    }

    // Output bitstream
    class output_bitstream {
    public:
        output_bitstream() : _bits(0), _data() {
            _data.reserve(65536);
        }
        
        void write_bit(bool bit) {
            if (_bits == 0) {
                _data.push_back(0);
            }
            if (bit) {
                _data.back() |= static_cast<unsigned char>(1 << _bits);
            }
            _bits = (_bits + 1) & 7;
        }

        template <typename T>
        void write_bits(T val, int bits) {
            T initial = val;
            while (bits > 0) {
                if (_bits == 0) {
                    _data.push_back(0);
                }
                int block = std::min(8 - _bits, bits);
                unsigned int mask = (1 << block) - 1;
                _data.back() |= static_cast<unsigned char>((val & mask) << _bits);

                _bits = (_bits + block) & 7;
                val = val >> block;
                bits -= block;
            }
            assert(!(initial > 0 && val > 0));
        }

        template <typename T>
        void rewrite_bits(T val, int bits, std::uint64_t offset) {
            T initial = val;
            while (bits > 0) {
                int shift = static_cast<int>(offset & 7);
                int block = std::min(8 - shift, bits);
                unsigned int mask = (1 << block) - 1;
                assert(static_cast<std::size_t>(offset / 8) < _data.size());
                _data.at(static_cast<std::size_t>(offset / 8)) |= static_cast<unsigned char>((val & mask) << shift);

                offset += block;
                val = val >> block;
                bits -= block;
            }
            assert(!(initial > 0 && val > 0));
        }

        std::uint64_t tell() const {
            return static_cast<std::uint64_t>(_data.size()) * 8 - (_bits == 0 ? 0 : 8 - _bits);
        }

        const std::vector<unsigned char>& data() const {
            return _data;
        }

    private:
        int _bits;
        std::vector<unsigned char> _data;
    };

    // Input bitstream
    class input_bitstream {
    public:
        explicit input_bitstream(const std::vector<unsigned char>& data) : _bits(0), _offset(0), _data(data) { }
        explicit input_bitstream(std::vector<unsigned char>&& data) : _bits(0), _offset(0), _data(std::move(data)) { }

        bool read_bit() {
            bool bit = ((_data.at(_offset) >> _bits) & 1) != 0;
            
            _bits += 1;
            _offset += _bits >> 3;
            _bits = _bits & 7;
            return bit;
        }

        template <typename T>
        T read_bits(int bits) {
            T val = 0;
            int shift = 0;
            while (bits > 0) {
                int block = std::min(8 - _bits, bits);
                unsigned int mask = (1 << block) - 1;
                val |= static_cast<T>((_data.at(_offset) >> _bits) & mask) << shift;
                
                _bits += block;
                _offset += _bits >> 3;
                _bits = _bits & 7;
                shift += block;
                bits -= block;
            }
            return val;
        }

    public:
        int _bits;
        std::size_t _offset;
        std::vector<unsigned char> _data;
    };

} // namespace bitstreams

#endif // _BITSTREAM_H_INCLUDED_
