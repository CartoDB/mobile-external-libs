#ifndef _EIFF_FILE_H_INCLUDED_
#define _EIFF_FILE_H_INCLUDED_

#include <cstddef>
#include <cassert>
#include <memory>
#include <array>
#include <vector>
#include <istream>
#include <ostream>
#include <algorithm>

namespace eiff {
    
    // A chunk in EIFF file
    class chunk {
    public:
        using tag_type = std::array<char, 4>;
        
        virtual ~chunk() = default;
        
        const tag_type& tag() const { return _tag; }
        
    protected:
        explicit chunk(const tag_type& tag) : _tag(tag) { }
        
    private:
        tag_type _tag = tag_type();
    };
    
    // FORM chunk, contains subchunks
    class form_chunk : public chunk {
    public:
        using iterator = std::vector<std::shared_ptr<chunk>>::const_iterator;
        
        form_chunk() : chunk(tag_type {{ 'F', 'O', 'R', 'M' }}), _chunks() { }
        explicit form_chunk(std::vector<std::shared_ptr<chunk>>&& chunks) : chunk(tag_type {{ 'F', 'O', 'R', 'M' }}), _chunks(chunks) { }
        explicit form_chunk(const std::vector<std::shared_ptr<chunk>>& chunks) : chunk(tag_type {{ 'F', 'O', 'R', 'M' }}), _chunks(chunks) { }
        
        iterator begin() const { return _chunks.begin(); }
        iterator end() const { return _chunks.end(); }
        iterator find(const tag_type& tag) const { return std::find_if(begin(), end(), [&tag](const std::shared_ptr<chunk>& chunk) { return chunk->tag() == tag; }); }
        template <typename Chunk> std::shared_ptr<Chunk> get(const tag_type& tag) const { iterator it = find(tag); return it == end() ? std::shared_ptr<Chunk>() : std::dynamic_pointer_cast<Chunk>(*it); }
        
        void clear() { _chunks.clear(); }
        void insert(const std::shared_ptr<chunk>& chunk) { _chunks.push_back(chunk); }
        
    private:
        std::vector<std::shared_ptr<chunk>> _chunks;
    };
    
    // Abstract data chunk
    class data_chunk : public chunk {
    public:
        using size_type = std::uint64_t;
        
        virtual size_type size() const = 0;
        virtual void read(std::vector<unsigned char>& data) const = 0;
        virtual void read(std::vector<unsigned char>& data, size_type offset, std::size_t size) const = 0;
        
    protected:
        explicit data_chunk(const tag_type& tag) : chunk(tag) { }
    };
    
    // Data chunk with data in memory
    class memory_data_chunk : public data_chunk {
    public:
        memory_data_chunk(const tag_type& tag, std::vector<unsigned char>&& data) : data_chunk(tag), _data(data) { }
        memory_data_chunk(const tag_type& tag, const std::vector<unsigned char>& data) : data_chunk(tag), _data(data) { }
        
        virtual size_type size() const override { return _data.size(); }
        virtual void read(std::vector<unsigned char>& data) const override { data = _data; }
        virtual void read(std::vector<unsigned char>& data, size_type offset, std::size_t size) const override { data.assign(_data.begin() + offset, _data.begin() + offset + size); }
        
    private:
        std::vector<unsigned char> _data;
    };
    
    // Data chunk with data in file at specific offset
    class file_data_chunk : public data_chunk {
    public:
        file_data_chunk(const tag_type& tag, const std::shared_ptr<std::istream>& stream) : data_chunk(tag), _stream(stream) { _offset = 0; stream->seekg(0, std::ios::end); _size = stream->tellg(); }
        file_data_chunk(const tag_type& tag, const std::shared_ptr<std::istream>& stream, size_type offset, size_type size) : data_chunk(tag), _stream(stream), _offset(offset), _size(size) { }
        
        virtual size_type size() const override { return _size; }
        virtual void read(std::vector<unsigned char>& data) const override { _stream->seekg(_offset); data.resize(static_cast<std::size_t>(_size)); _stream->read(reinterpret_cast<char*>(data.data()), data.size()); }
        virtual void read(std::vector<unsigned char>& data, size_type offset, std::size_t size) const override { _stream->seekg(_offset + offset); data.resize(static_cast<std::size_t>(size)); _stream->read(reinterpret_cast<char*>(data.data()), data.size()); }
        
    private:
        std::shared_ptr<std::basic_istream<char>> _stream;
        size_type _offset = 0;
        size_type _size = 0;
    };
    
    // Read chunk from stream. File-based flag specifies whether chunks are fully loaded or simply referenced
    template <typename Stream>
    inline std::shared_ptr<chunk> read_chunk(const std::shared_ptr<Stream>& stream, bool file_based) {
        chunk::tag_type tag { };
        stream->read(reinterpret_cast<char*>(tag.data()), sizeof(tag));
        std::uint64_t size = 0;
        stream->read(reinterpret_cast<char*>(&size), sizeof(size));
        data_chunk::size_type start_offset = stream->tellg();
        std::shared_ptr<chunk> result;
        if (tag == form_chunk().tag()) {
            std::uint64_t count = 0;
            stream->read(reinterpret_cast<char*>(&count), sizeof(count));
            std::vector<std::shared_ptr<chunk>> chunks;
            chunks.reserve(static_cast<std::size_t>(count));
            while (count-- > 0) {
                chunks.push_back(read_chunk(stream, file_based));
            }
            result = std::make_shared<form_chunk>(std::move(chunks));
        } else {
            if (file_based) {
                data_chunk::size_type offset = stream->tellg();
                result = std::make_shared<file_data_chunk>(tag, stream, offset, size);
            } else {
                std::vector<unsigned char> data(static_cast<std::size_t>(size));
                assert(data.size() == size);
                stream->read(reinterpret_cast<char*>(data.data()), data.size());
                result = std::make_shared<memory_data_chunk>(tag, std::move(data));
            }
        }
        stream->seekg(start_offset + size);
        return result;
    }
    
    // Write chunk to stream
    template <typename Stream>
    inline void write_chunk(Stream& stream, const std::shared_ptr<chunk>& input) {
        chunk::tag_type tag = input->tag();
        stream.write(reinterpret_cast<char*>(tag.data()), sizeof(tag));
        std::uint64_t size = 0;
        stream.write(reinterpret_cast<char*>(&size), sizeof(size));
        data_chunk::size_type start_offset = stream.tellp();
        if (auto form = std::dynamic_pointer_cast<form_chunk>(input)) {
            std::uint64_t count = std::distance(form->begin(), form->end());
            stream.write(reinterpret_cast<char*>(&count), sizeof(count));
            for (const std::shared_ptr<chunk>& chunk : *form) {
                write_chunk(stream, chunk);
            }
        } else if (auto data = std::dynamic_pointer_cast<data_chunk>(input)) {
            std::vector<unsigned char> buffer(65536);
            for (data_chunk::size_type offset = 0; offset < data->size(); ) {
                data_chunk::size_type block = std::min(data->size() - offset, static_cast<data_chunk::size_type>(buffer.size()));
                data->read(buffer, offset, static_cast<std::size_t>(block));
                stream.write(reinterpret_cast<char*>(buffer.data()), block);
                offset += block;
            }
        } else {
            assert(false);
        }
        data_chunk::size_type end_offset = stream.tellp();
        stream.seekp(start_offset - sizeof(size));
        size = end_offset - start_offset;
        stream.write(reinterpret_cast<char*>(&size), sizeof(size));
        stream.seekp(end_offset);
    }
    
} // namespace eiff

#endif


