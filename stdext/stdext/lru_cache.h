#ifndef _LRU_CACHE_H_INCLUDED_
#define _LRU_CACHE_H_INCLUDED_

#include <unordered_map>
#include <unordered_set>
#include <list>
#include <functional>
#include <cstddef>
#include <stdexcept>

namespace cache {

    template <
        typename key_t,
        typename value_t,
        typename hash_t = std::hash<key_t>,
        typename key_equal_t = std::equal_to<key_t>>
    class lru_cache {
    public:
        typedef typename std::pair<key_t, value_t> key_value_pair_t;
        typedef typename std::list<key_value_pair_t>::iterator list_iterator_t;

        explicit lru_cache(std::size_t max_size) : _max_size(max_size) { }

        void put(const key_t& key, const value_t& value) {
            auto it = _cache_items_map.find(key);
            if (it != _cache_items_map.end()) {
                _cache_items_list.erase(it->second);
                _cache_items_map.erase(it);
            }

            _cache_items_list.push_front(key_value_pair_t(key, value));
            _cache_items_map[key] = _cache_items_list.begin();

            evict();
        }
        
        const value_t& get(const key_t& key) {
            auto it = _cache_items_map.find(key);
            if (it == _cache_items_map.end()) {
                throw std::range_error("There is no such key in cache");
            }
            else {
                if (it->second != _cache_items_list.begin()) {
                    _cache_items_list.splice(_cache_items_list.begin(), _cache_items_list, it->second);
                }
                return it->second->second;
            }
        }

        bool peek(const key_t& key, value_t& value) const {
            auto it = _cache_items_map.find(key);
            if (it == _cache_items_map.end()) {
                return false;
            }
            else {
                value = it->second->second;
                return true;
            }
        }
        
        bool read(const key_t& key, value_t& value) {
            auto it = _cache_items_map.find(key);
            if (it == _cache_items_map.end()) {
                return false;
            }
            else {
                if (it->second != _cache_items_list.begin()) {
                    _cache_items_list.splice(_cache_items_list.begin(), _cache_items_list, it->second);
                }
                value = it->second->second;
                return true;
            }
        }
        
        bool exists(const key_t& key) const {
            return _cache_items_map.find(key) != _cache_items_map.end();
        }

        void clear() {
            _cache_items_list.clear();
            _cache_items_map.clear();
        }

        void resize(std::size_t size) {
            _max_size = size;
            evict();
        }

        std::size_t capacity() const {
            return _max_size;
        }

        std::size_t size() const {
            return _cache_items_map.size();
        }

        bool empty() const {
            return size() == 0;
        }
        
        std::unordered_set<key_t> keys() const {
            std::unordered_set<key_t> keys;
            for (auto it = _cache_items_map.begin(); it != _cache_items_map.end(); it++) {
                keys.insert(it->first);
            }
            return keys;
        }

    private:
        void evict() {
            while (_cache_items_map.size() > _max_size) {
                auto last = _cache_items_list.end();
                last--;
                _cache_items_map.erase(last->first);
                _cache_items_list.pop_back();
            }
        }

        std::list<key_value_pair_t> _cache_items_list;
        std::unordered_map<key_t, list_iterator_t, hash_t, key_equal_t> _cache_items_map;
        std::size_t _max_size;
    };

} // namespace cache

#endif	// _LRU_CACHE_H_INCLUDED_
