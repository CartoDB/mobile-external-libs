#ifndef _TIMED_LRU_CACHE_H_INCLUDED_
#define _TIMED_LRU_CACHE_H_INCLUDED_

#include <unordered_map>
#include <unordered_set>
#include <list>
#include <functional>
#include <cstddef>
#include <stdexcept>
#include <chrono>

namespace cache {

    template <
        typename key_t,
        typename value_t,
        typename hash_t = std::hash<key_t>,
        typename key_equal_t = std::equal_to<key_t>>
    class timed_lru_cache {
        public:
            explicit timed_lru_cache(std::size_t max_size) : _size(0), _max_size(max_size) { }

            void put(const key_t& key, const value_t& value, std::size_t size) {
                remove(key);

                _size += size;
                _cache_items_list.push_front(item(key, value, size));
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
                    return it->second->value;
                }
            }

            bool peek(const key_t& key, value_t& value) const {
                auto it = _cache_items_map.find(key);
                if (it == _cache_items_map.end()) {
                    return false;
                }
                else {
                    value = it->second->value;
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
                    value = it->second->value;
                    return true;
                }
            }

            bool exists(const key_t& key) const {
                return _cache_items_map.find(key) != _cache_items_map.end();
            }

            void invalidate_all(std::chrono::steady_clock::time_point expiration_time) {
                for (auto it = _cache_items_map.begin(); it != _cache_items_map.end(); it++) {
                    _cache_expiration_map[it->first] = expiration_time;
                }
            }

            bool invalidate(const key_t& key, std::chrono::steady_clock::time_point expiration_time) {
                auto it = _cache_items_map.find(key);
                if (it == _cache_items_map.end()) {
                    return false;
                }
                else {
                    _cache_expiration_map[key] = expiration_time;
                    return true;
                }
            }

            bool valid(const key_t& key) const {
                auto it = _cache_expiration_map.find(key);
                if (it == _cache_expiration_map.end()) {
                    return true;
                }
                else {
                    return it->second > std::chrono::steady_clock::now();
                }
            }

            void clear() {
                _cache_items_list.clear();
                _cache_items_map.clear();
                _cache_expiration_map.clear();
                _size = 0;
            }

            bool remove(const key_t& key) {
                auto it = _cache_items_map.find(key);
                if (it == _cache_items_map.end()) {
                    return false;
                }

                _size -= it->second->size;
                _cache_items_list.erase(it->second);
                _cache_items_map.erase(it);
                _cache_expiration_map.erase(key);
                return true;
            }

            bool move(const key_t& key, timed_lru_cache& cache) {
                auto it1 = _cache_items_map.find(key);
                if (it1 == _cache_items_map.end()) {
                    return false;
                }
                cache.put(key, it1->second->value, it1->second->size);

                auto it2 = _cache_expiration_map.find(key);
                if (it2 != _cache_expiration_map.end()) {
                    cache.invalidate(key, it2->second);
                }

                remove(key);
                
                return true;
            }

            void resize(std::size_t size) {
                _max_size = size;
                evict();
            }

            std::size_t capacity() const {
                return _max_size;
            }

            std::size_t size() const {
                return _size;
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
            struct item {
                explicit item(const key_t& key, const value_t& value, std::size_t size) : key(key), value(value), size(size) { }
                
                key_t key;
                value_t value;
                std::size_t size;
            };
            
            typedef typename std::list<item>::iterator list_iterator_t;

            void evict() {
                while (_size > _max_size) {
                    auto it = _cache_items_list.end();
                    it--;
                    _cache_items_map.erase(it->key);
                    _cache_expiration_map.erase(it->key);
                    _size -= it->size;
                    _cache_items_list.pop_back();
                }
            }

            std::list<item> _cache_items_list;
            std::unordered_map<key_t, list_iterator_t, hash_t, key_equal_t> _cache_items_map;
            std::unordered_map<key_t, std::chrono::steady_clock::time_point> _cache_expiration_map;
            std::size_t _size;
            std::size_t _max_size;
    };

} // namespace cache

#endif	// _TIMED_LRU_CACHE_H_INCLUDED_
