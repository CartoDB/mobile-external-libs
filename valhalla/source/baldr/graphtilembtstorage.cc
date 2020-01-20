#include "baldr/graphtilembtstorage.h"
#include <valhalla/midgard/pointll.h>
#include <valhalla/midgard/aabb2.h>
#include <valhalla/midgard/tiles.h>

#include <cmath>
#include <locale>
#include <iomanip>
#include <boost/algorithm/string.hpp>
#include <sqlite3pp.h>
#include "config.h"

#include <zlib.h>

namespace {
  bool inflate(const void* in_data, size_t in_size, std::vector<char>& out) {
    const unsigned char* in = reinterpret_cast<const unsigned char*>(in_data);

    if (in_size < 14) {
      return false;
    }

    size_t offset = 0;
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

    std::vector<unsigned char> buf(16384);
    ::z_stream infstream;
    std::memset(&infstream, 0, sizeof(infstream));
    infstream.zalloc = NULL;
    infstream.zfree = NULL;
    infstream.opaque = NULL;
    int err = Z_OK;
    infstream.avail_in = static_cast<unsigned int>(in_size - offset - 4); // size of input
    infstream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(in_data + offset)); // input char array
    infstream.avail_out = static_cast<unsigned int>(buf.size()); // size of output
    infstream.next_out = buf.data(); // output char array
    ::inflateInit2(&infstream, -Z_DEFAULT_WINDOW_BITS);
    do {
      infstream.avail_out = static_cast<unsigned int>(buf.size()); // size of output
      infstream.next_out = buf.data(); // output char array
      err = ::inflate(&infstream, infstream.avail_in > 0 ? Z_NO_FLUSH : Z_FINISH);
      if (err != Z_OK && err != Z_STREAM_END) {
        break;
      }
      out.insert(out.end(), reinterpret_cast<char*>(buf.data()), reinterpret_cast<char*>(buf.data() + buf.size() - infstream.avail_out));
    } while (err != Z_STREAM_END);
    ::inflateEnd(&infstream);
    return err == Z_OK || err == Z_STREAM_END;
  }

  bool inflate_raw(const void* in_data, std::size_t in_size, const void* dict, std::size_t dict_size, std::vector<char>& out) {
    const unsigned char* in = reinterpret_cast<const unsigned char*>(in_data);

    out.reserve(in_size);

    std::vector<unsigned char> buf(16384);
    ::z_stream infstream;
    std::memset(&infstream, 0, sizeof(infstream));
    infstream.zalloc = NULL;
    infstream.zfree = NULL;
    infstream.opaque = NULL;
    int err = Z_OK;
    infstream.avail_in = static_cast<unsigned int>(in_size); // size of input
    infstream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(in_data)); // input char array
    infstream.avail_out = static_cast<unsigned int>(buf.size()); // size of output
    infstream.next_out = buf.data(); // output char array
    ::inflateInit2(&infstream, -MAX_WBITS);
    if (dict) {
        ::inflateSetDictionary(&infstream, reinterpret_cast<const Bytef*>(dict), static_cast<unsigned int>(dict_size));
    }
    do {
      infstream.avail_out = static_cast<unsigned int>(buf.size()); // size of output
      infstream.next_out = buf.data(); // output char array
      err = ::inflate(&infstream, infstream.avail_in > 0 ? Z_NO_FLUSH : Z_FINISH);
      if (err != Z_OK && err != Z_STREAM_END) {
        break;
      }
      out.insert(out.end(), reinterpret_cast<char*>(buf.data()), reinterpret_cast<char*>(buf.data() + buf.size() - infstream.avail_out));
    } while (err != Z_STREAM_END);
    ::inflateEnd(&infstream);
    return err == Z_OK || err == Z_STREAM_END;
  }
}

namespace valhalla {
namespace baldr {

GraphTileMBTStorage::GraphTileMBTStorage(const std::vector<std::shared_ptr<sqlite3pp::database>>& dbs)
    : mbt_dbs_() {
    for (auto& db : dbs) {
      std::shared_ptr<std::vector<unsigned char>> dict;
      sqlite3pp::query query(*db, "SELECT value FROM metadata WHERE name='shared_zlib_dict'");
      for (auto qit = query.begin(); qit != query.end(); qit++) {
        const unsigned char* dataPtr = reinterpret_cast<const unsigned char*>(qit->get<const void*>(0));
        std::size_t dataSize = qit->column_bytes(0);
        dict.reset(new std::vector<unsigned char>(dataPtr, dataPtr + dataSize));
      }
      mbt_dbs_.emplace_back({ db, dict });
    }
}

std::unordered_set<GraphId> GraphTileMBTStorage::FindTiles(const TileHierarchy& tile_hierarchy) const {
  std::unordered_set<GraphId> graphids;
  for (auto& mbt_db : mbt_dbs_) {
    try {
      sqlite3pp::query query(*mbt_db.database, "SELECT zoom_level, tile_column, tile_row FROM tiles");
      for (auto it = query.begin(); it != query.end(); it++) {
        int z = (*it).get<int>(0);
        int x = (*it).get<int>(1);
        int y = (*it).get<int>(2);
        graphids.insert(ToGraphId(std::make_tuple(z, x, y), tile_hierarchy));
      }
    } catch (const std::exception&) {
    }
  }
  return graphids;
}

bool GraphTileMBTStorage::DoesTileExist(const GraphId& graphid, const TileHierarchy& tile_hierarchy) const {
  for (auto& mbt_db : mbt_dbs_) {
    try {
      std::tuple<int, int, int> tile_coords = FromGraphId(graphid, tile_hierarchy);
      sqlite3pp::query query(*mbt_db.database, "SELECT COUNT(*) FROM tiles WHERE zoom_level=:z AND tile_row=:y and tile_column=:y");
      query.bind(":z", std::get<0>(tile_coords));
      query.bind(":x", std::get<1>(tile_coords));
      query.bind(":y", std::get<2>(tile_coords));
      for (auto it = query.begin(); it != query.end(); it++) {
        if ((*it).get<int>(0) > 0) {
          return true;
        }
      }
    } catch (const std::exception&) {
    }
  }
  return false;
}

bool GraphTileMBTStorage::ReadTile(const GraphId& graphid, const TileHierarchy& tile_hierarchy, std::vector<char>& tile_data) const {
  for (auto& mbt_db : mbt_dbs_) {
    try {
      std::tuple<int, int, int> tile_coords = FromGraphId(graphid, tile_hierarchy);
      sqlite3pp::query query(*mbt_db.database, "SELECT tile_data FROM tiles WHERE zoom_level=:z AND tile_row=:y and tile_column=:x");
      query.bind(":z", std::get<0>(tile_coords));
      query.bind(":x", std::get<1>(tile_coords));
      query.bind(":y", std::get<2>(tile_coords));
      for (auto it = query.begin(); it != query.end(); it++) {
        std::size_t data_size = (*it).column_bytes(0);
        const unsigned char* data_ptr = static_cast<const unsigned char*>((*it).get<const void*>(0));
        tile_data.clear();
        tile_data.reserve(data_size + 64);
        if (mbt_db.zdict) {
          return inflate_raw(data_ptr, data_size, mbt_db.zdict->data(), mbt_db.zdict->size(), tile_data);
        } else {
          return inflate(data_ptr, data_size, tile_data);
        }
      }
    } catch (const std::exception&) {
    }
  }
  return false;
}

bool GraphTileMBTStorage::ReadTileRealTimeSpeeds(const GraphId& graphid, const TileHierarchy& tile_hierarchy, std::vector<uint8_t>& rts_data) const {
  return false;
}

std::tuple<int, int, int> GraphTileMBTStorage::FromGraphId(const GraphId& graphid, const TileHierarchy& tile_hierarchy) {
  auto it = tile_hierarchy.levels().find(graphid.level());
  if (it == tile_hierarchy.levels().end()) {
    return std::tuple<int, int, int>();
  }
  auto coords = it->second.tiles.GetRowColumn(graphid.tileid());
  return std::tuple<int, int, int>(graphid.level(), coords.second, coords.first);
}

GraphId GraphTileMBTStorage::ToGraphId(const std::tuple<int, int, int>& tile_coords, const TileHierarchy& tile_hierarchy) {
  auto it = tile_hierarchy.levels().find(std::get<0>(tile_coords));
  if (it == tile_hierarchy.levels().end()) {
    return GraphId();
  }
  uint32_t tileid = it->second.tiles.TileId(std::get<1>(tile_coords), std::get<2>(tile_coords));
  return GraphId(tileid, std::get<0>(tile_coords), 0);
}

}
}
