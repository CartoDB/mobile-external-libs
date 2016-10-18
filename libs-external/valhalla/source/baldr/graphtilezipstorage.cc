#include "baldr/graphtilezipstorage.h"
#include <valhalla/midgard/pointll.h>
#include <valhalla/midgard/aabb2.h>
#include <valhalla/midgard/tiles.h>

#include <cmath>
#include <locale>
#include <iomanip>
#include <boost/algorithm/string.hpp>
#include "config.h"

#define MINIZ_HEADER_FILE_ONLY
#include <miniz.c>

namespace {
  struct dir_facet : public std::numpunct<char> {
   protected:
    virtual char do_thousands_sep() const {
        return '/';
    }

    virtual std::string do_grouping() const {
        return "\03";
    }
  };
  template <class numeric_t>
  size_t digits(numeric_t number) {
    size_t digits = (number < 0 ? 1 : 0);
    while (static_cast<long long int>(number)) {
        number /= 10;
        digits++;
    }
    return digits;
  }
  const std::locale dir_locale(std::locale("C"), new dir_facet());
  const valhalla::midgard::AABB2<valhalla::midgard::PointLL> world_box(valhalla::midgard::PointLL(-180, -90), valhalla::midgard::PointLL(180, 90));
}

namespace valhalla {
namespace baldr {

GraphTileZipStorage::GraphTileZipStorage(const std::string& zip_file)
    : zip_file_(zip_file) {
}

std::vector<GraphId> GraphTileZipStorage::FindTiles(const TileHierarchy& tile_hierarchy) const {
  std::vector<GraphId> graphids;
  mz_zip_archive zip;
  memset(&zip, 0, sizeof(mz_zip_archive));
  if (mz_zip_reader_init_file(&zip, zip_file_.c_str(), 0)) {
    for (int i = 0; i < mz_zip_reader_get_num_files(&zip); i++) {
      char filename[1024];
      memset(filename, 0, sizeof(filename));
      mz_zip_reader_get_filename(&zip, i, filename, sizeof(filename) - 1);
      GraphId id = GetTileId(filename, "");
      graphids.push_back(id);
    }
    mz_zip_reader_end(&zip);
  }
  return graphids;
}

bool GraphTileZipStorage::DoesTileExist(const GraphId& graphid, const TileHierarchy& tile_hierarchy) const {
  std::vector<char> tile_data;
  return ReadTile(graphid, tile_hierarchy, tile_data); // TODO: optimize
}

bool GraphTileZipStorage::ReadTile(const GraphId& graphid, const TileHierarchy& tile_hierarchy, std::vector<char>& tile_data) const {
  mz_zip_archive zip;
  memset(&zip, 0, sizeof(mz_zip_archive));
  if (mz_zip_reader_init_file(&zip, zip_file_.c_str(), 0)) {
    std::string file_location = FileSuffix(graphid.Tile_Base(), tile_hierarchy);
    
    size_t filesize = 0;
    std::shared_ptr<char> filedata(static_cast<char*>(mz_zip_reader_extract_file_to_heap(&zip, file_location.c_str(), &filesize, 0)), mz_free);
    mz_zip_reader_end(&zip);

    if (filedata) {
      tile_data = std::vector<char>(filedata.get(), filedata.get() + filesize);
      return true;
    }
  }
  return false;
}

bool GraphTileZipStorage::ReadTileRealTimeSpeeds(const GraphId& graphid, const TileHierarchy& tile_hierarchy, std::vector<uint8_t>& rts_data) const {
  return false;
}

// Get the tile Id given the full path to the file.
GraphId GraphTileZipStorage::GetTileId(const std::string& fname, const std::string& tile_dir) {
  //strip off the unuseful part
  auto pos = tile_dir.empty() ? 0 : fname.find(tile_dir);
  if(pos == std::string::npos)
    throw std::runtime_error("File name for tile does not match hierarchy root dir");
  auto name = fname.substr(pos + tile_dir.size());
  boost::algorithm::trim_if(name, boost::is_any_of("/.gph"));

  //split on slash
  std::vector<std::string> tokens;
  boost::split(tokens, name, boost::is_any_of("/"));

  //need at least level and id
  if(tokens.size() < 2)
    throw std::runtime_error("Invalid tile path");

  // Compute the Id
  uint32_t id = 0;
  uint32_t multiplier = std::pow(1000, tokens.size() - 2);
  bool first = true;
  for(const auto& token : tokens) {
    if(first) {
      first = false;
      continue;
    }
    id += std::atoi(token.c_str()) * multiplier;
    multiplier /= 1000;
  }
  uint32_t level = std::atoi(tokens.front().c_str());
  return {id, level, 0};
}

std::string GraphTileZipStorage::FileSuffix(const GraphId& graphid, const TileHierarchy& tile_hierarchy) {
  /*
  if you have a graphid where level == 8 and tileid == 24134109851
  you should get: 8/024/134/109/851.gph
  since the number of levels is likely to be very small this limits
  the total number of objects in any one directory to 1000, which is an
  empirically derived good choice for mechanical harddrives
  this should be fine for s3 (even though it breaks the rule of most
  unique part of filename first) because there will be just so few
  objects in general in practice
  */

  //figure the largest id for this level
  auto level = tile_hierarchy.levels().find(graphid.level());
  if(level == tile_hierarchy.levels().end() &&
     graphid.level() == ((tile_hierarchy.levels().rbegin())->second.level + 1))
    level = tile_hierarchy.levels().begin();

  if(level == tile_hierarchy.levels().end())
    throw std::runtime_error("Could not compute FileSuffix for non-existent level");

  const uint32_t max_id = valhalla::midgard::Tiles<valhalla::midgard::PointLL>::MaxTileId(world_box, level->second.tiles.TileSize());

  //figure out how many digits
  //TODO: dont convert it to a string to get the length there are faster ways..
  size_t max_length = digits<uint32_t>(max_id);
  const size_t remainder = max_length % 3;
  if(remainder)
    max_length += 3 - remainder;

  //make a locale to use as a formatter for numbers
  std::ostringstream stream;
  stream.imbue(dir_locale);

  //if it starts with a zero the pow trick doesn't work
  if(graphid.level() == 0) {
    stream << static_cast<uint32_t>(std::pow(10, max_length)) + graphid.tileid() << ".gph";
    std::string suffix = stream.str();
    suffix[0] = '0';
    return suffix;
  }
  //it was something else
  stream << graphid.level() * static_cast<uint32_t>(std::pow(10, max_length)) + graphid.tileid() << ".gph";
  return stream.str();
}

}
}
