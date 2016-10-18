#include "baldr/graphtilefsstorage.h"
#include <valhalla/midgard/pointll.h>
#include <valhalla/midgard/aabb2.h>
#include <valhalla/midgard/tiles.h>

#include <cmath>
#include <locale>
#include <iomanip>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

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

GraphTileFsStorage::GraphTileFsStorage(const std::string& tile_dir)
    : tile_dir_(tile_dir) {
}

std::vector<GraphId> GraphTileFsStorage::FindTiles(const TileHierarchy& tile_hierarchy) const {
  // Set the transit level
  transit_level = tile_hierarchy.levels().rbegin()->second.level + 1;

  // Populate a map for each level of the tiles that exist
  std::vector<GraphId> graphids;
  for (uint32_t tile_level = 0; tile_level <= transit_level; tile_level++) {
    boost::filesystem::path root_dir(tile_dir_ + '/' + std::to_string<std::string>(tile_level) + '/');
    if(boost::filesystem::exists(root_dir) && boost::filesystem::is_directory(root_dir)) {
      for (boost::filesystem::recursive_directory_iterator i(root_dir), end; i != end; ++i) {
        if (!boost::filesystem::is_directory(i->path())) {
          GraphId id = GetTileId(i->path().string(), tile_dir_);
          graphids.push_back(id);
        }
      }
    }
  }
  return graphids;
}

bool GraphTileFsStorage::DoesTileExist(const GraphId& graphid, const TileHierarchy& tile_hierarchy) const {
  std::string file_location = tile_dir_ + "/" + FileSuffix(graphid.Tile_Base(), tile_hierarchy);
  struct stat buffer;
  return stat(file_location.c_str(), &buffer) == 0;
}

bool GraphTileFsStorage::ReadTile(const GraphId& graphid, const TileHierarchy& tile_hierarchy, std::vector<char>& tile_data) const {
  // Open to the end of the file so we can immediately get size;
  std::string file_location = tile_dir + "/" + FileSuffix(graphid.Tile_Base(), tile_hierarchy);
  std::ifstream file(file_location, std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open()) {
    // Read binary file into memory. TODO - protect against failure to
    // allocate memory
    size_t filesize = file.tellg();
    tile_data.resize(filesize);
    file.seekg(0, std::ios::beg);
    size_t readsize = file.read(tile_data.data(), filesize);
    file.close();
    return readsize == filesize;
  }
  return false;
}

bool GraphTileFsStorage::ReadTileRealTimeSpeeds(const GraphId& graphid, const TileHierarchy& tile_hierarchy, std::vector<uint8_t>& rts_data) const {
  // Try to load the speeds file
  std::string traffic_dir = tile_dir_ + "/traffic/";
  std::string file_location = traffic_dir + std::to_string(tileid) + ".spd";
  std::ifstream rtsfile(file_location, std::ios::binary | std::ios::in | std::ios::ate);
  if (rtsfile.is_open()) {
    size_t filesize = rtsfile.tellg();
    LOG_INFO("Load real time speeds: count = " + std::to_string(filesize));
    rts_data.resize(filesize);
    rtsfile.seekg(0, std::ios::beg);
    size_t readsize = rtsfile.read((char*)(&rts_data.front()), filesize);
    rtsfile.close();
    return readsize == filesize;
  }
  return false;
}

// Get the tile Id given the full path to the file.
GraphId GraphTileFsStorage::GetTileId(const std::string& fname, const std::string& tile_dir) {
  //strip off the unuseful part
  auto pos = fname.find(tile_dir);
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

std::string GraphTileFsStorage::FileSuffix(const GraphId& graphid, const TileHierarchy& tile_hierarchy) {
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
