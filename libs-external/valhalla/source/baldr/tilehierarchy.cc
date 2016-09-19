#include "baldr/tilehierarchy.h"
#include "baldr/graphtileheader.h"

using namespace valhalla::midgard;

namespace valhalla {
namespace baldr {

TileHierarchy::TileHierarchy(const std::string& tile_dir):tile_dir_(tile_dir) {
  levels_ = {
    {(uint8_t)2, TileLevel{(uint8_t)2, stringToRoadClass.find("ServiceOther")->second, "local", Tiles<PointLL>{{{-180, -90}, {180, 90}}, .25, (unsigned short)kBinsDim}}},
    {(uint8_t)1, TileLevel{(uint8_t)1, stringToRoadClass.find("Tertiary")->second, "arterial", Tiles<PointLL>{{{-180, -90}, {180, 90}}, 1, (unsigned short)kBinsDim}}},
    {(uint8_t)0, TileLevel{(uint8_t)0, stringToRoadClass.find("Primary")->second, "highway", Tiles<PointLL>{{{-180, -90}, {180, 90}}, 4, (unsigned short)kBinsDim}}}
  };
}

TileHierarchy::TileHierarchy(){}

const std::map<unsigned char, TileHierarchy::TileLevel>& TileHierarchy::levels() const {
  return levels_;
}

const std::string& TileHierarchy::tile_dir() const {
  return tile_dir_;
}

GraphId TileHierarchy::GetGraphId(const midgard::PointLL& pointll, const unsigned char level) const {
  GraphId id;
  const auto& tl = levels_.find(level);
  if(tl != levels_.end()) {
    id.Set(static_cast<int32_t>(tl->second.tiles.TileId(pointll)), level, 0);
  }
  return id;
}

}
}
