#include "baldr/tilehierarchy.h"
#include "baldr/graphtileheader.h"

using namespace valhalla::midgard;

namespace valhalla {
namespace baldr {

TileHierarchy::TileHierarchy(const std::shared_ptr<GraphTileStorage>& tile_storage) :tile_storage_(tile_storage) {
  levels_ = {
    {2, TileLevel{2, stringToRoadClass.find("ServiceOther")->second, "local", Tiles<PointLL>{{{-180, -90}, {180, 90}}, .25, static_cast<unsigned short>(kBinsDim)}}},
    {1, TileLevel{1, stringToRoadClass.find("Tertiary")->second, "arterial", Tiles<PointLL>{{{-180, -90}, {180, 90}}, 1, static_cast<unsigned short>(kBinsDim)}}},
    {0, TileLevel{0, stringToRoadClass.find("Primary")->second, "highway", Tiles<PointLL>{{{-180, -90}, {180, 90}}, 4, static_cast<unsigned short>(kBinsDim)}}}
  };
}

TileHierarchy::TileHierarchy(){}

const std::map<unsigned char, TileHierarchy::TileLevel>& TileHierarchy::levels() const {
  return levels_;
}

const std::shared_ptr<GraphTileStorage>& TileHierarchy::tile_storage() const {
  return tile_storage_;
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
