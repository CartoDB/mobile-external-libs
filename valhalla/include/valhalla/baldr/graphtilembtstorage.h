#ifndef VALHALLA_BALDR_GRAPHTILEMBTSTORAGE_H_
#define VALHALLA_BALDR_GRAPHTILEMBTSTORAGE_H_

#include <tuple>
#include <memory>
#include <vector>
#include <valhalla/baldr/graphtilestorage.h>

namespace sqlite3pp {
  class database;
}

namespace valhalla {
namespace baldr {

class GraphTileMBTStorage : public GraphTileStorage {
 public:

  /**
   * Constructor
   * @param dbs The routing databases.
   */
  GraphTileMBTStorage(const std::vector<std::shared_ptr<sqlite3pp::database>>& dbs);

  /**
   * Destructor
   */
  ~GraphTileMBTStorage() = default;

  /**
   * Gets the list of all tile ids available given tile hierarchy.
   * @param  tile_hierarchy The tile hierachy to use.
   * @return Returns the list of all available tile ids.
   */
  std::unordered_set<GraphId> FindTiles(const TileHierarchy& tile_hierarchy) const override;

  /**
   * Checks if the specified tile exists.
   * @param  graphid        The tile id to check.
   * @param  tile_hierarchy The tile hierachy to use.
   * @return Returns true if the tile exists and false otherwise.
   */
  bool DoesTileExist(const GraphId& graphid, const TileHierarchy& tile_hierarchy) const override;

  /**
   * Reads the specified tile.
   * @param graphid        The tile id to read.
   * @param tile_hierarchy The tile hierachy to use.
   * @param tile_data      The buffer to use for storing the raw tile data.
   * @return Returns true if the tile exists and false otherwise.
   */
  bool ReadTile(const GraphId& graphid, const TileHierarchy& tile_hierarchy, std::vector<char>& tile_data) const override;

  /**
   * Reads the optional Real-Time-Speeds associated with the tile.
   * @param  graphid        The tile id to read.
   * @param  tile_hierarchy The tile hierachy to use.
   * @param  rts_data       The buffer to use for storing real-time-speed data.
   * @return Returns true if the RTS data exists for the tile and was successfully read and false otherwise.
   */
  bool ReadTileRealTimeSpeeds(const GraphId& graphid, const TileHierarchy& tile_hierarchy, std::vector<uint8_t>& rts_data) const override;

 private:

  struct MBTDatabase {
    std::shared_ptr<sqlite3pp::database> database;
    std::shared_ptr<std::vector<unsigned char>> zdict;
  };

  static std::tuple<int, int, int> FromGraphId(const GraphId& graphid, const TileHierarchy& tile_hierarchy);

  static GraphId ToGraphId(const std::tuple<int, int, int>& tile_coords, const TileHierarchy& tile_hierarchy);

  std::vector<MBTDatabase> mbt_dbs_;

};

}
}

#endif  // VALHALLA_BALDR_GRAPHTILEMBTSTORAGE_H_
