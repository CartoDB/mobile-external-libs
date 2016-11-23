#ifndef VALHALLA_BALDR_GRAPHTILEMBTSTORAGE_H_
#define VALHALLA_BALDR_GRAPHTILEMBTSTORAGE_H_

#include <tuple>
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
   * @param mbt_file The name of the MBTiles file that stores the tiles.
   */
  GraphTileMBTStorage(const std::string& mbt_file);

  /**
   * Destructor
   */
  ~GraphTileMBTStorage() = default;

  /**
   * Gets the list of all tile ids available given tile hierarchy.
   * @param  tile_hierarchy The tile hierachy to use.
   * @return Returns the list of all available tile ids.
   */
  std::vector<GraphId> FindTiles(const TileHierarchy& tile_hierarchy) const override;

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

  static std::tuple<int, int, int> FromGraphId(const GraphId& graphid, const TileHierarchy& tile_hierarchy);

  static GraphId ToGraphId(const std::tuple<int, int, int>& tile_coords, const TileHierarchy& tile_hierarchy);

  std::shared_ptr<sqlite3pp::database> mbt_db_;

};

}
}

#endif  // VALHALLA_BALDR_GRAPHTILEMBTSTORAGE_H_
