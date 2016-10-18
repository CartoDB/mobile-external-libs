#ifndef VALHALLA_BALDR_GRAPHTILEZIPSTORAGE_H_
#define VALHALLA_BALDR_GRAPHTILEZIPSTORAGE_H_

#include <valhalla/baldr/graphtilestorage.h>

namespace valhalla {
namespace baldr {

class GraphTileZipStorage : public GraphTileStorage {
 public:

  /**
   * Constructor
   * @param zip_file The name of the zip file that stores the tiles.
   */
  GraphTileZipStorage(const std::string& zip_file);

  /**
   * Destructor
   */
  ~GraphTileZipStorage() = default;

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

  static GraphId GetTileId(const std::string& fname, const std::string& tile_dir);

  static std::string FileSuffix(const GraphId& graphid, const TileHierarchy& tile_hierarchy);

  std::string zip_file_;

};

}
}

#endif  // VALHALLA_BALDR_GRAPHTILEZIPSTORAGE_H_
