#ifndef VALHALLA_BALDR_GRAPHFSREADER_H_
#define VALHALLA_BALDR_GRAPHFSREADER_H_

#include <valhalla/baldr/graphreader.h>

namespace valhalla {
namespace baldr {

/**
 * A shortcut graph reader class that uses file system storage.
 */
class GraphFsReader : public GraphReader {
 public:
  /**
   * Constructor
   * @param pt the configuration for the tilehierarchy
   */
  GraphFsReader(const boost::property_tree::ptree& pt);
};

}
}

#endif  // VALHALLA_BALDR_GRAPHFSREADER_H_
