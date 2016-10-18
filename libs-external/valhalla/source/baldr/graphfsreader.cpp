#include "baldr/graphfsreader.h"
#include "baldr/graphtilefsstorage.h"

namespace valhalla {
namespace baldr {

GraphFsReader::GraphFsReader(const boost::property_tree::ptree& pt)
    : GraphReader(std::make_shared<GraphTileFsStorage>(pt.get<std::string>("tile_dir")), pt) {
}

}
}
