#include <iostream>
#include <utility>
#include <vector>

#include "baldr/streetnames.h"
#include "baldr/streetnames_factory.h"
#include "baldr/streetnames_us.h"
#include "midgard/util.h"

// CARTOHACK
namespace stdext {
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
}

namespace valhalla {
namespace baldr {

std::unique_ptr<StreetNames>
StreetNamesFactory::Create(const std::string& country_code,
                           const std::vector<std::pair<std::string, bool>>& names) {
  if (country_code == "US") {
    return stdext::make_unique<StreetNamesUs>(names);
  }

  return stdext::make_unique<StreetNames>(names);
}

} // namespace baldr
} // namespace valhalla
