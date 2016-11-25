#pragma once

#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION "unstable"
#endif

#ifdef DEBUG
#undef DEBUG
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifdef __ANDROID__
#include <boost/lexical_cast.hpp>
#include <cmath>

namespace std {
  template<typename T>
  inline std::string to_string(T val) { return boost::lexical_cast<std::string>(val); }

  inline int stoi(const std::string& str) { return boost::lexical_cast<int>(str); }
  inline float stof(const std::string& str) { return boost::lexical_cast<float>(str); }
  inline double stod(const std::string& str) { return boost::lexical_cast<double>(str); }
  inline unsigned long stoul( const std::string& str) { return boost::lexical_cast<unsigned long>(str); }

  inline float round(float val) { return ceil(val - 0.5f); }
  inline double round(double val) { return ceil(val - 0.5); }
}

#endif

#ifdef _MSC_VER
#include <ctime>

inline int ffs(int x) {
    if (x == 0)
        return 0;
    int t = 1, r = 1;
    while ((x & t) == 0) {
        t <<= 1;
        r++;
    }
    return r;
}

inline void gmtime_r(std::time_t* tt, std::tm* gmt) {
  if (auto gmptr = gmtime(tt)) {
    *gmt = *gmptr;
  }
}
#endif
