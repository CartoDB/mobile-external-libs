#ifndef _UTF8_FILESYSTEM_H_INCLUDED_
#define _UTF8_FILESYSTEM_H_INCLUDED_

#include <cstdint>
#include <string>

#ifdef _WIN32
#include <utf8.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

namespace utf8_filesystem {

#ifdef _WIN32
    typedef struct ::__stat64 stat;
#else
    typedef struct ::stat stat;
#endif
    
    inline FILE* fopen(const char* fileName, const char* mode) {
#ifdef _WIN32
        std::wstring wfileName, wmode;
        utf8::utf8to16(fileName, fileName + strlen(fileName), std::back_inserter(wfileName));
        utf8::utf8to16(mode, mode + strlen(mode), std::back_inserter(wmode));
        return ::_wfopen(wfileName.c_str(), wmode.c_str());
#else
        return ::fopen(fileName, mode);
#endif
    }

    inline std::int64_t fseek64(FILE* fp, std::int64_t offset, int whence) {
#ifdef _WIN32
        return ::_fseeki64(fp, offset, whence);
#elif defined(__ANDROID__)
        return ::lseek64(fileno(fp), offset, whence);
#elif defined(__APPLE__)
        return ::fseeko(fp, offset, whence);
#endif
    }

    inline std::int64_t ftell64(FILE* fp) {
#ifdef _WIN32
        return ::_ftelli64(fp);
#elif defined(__ANDROID__)
        return ::lseek64(fileno(fp), 0, SEEK_CUR);
#elif defined(__APPLE__)
        return ::ftello(fp);
#endif
    }

    inline int ftruncate64(FILE* fp, std::int64_t size) {
#ifdef _WIN32
        return ::_chsize_s(_fileno(fp), size);
#elif defined(__ANRDOID__)
        return ::ftruncate64(fileno(fp), size);
#elif defined(__APPLE__)
        return ::ftruncate(fileno(fp), size);
#endif
    }

    inline int fstat(const char* fileName, stat* st) {
#ifdef _WIN32
        std::wstring wfileName;
        utf8::utf8to16(fileName, fileName + strlen(fileName), std::back_inserter(wfileName));
        return ::_wstat64(wfileName.c_str(), st);
#else
        return ::stat(fileName, st);
#endif
    }

    inline int rename(const char* oldFileName, const char* newFileName) {
#ifdef _WIN32
        std::wstring woldFileName, wnewFileName;
        utf8::utf8to16(oldFileName, oldFileName + strlen(oldFileName), std::back_inserter(woldFileName));
        utf8::utf8to16(newFileName, newFileName + strlen(newFileName), std::back_inserter(wnewFileName));
        return ::_wrename(woldFileName.c_str(), wnewFileName.c_str());
#else
        return ::rename(oldFileName, newFileName);
#endif
    }

    inline int unlink(const char* fileName) {
#ifdef _WIN32
        std::wstring wfileName;
        utf8::utf8to16(fileName, fileName + strlen(fileName), std::back_inserter(wfileName));
        return ::_wunlink(wfileName.c_str());
#else
        return ::unlink(fileName);
#endif
    }

} // namespace utf8_filesystem

#endif //_UTF8_FILESYSTEM_H_INCLUDED_
