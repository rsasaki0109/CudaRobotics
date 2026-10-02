// portable_io.h
//
// Dependency-free helpers for demo output paths and shell commands that work
// with both POSIX shells and Windows cmd.

#pragma once

#include <cerrno>
#include <initializer_list>
#include <string>

#ifdef _WIN32
#include <direct.h>
#define CUDABOT_NULL_DEVICE "NUL"
#else
#include <sys/stat.h>
#define CUDABOT_NULL_DEVICE "/dev/null"
#endif

namespace cudabot {

// `mkdir -p` for each path without going through the shell (cmd's mkdir has
// no -p and would create a "-p" directory). Returns 0 when all directories
// exist afterwards.
inline int ensure_dirs(std::initializer_list<const char*> paths) {
    int rc = 0;
    for (const char* p : paths) {
        const std::string path(p);
        std::string partial;
        for (size_t i = 0; i <= path.size(); ++i) {
            if ((i == path.size() || path[i] == '/' || path[i] == '\\') && !partial.empty()) {
#ifdef _WIN32
                int made = _mkdir(partial.c_str());
#else
                int made = mkdir(partial.c_str(), 0755);
#endif
                if (made != 0 && errno != EEXIST) rc = -1;
            }
            if (i < path.size()) partial += path[i];
        }
    }
    return rc;
}

}  // namespace cudabot
