// cuda_video.h
//
// Shared ffmpeg helper for converting AVI captures to compact GIFs using
// the palettegen + paletteuse pipeline.  Host-only; safe to include from
// .cpp or .cu translation units.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace cudabot {

// Convert an AVI file to a GIF using a 2-pass palette pipeline.
// scale_w controls the long-edge width in pixels (use 720 / 900 / 1080
// depending on how dense the demo is).
inline void avi_to_gif(const std::string& avi, const std::string& gif,
                       int fps = 24, int scale_w = 720) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -i %s "
                  "-vf \"fps=%d,scale=%d:-1:flags=lanczos,"
                  "split[a][b];[a]palettegen=stats_mode=diff[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle\" "
                  "%s 2>/dev/null",
                  avi.c_str(), fps, scale_w, gif.c_str());
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg failed (%d) for %s\n", rc, gif.c_str());
}

}  // namespace cudabot
