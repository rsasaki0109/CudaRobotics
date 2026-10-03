// cuda_video.h
//
// Shared helpers for demo output: AVI capture, ffmpeg GIF conversion and
// output directories. Host-only; safe to include from .cpp or .cu
// translation units, and portable to Windows (no ffmpeg/POSIX shell needed
// for the AVI itself).

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

#include <opencv2/videoio.hpp>
#include <opencv2/videoio/registry.hpp>

#include "portable_io.h"

namespace cudabot {

// FOURCC for AVI captures. XVID needs an FFmpeg-enabled OpenCV (e.g. vcpkg
// builds ship without it), so fall back to OpenCV's built-in MJPG writer.
inline int avi_fourcc() {
    static const int code = cv::videoio_registry::hasBackend(cv::CAP_FFMPEG)
        ? cv::VideoWriter::fourcc('X', 'V', 'I', 'D')
        : cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    return code;
}

// Convert an AVI file to a GIF using a 2-pass palette pipeline.
// scale_w controls the long-edge width in pixels (use 720 / 900 / 1080
// depending on how dense the demo is).
inline void avi_to_gif(const std::string& avi, const std::string& gif,
                       int fps = 24, int scale_w = 720) {
    const size_t slash = gif.find_last_of("/\\");
    if (slash != std::string::npos) ensure_dirs({gif.substr(0, slash).c_str()});
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -i %s "
                  "-vf \"fps=%d,scale=%d:-1:flags=lanczos,"
                  "split[a][b];[a]palettegen=stats_mode=diff[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle\" "
                  "%s 2>" CUDABOT_NULL_DEVICE,
                  avi.c_str(), fps, scale_w, gif.c_str());
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg failed (%d) for %s\n", rc, gif.c_str());
}

// The demos' original single-pass conversion (scale, optional lanczos, loop
// forever). Kept so regenerated GIFs match the published ones; new demos should
// prefer avi_to_gif.
inline void avi_to_gif_simple(const std::string& avi, const std::string& gif,
                              int fps, int scale_w, bool lanczos = true) {
    const size_t slash = gif.find_last_of("/\\");
    if (slash != std::string::npos) ensure_dirs({gif.substr(0, slash).c_str()});
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -i %s -vf \"fps=%d,scale=%d:-1%s\" -loop 0 %s 2>" CUDABOT_NULL_DEVICE,
                  avi.c_str(), fps, scale_w, lanczos ? ":flags=lanczos" : "", gif.c_str());
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg failed (%d) for %s\n", rc, gif.c_str());
}

}  // namespace cudabot
