// display.h
//
// OpenCV window helpers that turn into no-ops in headless runs, so demos can
// finish unattended (CI, SSH, containers) instead of blocking on
// waitKey(0) or failing to open a display. Host-only.
//
// Headless when CUDABOT_HEADLESS is set to anything other than "0", or, on
// Linux/BSD, when neither DISPLAY nor WAYLAND_DISPLAY is set.

#pragma once

#include <cstdlib>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace cudabot {

inline bool headless() {
    static const bool value = [] {
        const char* env = std::getenv("CUDABOT_HEADLESS");
        if (env && *env) return std::string(env) != "0";
#if defined(_WIN32) || defined(__APPLE__)
        return false;
#else
        return !std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY");
#endif
    }();
    return value;
}

inline void imshow(const std::string& window, cv::InputArray image) {
    if (!headless()) cv::imshow(window, image);
}

// Returns -1 immediately when headless (as if no key was pressed).
inline int waitKey(int delay_ms = 0) {
    return headless() ? -1 : cv::waitKey(delay_ms);
}

inline void namedWindow(const std::string& window, int flags = cv::WINDOW_AUTOSIZE) {
    if (!headless()) cv::namedWindow(window, flags);
}

inline void resizeWindow(const std::string& window, int width, int height) {
    if (!headless()) cv::resizeWindow(window, width, height);
}

inline void destroyAllWindows() {
    if (!headless()) cv::destroyAllWindows();
}

}  // namespace cudabot
