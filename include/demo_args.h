// demo_args.h
//
// Minimal command-line options shared by the demos, so parameters can be
// changed without recompiling. Host-only, no dependencies.
//
//   int main(int argc, char** argv) {
//       cudabot::DemoArgs args(argc, argv, "CUDA MPPI with a bicycle model");
//       const int K = args.get_int("samples", 4096, "sampled trajectories per step", 1);
//       const bool video = !args.flag("no-video", "skip the AVI/GIF output");
//       args.finish();   // handles --help / --headless, rejects unknown options
//
// Options accept "--name value" and "--name=value". Every demo using this
// also understands --headless (same as CUDABOT_HEADLESS=1) and --help.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace cudabot {

class DemoArgs {
public:
    DemoArgs(int argc, char** argv, const char* summary)
        : program_(argc > 0 ? base_name(argv[0]) : "demo"), summary_(summary) {
        for (int i = 1; i < argc; ++i) args_.push_back(argv[i]);
        used_.assign(args_.size(), false);
    }

    int get_int(const char* name, int fallback, const char* help, int min_value = -2147483647) {
        std::string value;
        add_help(name, "N", help, std::to_string(fallback));
        if (!take(name, true, value)) return fallback;
        char* end = nullptr;
        long parsed = std::strtol(value.c_str(), &end, 10);
        if (value.empty() || *end != '\0' || parsed < min_value || parsed > 2147483647L)
            fail("--" + std::string(name) + " expects an integer >= " + std::to_string(min_value) +
                 ", got '" + value + "'");
        return static_cast<int>(parsed);
    }

    float get_float(const char* name, float fallback, const char* help) {
        std::string value;
        add_help(name, "X", help, trim_float(fallback));
        if (!take(name, true, value)) return fallback;
        char* end = nullptr;
        float parsed = std::strtof(value.c_str(), &end);
        if (value.empty() || *end != '\0')
            fail("--" + std::string(name) + " expects a number, got '" + value + "'");
        return parsed;
    }

    bool flag(const char* name, const char* help) {
        std::string unused;
        add_help(name, "", help, "");
        return take(name, false, unused);
    }

    // Call once after all options were read.
    void finish() {
        if (flag("headless", "no windows or key waits (same as CUDABOT_HEADLESS=1)")) {
#ifdef _WIN32
            _putenv_s("CUDABOT_HEADLESS", "1");
#else
            setenv("CUDABOT_HEADLESS", "1", 1);
#endif
        }
        const bool help = flag("help", "show this message");
        for (size_t i = 0; i < args_.size(); ++i)
            if (!used_[i]) fail("unknown option '" + args_[i] + "'");
        if (help) {
            print_usage(stdout);
            std::exit(0);
        }
    }

private:
    struct HelpLine {
        std::string option, help;
    };

    static std::string base_name(const char* path) {
        std::string s(path);
        size_t slash = s.find_last_of("/\\");
        if (slash != std::string::npos) s = s.substr(slash + 1);
        if (s.size() > 4 && s.compare(s.size() - 4, 4, ".exe") == 0) s.resize(s.size() - 4);
        return s;
    }

    static std::string trim_float(float v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%g", v);
        return buf;
    }

    void add_help(const char* name, const char* metavar, const char* help, const std::string& fallback) {
        std::string option = "--" + std::string(name) + (*metavar ? " " + std::string(metavar) : "");
        std::string text = help;
        if (!fallback.empty()) text += " (default " + fallback + ")";
        help_.push_back({option, text});
    }

    // Finds --name / --name=value; marks the consumed arguments.
    bool take(const char* name, bool needs_value, std::string& value) {
        const std::string key = "--" + std::string(name);
        bool found = false;
        for (size_t i = 0; i < args_.size(); ++i) {
            if (used_[i]) continue;
            const std::string& a = args_[i];
            if (a == key) {
                used_[i] = true;
                found = true;
                if (needs_value) {
                    if (i + 1 >= args_.size()) fail(key + " expects a value");
                    value = args_[++i];
                    used_[i] = true;
                }
            } else if (needs_value && a.compare(0, key.size() + 1, key + "=") == 0) {
                used_[i] = true;
                found = true;
                value = a.substr(key.size() + 1);
            }
        }
        return found;
    }

    void print_usage(std::FILE* out) const {
        std::fprintf(out, "usage: %s [options]\n%s\n\noptions:\n", program_.c_str(), summary_.c_str());
        for (const HelpLine& h : help_) std::fprintf(out, "  %-18s %s\n", h.option.c_str(), h.help.c_str());
    }

    [[noreturn]] void fail(const std::string& message) const {
        std::fprintf(stderr, "%s: %s\n\n", program_.c_str(), message.c_str());
        print_usage(stderr);
        std::exit(2);
    }

    std::string program_, summary_;
    std::vector<std::string> args_;
    std::vector<bool> used_;
    std::vector<HelpLine> help_;
};

}  // namespace cudabot
