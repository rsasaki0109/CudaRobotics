#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include "cpprobotics_types.h"

static int g_fail = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        std::printf("  PASS: %s\n", name);
    } else {
        std::printf("  FAIL: %s\n", name);
        g_fail++;
    }
}

static void test_types() {
    std::printf("[test_types]\n");

    cpprobotics::Vec_f v = {1.0f, 2.0f, 3.0f};
    check(v.size() == 3, "Vec_f size");
    check(std::fabs(v[1] - 2.0f) < 1e-6f, "Vec_f element access");

    cpprobotics::Poi_f p = {4.0f, 5.0f};
    check(std::fabs(p[0] - 4.0f) < 1e-6f, "Poi_f x");
    check(std::fabs(p[1] - 5.0f) < 1e-6f, "Poi_f y");

    cpprobotics::Vec_Poi vp = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    check(vp.size() == 2, "Vec_Poi size");
    check(std::fabs(vp[1][0] - 3.0f) < 1e-6f, "Vec_Poi element access");
}

static void test_csv_parse() {
    std::printf("[test_csv_parse]\n");

    // Simulate parsing a simple CSV line "1.5,2.5,3.5"
    std::string line = "1.5,2.5,3.5";
    cpprobotics::Vec_f values;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        values.push_back(std::stof(token));
    }
    check(values.size() == 3, "CSV field count");
    check(std::fabs(values[0] - 1.5f) < 1e-6f, "CSV field 0");
    check(std::fabs(values[1] - 2.5f) < 1e-6f, "CSV field 1");
    check(std::fabs(values[2] - 3.5f) < 1e-6f, "CSV field 2");

    // Empty line produces no tokens
    cpprobotics::Vec_f empty_vals;
    std::stringstream ss2("");
    while (std::getline(ss2, token, ',')) {
        if (!token.empty()) empty_vals.push_back(std::stof(token));
    }
    check(empty_vals.empty(), "CSV empty line");
}

int main() {
    std::printf("=== test_host_math ===\n");
    test_types();
    test_csv_parse();
    std::printf("======================\n");
    if (g_fail == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) FAILED.\n", g_fail);
    return 1;
}
