#pragma once

#include <string>

#include "base/basic_types.h"
#include "core/scene.h"

class Renderer {
public:
    Renderer() {}
    Renderer(const uint spl_cnt, const uint nth)
        : sample_count(spl_cnt)
        , nthreads(nth)
    {}

    inline bool load_scene(const std::string& filepath) {
        scene.parse_from_file(filepath);
        return true;
    }

    bool render(const std::string& output="./test.jpg");

    Scene scene;
    uint sample_count = 10;
    uint nthreads = 0;
};