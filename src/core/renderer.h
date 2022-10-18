#pragma once

#include <string>

#include "base/basic_types.h"
#include "core/scene.h"

class Renderer {
public:
    Renderer() {}
    Renderer(const uint spl_cnt, const uint nth, const std::string o)
        : sampler_cnt(spl_cnt)
        , nthreads(nth)
        , output(o)
    {}

    bool render();

    Scene scene;
    uint sampler_cnt = 10;
    uint nthreads = 0;
    std::string output = "./test.jpg";
};