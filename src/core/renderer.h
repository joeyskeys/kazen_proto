#pragma once

#include <iostream>
#include <string>

#include "base/basic_types.h"
#include "core/scene.h"

class RenderCallback {
public:
    virtual void on_tile_end(Film* film) {
        std::cout << "default tile end impl" << std::endl;
    }
};

class Renderer {
public:
    Renderer() {}
    Renderer(const uint spl_cnt, const uint nth,
        const RenderCallback cbk=RenderCallback())
        : sample_count(spl_cnt)
        , nthreads(nth)
        ,render_cbk(cbk)
    {}

    inline bool load_scene(const std::string& filepath) {
        scene.parse_from_file(filepath);
        return true;
    }

    bool render(const std::string& output="./test.jpg");

    Scene scene;
    uint sample_count = 10;
    uint nthreads = 0;
    RenderCallback render_cbk;
};