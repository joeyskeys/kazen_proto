#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "base/basic_types.h"
#include "core/scene.h"

class RenderCallback {
public:
    virtual void on_tile_end(Film* film, uint32_t tile_id) {
        std::cout << "default tile end impl: " << tile_id << std::endl;
    }
};

class Renderer {
public:
    Renderer() {}
    Renderer(const uint spl_cnt, const uint nth,
        RenderCallback* const cbk=nullptr)
        : sample_count(spl_cnt)
        , nthreads(nth)
        , render_cbk(cbk)
    {}

    /*
    inline bool load_scene(const std::string& filepath) {
        scene.parse_from_file(filepath);
        return true;
    }
    */

    bool render(const std::string& scene_file, const std::string& output="./test.jpg") const;
    bool render(Scene& scene, const std::string& output="./test.jpg") const;
    bool render_debug(Scene& scene, const uint32_t x, const uint32_t y) const;

    //Scene scene;
    uint sample_count = 10;
    uint nthreads = 0;
    RenderCallback* render_cbk;
};