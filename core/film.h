#pragma once

#include <memory>

#include <OpenImageIO/imageio.h>

#include "base/types.h"
#include "core/pixel.h"

class Tile {
public:
    Tile(uint oc, uint oy, uint w, uint h);

    void set_pixel_world(uint x, uint y, Pixel&& p);
    void set_pixel_local(uint x, uint y, Pixel&& p);

private:
    uint origin_x;
    uint origin_y;
    uint width;
    uint height;

    std::unique_ptr<Pixel[]> buf;
};

class Film {
public:
    Film(uint w, uint h, const std::string& f);
    Film(uint w, uint h, std::string&& f);

    bool write(void* data);
    bool write_tile();

private:
    uint width;
    uint height;
    std::string  filename;
    std::unique_ptr<OIIO::ImageOutput>  output;
    OIIO::ImageSpec     spec;
};
