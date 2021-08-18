#pragma once

#include <memory>
#include <vector>

#include <OpenImageIO/imageio.h>

#include "base/types.h"
#include "core/pixel.h"
#include "core/spectrum.h"

class Tile {
public:
    Tile(uint oc, uint oy, uint w, uint h);

    void set_pixel_color(uint x, uint y, const RGBSpectrum& s);
    void set_tile_color(const RGBSpectrum& s);

    inline float* const get_data_ptr() {
        return reinterpret_cast<float*>(buf.get());
    }

public:
    const uint origin_x;
    const uint origin_y;
    const uint width;
    const uint height;

private:
    std::unique_ptr<Pixel[]> buf;
};

class Film {
public:
    Film(uint w, uint h, const std::string& f);
    Film(uint w, uint h, std::string&& f);

    bool write(void* data, OIIO::TypeDesc pixel_format);

    void generate_tiles(const uint xres=4, const uint yres=4);
    bool write_tiles();

    inline void set_film_color(const RGBSpectrum& s) {
        for (auto& tile : tiles)
            tile.set_tile_color(s);
    }

    inline void set_tile_color(const RGBSpectrum& s, const uint xidx, const uint yidx) {
        tiles[tile_res_x * yidx + xidx].set_tile_color(s);
    }

    inline auto get_tile_count() {
        return tiles.size();
    }

public:
    const uint width;
    const uint height;
    std::string  filename;

    uint tile_res_x, tile_res_y;
    uint tile_width, tile_height;

    std::vector<Tile> tiles;

private:
    std::unique_ptr<OIIO::ImageOutput>  output;
    OIIO::ImageSpec     spec;
};
