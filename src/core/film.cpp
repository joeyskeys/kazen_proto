#include <iostream>
#include <cmath>

#include "film.h"

Tile::Tile(uint oc, uint oy, uint w, uint h)
    : origin_x(oc)
    , origin_y(oy)
    , width(w)
    , height(h)
{
    //buf = std::unique_ptr<Pixel[]>(new Pixel[w * h]);
    buf = std::make_unique<Pixel[]>(w * h);
}

void Tile::set_pixel_color(uint x, uint y, const RGBSpectrum& s)
{
    // No bound check here
    buf[y * width + x] = s;
}

void Tile::set_tile_color(const RGBSpectrum& s)
{
    for (int i = 0; i < width * height; i++)
        buf[i] = s;
}

Film::Film()
    : width(800)
    , height(600)
    , filename("test.jpg")
{
    output = OIIO::ImageOutput::create(filename);
    spec = OIIO::ImageSpec(width, height, OIIO::TypeDesc::UINT8);
}

Film::Film(unsigned int w, unsigned int h, const std::string& f)
    : width(w)
    , height(h)
    , filename(f)
{
    output = OIIO::ImageOutput::create(f);
    spec = OIIO::ImageSpec(w, h, 3, OIIO::TypeDesc::UINT8);
}

Film::Film(unsigned int w, unsigned int h, std::string&& f)
    : width(w)
    , height(h)
    , filename(f)
{
    output = OIIO::ImageOutput::create(f);
    spec = OIIO::ImageSpec(w, h, 3, OIIO::TypeDesc::UINT8);
    output->open(f, spec);
}

bool Film::write(void* data, OIIO::TypeDesc pixel_format)
{
    output->open(filename, spec);
    output->write_image(pixel_format, data);
    return true;
}

void Film::generate_tiles(uint xres, uint yres)
{
    tile_res_x = xres;
    tile_res_y = yres;
    
    tile_width = width / xres;
    tile_height = height / yres;

    for (int j = 0; j < yres; j++) {
        uint cur_tile_height = j == yres - 1 ? tile_height : height - tile_height * (yres - 1);
        for (int i = 0; i < xres; i++) {
            uint cur_tile_width = i == xres - 1 ? tile_width : width - tile_width * (xres - 1);
            tiles.emplace_back(Tile(i * tile_width, j * tile_height, cur_tile_width, cur_tile_height));
        }
    }
}

bool Film::write_tiles()
{
    spec.tile_width = tile_width;
    spec.tile_height = tile_height;
    output->open(filename, spec);

    auto xstride = spec.nchannels * sizeof(float);
    for (auto& tile : tiles) {
        auto ystride = xstride * std::min(width - tile.origin_x, tile_width);
        output->write_tile(tile.origin_x, tile.origin_y, 0,
            OIIO::TypeDesc::FLOAT, tile.get_data_ptr(),
            xstride, ystride);
    }
    return true;
}

void* Film::address_of(const std::string& name) {
    if (name == "width")
        return &width;
    else if (name == "height")
        return &height;
    else if (name == "filename")
        return &filename;
    return nullptr;
}