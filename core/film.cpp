

#include "film.h"

Tile::Tile(uint oc, uint oy, uint w, uint h)
    : origin_x(oc)
    , origin_y(oy)
    , width(w)
    , height(h)
{
    buf = std::unique_ptr<Pixel[]>(new Pixel[w * h]);
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

Film::Film(unsigned int w, unsigned int h, const std::string& f)
    : width(w)
    , height(h)
    , filename(f)
{
    output = OIIO::ImageOutput::create(f);
    spec = OIIO::ImageSpec(w, h, 3, OIIO::TypeDesc::UINT8);
    output->open(f, spec);
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

bool Film::write(void* data)
{
    output->write_image(OIIO::TypeDesc::UINT8, data);
    return true;
}

void Film::generate_tiles(uint xres, uint yres)
{
    uint regular_tile_width = width / xres;
    uint regular_tile_height = height / yres;

    for (int j = 0; j < yres; j++) {
        uint tile_height = j == yres - 1 ? regular_tile_height : tile_height - regular_tile_height * (yres - 1);
        for (int i = 0; i < xres; i++) {
            uint tile_width = i == xres - 1 ? regular_tile_width : tile_width - regular_tile_width * (xres - 1);
            tiles.emplace_back(Tile(i * regular_tile_width, j * regular_tile_height, tile_width, tile_height));
        }
    }
}

bool Film::write_tiles()
{
    for (auto& tile : tiles)
        output->write_tile(tile.origin_x, tile.origin_y, 0, OIIO::TypeDesc::FLOAT, tile.get_data_ptr());
    return true;
}
