

#include "film.h"

Tile::Tile(uint oc, uint oy, uint w, uint h)
    : origin_x(oc)
    , origin_y(oy)
    , width(w)
    , height(h)
{
    buf = std::unique_ptr<Pixel[]>(new Pixel[w * h]);
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

bool Film::write_tile()
{
    return true;
}
