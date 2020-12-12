

#include "core/film.h"

#include <cstdio>
#include <memory>
#include <array>
#include <algorithm>

int main()
{
    Film film1(640, 480, "write_buffer.jpg");
    std::array<float, 640 * 480 * 3> buf;
    std::fill(buf.begin(), buf.end(), 0.1f);
    film1.write(buf.data(), OIIO::TypeDesc::FLOAT);
    
    Film film2(640, 480, "write_tiles.jpg");
    film2.generate_tiles();
    film2.set_film_color({.1f, .1f, .1f});
    film2.set_tile_color({.9f, .4f, .1f}, 1, 1);
    film2.write_tiles();
}
