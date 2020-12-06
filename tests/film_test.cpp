

#include "core/film.h"

#include <cstdio>
#include <memory>

int main()
{
    Film film(640, 480, "test.jpg");
    //std::unique_ptr<char[]> buf{ new char[640 * 480 * 3 * 1] };
    //memset(buf.get(), 166, 640 * 480 * 3 * 1);
    //film.write(buf.get());
    film.generate_tiles();
    film.set_film_color({.1f, .1f, .1f});
    film.set_tile_color({.9f, .1f, .1f}, 1, 1);
    film.write_tiles();
}
