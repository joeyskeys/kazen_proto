

#include "core/film.h"

#include <cstdio>
#include <memory>

int main()
{
    Film film(640, 480, "test.jpg");
    std::unique_ptr<char[]> buf{ new char[640 * 480 * 3 * 1] };
    memset(buf.get(), 255, 640 * 480 * 3 * 1);
    film.write(buf.get());
}
