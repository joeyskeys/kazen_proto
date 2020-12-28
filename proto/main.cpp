#include "core/camera.h"
#include "core/film.h"
#include "core/integrator.h"

int main() {
    Film film{400, 300, "test.jpg"};
    film.generate_tiles();

    Camera cam{
        Vec3f{0.f, 0.f, 0.f},
        Vec3f{0.f, 0.f, -1.f},
        Vec3f{0.f, 1.f, 0.f},
        1.f,
        100.f,
        45.f,
        16.f / 9.f,
        &film
    };

    Integrator integrator{&cam, &film};
    integrator.render();

    film.write_tiles();

    return 0;
}