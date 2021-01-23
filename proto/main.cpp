#include "core/camera.h"
#include "core/film.h"
#include "core/integrator.h"
#include "core/material.h"
#include "core/shape.h"
#include "core/accel.h"

int main() {
    Film film{400, 300, "test.jpg"};
    film.generate_tiles();

    Camera cam{
        Vec3f{3.f, 8.f, 10.f},
        Vec3f{0.f, 3.f, -20.f},
        Vec3f{0.f, 1.f, 0.f},
        1.f,
        1000.f,
        60.f,
        &film
    };

    Integrator integrator{&cam, &film};

    Transform t1;
    t1.translate(Vec3f{0.f, 5.f, -20.f});
    Sphere s{t1, 0, 5.f};

    LambertianBxDF lamb{RGBSpectrum{0.8f, 0.4f, 0.1f}};
    Material mat;
    mat.bxdf = &lamb;
    s.mat = &mat;

    Transform t2;
    t2.translate(Vec3f{0.f, -1000.f, -20.f});
    Sphere s_bottom{t2, 1, 1000.f};

    LambertianBxDF lamb2{RGBSpectrum{0.5f, 0.5f, 0.5f}};
    Material mat2;
    mat2.bxdf = &lamb2;
    s_bottom.mat = &mat2;

    ListAccel list;
    list.add_hitable(&s);
    list.add_hitable(&s_bottom);
    integrator.accel_ptr = &list;

    integrator.render();

    film.write_tiles();

    return 0;
}