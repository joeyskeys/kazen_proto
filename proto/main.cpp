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
        Vec3f{7.f, 8.f, 15.f},
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
    t2.translate(Vec3f{10.f, 3.f, -20.f});
    Sphere s2{t2, 1, 3.f};

    MetalBxDF metal1{RGBSpectrum{0.9f, 0.9f, 0.5f}};
    Material mat2;
    mat2.bxdf = &metal1;
    s2.mat = &mat2;

    Transform tt;
    tt.translate(Vec3f{0.f, 0.f, -10.f});
    Triangle t{tt, Vec3f{0.f, 0.f, 0.f}, Vec3f{2.f, 0.f, 0.f}, Vec3f{0.f, 2.f, 0.f}};
    t.mat = &mat;

    Transform tb;
    tb.translate(Vec3f{0.f, -1000.f, -20.f});
    Sphere s_bottom{tb, 1, 1000.f};

    LambertianBxDF lamb2{RGBSpectrum{0.5f, 0.5f, 0.5f}};
    Material matb;
    matb.bxdf = &lamb2;
    s_bottom.mat = &matb;

    auto triangle_meshes = load_triangle_mesh("../resource/obj/cube.obj");
    auto triangle_mesh = triangle_meshes[0];
    triangle_mesh.mat = &mat;

    ListAccel list;
    list.add_hitable(&s);
    list.add_hitable(&s2);
    list.add_hitable(&t);
    list.add_hitable(&triangle_mesh);
    list.add_hitable(&s_bottom);
    integrator.accel_ptr = &list;

    integrator.render();

    film.write_tiles();

    return 0;
}