#include "core/scene.h"

int main() {
    /*
    // Camera
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

    // Integrator
    Integrator integrator{&cam, &film};
    ListAccel list;

    // Construct scene
    Transform t1;
    t1.translate(Vec3f{0.f, 5.f, -20.f});

    LambertianBxDF lamb{RGBSpectrum{0.8f, 0.4f, 0.1f}};
    Material mat;
    mat.bxdf = &lamb;

    list.add_hitable(std::make_shared<Sphere>(t1, 0, 5.f, &mat));

    Transform t2;
    t2.translate(Vec3f{10.f, 3.f, -20.f});

    MetalBxDF metal1{RGBSpectrum{0.9f, 0.9f, 0.5f}};
    Material mat2;
    mat2.bxdf = &metal1;

    list.add_hitable(std::make_shared<Sphere>(t2, 0, 3.f, &mat2));

    Transform tt;
    tt.translate(Vec3f{0.f, 0.f, -10.f});
    Triangle t{tt, Vec3f{0.f, 0.f, 0.f}, Vec3f{2.f, 0.f, 0.f}, Vec3f{0.f, 2.f, 0.f}, &mat};

    list.add_hitable(std::make_shared<Triangle>(tt, Vec3f{0.f, 0.f, 0.f}, Vec3f{2.f, 0.f, 0.f}, Vec3f{0.f, 2.f, 0.f}, &mat));

    Transform tb;
    tb.translate(Vec3f{0.f, -1000.f, -20.f});

    LambertianBxDF lamb2{RGBSpectrum{0.5f, 0.5f, 0.5f}};
    Material matb;
    matb.bxdf = &lamb2;

    list.add_hitable(std::make_shared<Sphere>(tb, 1, 1000.f, &matb));

    auto triangle_meshes = load_triangle_mesh("../resource/obj/cube.obj", &mat);

    list.add_hitables(triangle_meshes);

    std::cout << "list size : " << list.size() << std::endl;

    BVHAccel bvh(list.hitables, 0, list.size());
    bvh.print_bound();

    integrator.accel_ptr = &bvh;

    // Lights
    PointLight pt1{RGBSpectrum{0.3f, 0.9f, 0.3f}, Vec3f{0.f, 10.f, -20.f}};
    integrator.lights.push_back(&pt1);

    // Start render
    integrator.render();

    // Write out rendered image
    film.write_tiles();
    */

    Scene scene;
    scene.parse_from_file("../resource/scene/test_scene.xml");
    scene.integrator->render();
    scene.film->write_tiles();

    return 0;
}