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
    scene.parse_from_file("../resource/scene/cornell_box/cornell_box.xml");

    //scene.integrator->render();
    constexpr static int sample_count = 64;

    auto render_start = get_time();

#define WITH_TBB

#ifdef WITH_TBB
    tbb::parallel_for (tbb::blocked_range<size_t>(0, scene.film->tiles.size()),
        [&](const tbb::blocked_range<size_t>& r) {
#else
    {
#endif

        OSL::PerThreadInfo *thread_info = scene.shadingsys->create_thread_info();
        OSL::ShadingContext *ctx = scene.shadingsys->get_context(thread_info);

#ifdef WITH_TBB
        for (int t = r.begin(); t != r.end(); ++t) {
#else
        for (int t = 0; t < scene.film->tiles.size(); ++t) {
#endif

            auto tile_start = get_time();
            Tile& tile = scene.film->tiles[t];

            for (int j = 0; j < tile.height; j++) {
                for (int i = 0; i < tile.width; i++) {
                    RGBSpectrum pixel_radiance{0};

                    for (int s = 0; s < sample_count; s++) {
                        uint x = tile.origin_x + i;
                        uint y = tile.origin_y + j;

                        auto ray = scene.camera->generate_ray(x, y);
                        pixel_radiance += scene.integrator->Li(r);
                    }

                    pixel_radiance /= sample_count;
                    tile.set_pixel_color(i, j, pixel_radiance);
                }
            }

            auto tile_end = get_time();
            auto tile_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tile_end - tile_start);
            std::cout << "tile duration : " << tile_duration.count() << " ms\n";
        }

        scene.shadingsys->release_context(ctx);
        scene.shadingsys->destroy_thread_info(thread_info);

#ifdef WITH_TBB
    });
#else
    }
#endif

    auto render_end = get_time();
    auto render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(render_end - render_start);
    std::cout << "render duration : " << render_duration.count() << " ms\n";

    scene.film->write_tiles();

    return 0;
}