#include <tbb/tbb.h>

#include "core/scene.h"
#include "core/state.h"

int main() {
    Scene scene;
    scene.parse_from_file("../resource/scene/cornell_box/cornell_box.xml");

    constexpr static int sample_count = 5;

    auto render_start = get_time();
    bool hit = false;

#define WITH_TBB

#ifdef WITH_TBB
    tbb::parallel_for (tbb::blocked_range<size_t>(0, scene.film->tiles.size()),
        [&](const tbb::blocked_range<size_t>& r) {
#else
    {
#endif

#ifdef WITH_TBB
        for (int t = r.begin(); t != r.end(); ++t) {
#else
        for (int t = 0; t < scene.film->tiles.size(); ++t) {
#endif

            auto tile_start = get_time();
            Tile& tile = scene.film->tiles[t];
            auto integrator_ptr = scene.integrator_fac.create(scene.camera.get(), scene.film.get());
            integrator_ptr->setup(&scene);

            for (int j = 0; j < tile.height; j++) {
                for (int i = 0; i < tile.width; i++) {
                    RGBSpectrum pixel_radiance{0};

                    for (int s = 0; s < sample_count; s++) {
                        uint x = tile.origin_x + i;
                        uint y = tile.origin_y + j;

                        auto ray = scene.camera->generate_ray(x, y);
                        pixel_radiance += integrator_ptr->Li(ray);
                        if (!pixel_radiance.is_zero())
                            hit = true;
                    }

                    pixel_radiance /= sample_count;
                    tile.set_pixel_color(i, j, pixel_radiance);
                }
            }

            auto tile_end = get_time();
            auto tile_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tile_end - tile_start);
            std::cout << "tile duration : " << tile_duration.count() << " ms\n";
        }

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