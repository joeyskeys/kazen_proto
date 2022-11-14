#include "core/renderer.h"
#include "core/sampler.h"
#include "core/state.h"
#include "config.h"

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

bool Renderer::render(const std::string& output) {
    auto render_start = get_time();

#ifdef USE_TBB
    if (nthreads > 0)
        tbb::task_scheduler_init init(nthreads);
    
    tbb::parallel_for (tbb::blocked_range<size_t>(0, scene.film->tiles.size()),
        [&](const tbb::blocked_range<size_t>& r)
    {
        for (int t = r.begin(); t != r.end(); ++t) {
#else
    {
        for (int t = 0; t < scene.film->tiles.size(); ++t) {
#endif

            auto tile_start = get_time();
            Tile& tile = scene.film->tiles[t];

            Sampler sampler;
            sampler.seed(tile.origin_x, tile.origin_y);
            
            auto integrator_ptr = scene.integrator_fac.create(scene.camera.get(), scene.film.get(), &sampler, &scene.recorder);
            integrator_ptr->setup(&scene);

            RecordContext rctx;

            for (int j = 0; j < tile.height; j++) {
                for (int i = 0; i < tile.width; i++) {
                    RGBSpectrum pixel_radiance{0};

                    for (int s = 0; s < sample_count; s++) {
                        uint x = tile.origin_x + i;
                        uint y = tile.origin_y + j;

                        rctx.pixel_x = x;
                        rctx.pixel_y = y;

                        //auto ray = scene.camera->generate_ray(x, y);
                        auto ray = scene.camera->generate_ray(Vec2f(x, y) + sampler.random2f());
                        pixel_radiance += integrator_ptr->Li(ray, &rctx);
                    }

                    pixel_radiance /= sample_count;
                    tile.set_pixel_color(i, j, pixel_radiance);
                }
            }

            auto tile_end = get_time();
            auto tile_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tile_end - tile_start);
            std::cout << "tile duration : " << tile_duration.count() << " ms\n";
        }

#ifdef USE_TBB
    });
#else
    }
#endif

    auto render_end = get_time();
    auto render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(render_end - render_start);
    std::cout << "render duration : " << render_duration.count() << " ms\n";

    // TODO : do some validation first
    scene.film->filename = output;
    scene.film->write_tiles();
    scene.recorder.output(std::cout);

    return true;
}