
#include <tbb/tbb.h>
#include <OpenImageIO/argparse.h>

#include "core/sampler.h"
#include "core/scene.h"
#include "core/state.h"

int main(int argc, const char **argv) {
    std::string filename;
    OIIO::ArgParse ap;
    int nthreads = 0;

    ap.intro("Kazen Render")
        .usage("kazen [options] filename")
        .print_defaults(true);

    ap.arg("filename")
        .hidden()
        .action([&](OIIO::cspan<const char*> argv) { filename = argv[0]; });

    ap.separator("Options:");
    ap.arg("-t", &nthreads)
        .help("number of threads")
        .defaultval(0);

    if (ap.parse(argc, argv) < 0 || filename.size() == 0) {
        std::cerr << ap.geterror() << std::endl;
        ap.print_help();
        return 0;
    }

    Scene scene;
    //scene.parse_from_file("../resource/scene/veach_mi/veach_mats.xml");
    scene.parse_from_file(filename);

    constexpr static int sample_count = 10;
            
    scene.recorder.x_min = 480;
    scene.recorder.x_max = 482;
    scene.recorder.y_min = 300;
    scene.recorder.y_max = 302;
    scene.recorder.setup();

    auto render_start = get_time();
    bool hit = false;

#define WITH_TBB

#ifdef WITH_TBB
    if (nthreads > 0)
        tbb::task_scheduler_init init(nthreads);
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
                        pixel_radiance += integrator_ptr->Li(ray, rctx);
                        if (!base::is_zero(pixel_radiance))
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
    //scene.recorder.output(std::cout);

    return 0;
}