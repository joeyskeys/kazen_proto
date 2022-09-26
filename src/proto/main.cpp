
#include <OpenImageIO/argparse.h>

#include "core/sampler.h"
#include "core/scene.h"
#include "core/state.h"
#include "config.h"

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

int main(int argc, const char **argv) {
    std::string filename;
    OIIO::ArgParse ap;
    int nthreads = 0;
    int sample_count = 10;
    bool debug = false;
    int debug_x = -1;
    int debug_y = -1;
    std::string output = "./test.jpg";

    ap.intro("Kazen Render")
        .usage("kazen [options] filename")
        .print_defaults(true);

    ap.arg("filename")
        .hidden()
        .action([&](OIIO::cspan<const char*> argv) { filename = argv[0]; });

    ap.separator("Options:");
    ap.arg("-t %d", &nthreads)
        .help("number of threads")
        .defaultval(0);

    ap.arg("-o %s", &output)
        .help("output filename");

    ap.arg("-s %d", &sample_count)
        .help("number of samples per pixel")
        .defaultval(10);

    ap.arg("-d", &debug)
        .help("enable debug mode");

    ap.arg("--pixel %d %d", &debug_x, &debug_y)
        .help("specify the pixel of interest");

    if (ap.parse(argc, argv) < 0 || filename.size() == 0) {
        std::cerr << ap.geterror() << std::endl;
        ap.print_help();
        return 0;
    }

    Scene scene;
    scene.parse_from_file(filename);


    auto render_start = get_time();

    if (debug) {
        if (debug_x < 0 || debug_x >= scene.film->width ||
            debug_y < 0 || debug_y >= scene.film->height)
        {
            std::cout << "Pixel of interest (--pixel) must be set and the value should be valid\n"
                << "Scene film wdith : " << scene.film->width << ", height : "
                << scene.film->height << std::endl;
        }

        Sampler sampler;
        sampler.seed(debug_x, debug_y);

        scene.recorder.x_min = debug_x;
        scene.recorder.x_max = debug_x + 1;
        scene.recorder.y_min = debug_y;
        scene.recorder.y_max = debug_y + 1;
        scene.recorder.setup();

        RecordContext rctx;
        rctx.pixel_x = debug_x;
        rctx.pixel_y = debug_y;

        auto integrator_ptr = scene.integrator_fac.create(scene.camera.get(), scene.film.get(), &sampler, &scene.recorder);
        integrator_ptr->setup(&scene);

        for (int i = 0; i < sample_count; ++i) {
            auto ray = scene.camera->generate_ray(Vec2f(debug_x, debug_y) + sampler.random2f());
            auto radiance = integrator_ptr->Li(ray, &rctx);

            std::cout << "radiance value : " << radiance << std::endl;
        }
        
        scene.recorder.output(std::cout);

        return 0;
    }

#ifdef USE_TBB
    if (nthreads > 0)
        tbb::task_scheduler_init init(nthreads);
    tbb::parallel_for (tbb::blocked_range<size_t>(0, scene.film->tiles.size()),
        [&](const tbb::blocked_range<size_t>& r) {
#else
    {
#endif

#ifdef USE_TBB
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

    return 0;
}