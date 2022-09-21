
#include "core/sampler.h"
#include "core/scene.h"

int main() {
    Scene scene;
    scene.parse_from_file("/home/joeyskeys/assets/kazen_proj/materials/materials.xml");
    Sampler sampler;

    auto integrator_ptr = scene.integrator_fac.create(
        scene.camera.get(), scene.film.get(), &sampler, &scene.recorder);
    integrator_ptr->setup(&scene);
    scene.recorder.x_min = 322;
    scene.recorder.x_max = 323;
    scene.recorder.y_min = 383;
    scene.recorder.y_max = 384;
    scene.recorder.setup();

    RecordContext rctx;
    rctx.pixel_x = 322;
    rctx.pixel_y = 383;

    auto ray = scene.camera->generate_ray(Vec2f{250.5, 400.5});
    auto radiance = integrator_ptr->Li(ray, rctx);

    scene.recorder.output(std::cout);
}