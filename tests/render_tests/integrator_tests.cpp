
#include "core/scene.h"

int main() {
    Scene scene;
    scene.parse_from_file("../resource/scene/cbox/cbox_mis.xml");

    auto integrator_ptr = scene.integrator_fac.create(
        scene.camera.get(), scene.film.get(), &scene.recorder);
    integrator_ptr->setup(&scene);
    scene.recorder.x_min = 159;
    scene.recorder.x_max = 161;
    scene.recorder.y_min = 99;
    scene.recorder.y_max = 101;
    scene.recorder.setup();

    RecordContext rctx;
    rctx.pixel_x = 160;
    rctx.pixel_y = 100;

    auto ray = scene.camera->generate_ray(250, 450);
    auto radiance = integrator_ptr->Li(ray, rctx);

    scene.recorder.output(std::cout);
}