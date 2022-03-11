
#include "core/scene.h"

int main() {
    Scene scene;
    scene.parse_from_file("../resource/scene/textured_cornell_box/cornell_box.xml");

    auto integrator_ptr = scene.integrator_fac.create(
        scene.camera.get(), scene.film.get(), &scene.recorder);
    integrator_ptr->setup(&scene);
    scene.recorder.x_min = 581;
    scene.recorder.x_max = 582;
    scene.recorder.y_min = 300;
    scene.recorder.y_max = 301;
    scene.recorder.setup();

    RecordContext rctx;
    rctx.pixel_x = 581;
    rctx.pixel_y = 300;

    auto ray = scene.camera->generate_ray(581, 300);
    auto radiance = integrator_ptr->Li(ray, rctx);

    scene.recorder.output(std::cout);
}