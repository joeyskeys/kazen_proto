
#include <iostream>

#include "integrator.h"

void Integrator::render() {
    auto film_width = film_ptr->width;
    auto film_height = film_ptr->height;

    for (auto& tile : film_ptr->tiles) {
        for (int j = 0; j < tile.height; j++) {
            for (int i = 0; i < tile.width; i++) {
                auto ray = camera_ptr->generate_ray(tile.origin_x + i, tile.origin_y + j);

                Intersection isect;
                if (sphere->intersect(ray, isect)) {
                    tile.set_pixel_color(i, j, RGBSpectrum{1.f, 0.f, 0.f});
                }
                else {                
                    auto front_vec = Vec3f{0.f, 0.f, -1.f};
                    auto t = 0.5f * (ray.direction.y() + 1.f);
                    auto spec = (1.f - t) * RGBSpectrum{1.f, 1.f, 1.f} + t * RGBSpectrum{0.5f, 0.7f, 1.f};
                    tile.set_pixel_color(i, j, spec);
                }
            }
        }
    }
}