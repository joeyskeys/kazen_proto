
#include <iostream>
#include <limits>

#include "integrator.h"
#include "material.h"

void Integrator::render() {
    auto film_width = film_ptr->width;
    auto film_height = film_ptr->height;
    float depth = 50;

    constexpr static int sample_count = 50;

    for (auto& tile : film_ptr->tiles) {
        for (int j = 0; j < tile.height; j++) {
            for (int i = 0; i < tile.width; i++) {
                auto ray = camera_ptr->generate_ray(tile.origin_x + i, tile.origin_y + j);
                Intersection isect;
                RGBSpectrum l{0.f, 0.f, 0.f};
                RGBSpectrum spec{0.f, 0.f, 0.f};

                for (int s = 0; s < sample_count; s++) {

                    RGBSpectrum beta{1.f, 1.f, 1.f};
                    bool hit = false;
                    for (int k = 0; k < depth; k++) {
                        //if (sphere->intersect(ray, isect)) {
                        if (k == depth - 1)
                            beta = RGBSpectrum{0.f, 0.f, 0.f};

                        isect.ray_t = std::numeric_limits<float>::max();
                        if (accel_ptr->intersect(ray, isect)) {
                            auto mat_ptr = isect.mat;
                            auto wo = -ray.direction;
                            float p;
                            beta *= mat_ptr->calculate_response(isect, ray);

                            ray.origin = isect.position;
                            ray.direction = isect.wi;

                            hit = true;
                        }
                        else {                
                            auto front_vec = Vec3f{0.f, 0.f, -1.f};
                            auto t = 0.5f * (ray.direction.y() + 1.f);
                            beta *= (1.f - t) * RGBSpectrum{1.f, 1.f, 1.f} + t * RGBSpectrum{0.5f, 0.7f, 1.f};
                            break;
                        }
                    }

                    if (hit)
                        std::cout << "beta : " << beta;
                    spec += beta;
                    if (hit)
                        std::cout << "spec : " << spec;
                }

                spec /= sample_count;

                tile.set_pixel_color(i, j, spec);
            }
        }
    }
}