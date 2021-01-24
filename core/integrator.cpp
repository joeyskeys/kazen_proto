
#include <iostream>
#include <limits>

#include "integrator.h"
#include "material.h"

void Integrator::render() {
    auto film_width = film_ptr->width;
    auto film_height = film_ptr->height;
    float depth = 10;

    constexpr static int sample_count = 5;

    for (auto& tile : film_ptr->tiles) {
        for (int j = 0; j < tile.height; j++) {
            for (int i = 0; i < tile.width; i++) {
                Intersection isect;
                RGBSpectrum l{0.f, 0.f, 0.f};
                RGBSpectrum spec{0.f, 0.f, 0.f};

                for (int s = 0; s < sample_count; s++) {

                    uint x = tile.origin_x + i;
                    uint y = tile.origin_y + j;

                    auto ray = camera_ptr->generate_ray(tile.origin_x + i, tile.origin_y + j);
                    RGBSpectrum beta{1.f, 1.f, 1.f};
                    bool hit = false;
                    Ray tmp_ray = ray;

                    for (int k = 0; k < depth; k++) {
                        isect.ray_t = std::numeric_limits<float>::max();
                        if (accel_ptr->intersect(ray, isect) && k < depth - 1) {
                            auto mat_ptr = isect.mat;
                            auto wo = -ray.direction;
                            float p;
                            beta *= mat_ptr->calculate_response(isect, ray);

                            ray.origin = isect.position;
                            ray.direction = isect.wi;
                            ray.tmin = 0;
                            ray.tmax = std::numeric_limits<float>::max();

                            /*
                            if (isect.obj_id == 1 && k > 0) {
                                std::cout << "k : " << k << ", beta value on bottom : " << beta;
                                std::cout << "position  : " << isect.position;
                                std::cout << "ray dir : " << ray.direction;
                            }
                            */

                            hit = true;
                        }
                        else {                
                            auto front_vec = Vec3f{0.f, 0.f, -1.f};
                            auto t = 0.5f * (ray.direction.y() + 1.f);
                            beta *= (1.f - t) * RGBSpectrum{1.f, 1.f, 1.f} + t * RGBSpectrum{0.5f, 0.7f, 1.f};
                            spec += beta;
                            //if (hit)
                                //std::cout << "s " << s << ", depth : " << k << ", beta : " << beta;
                            break;
                        }
                    }

                    //std::cout << "s " << s << " beta : " << beta;
                }

                spec /= sample_count;

                tile.set_pixel_color(i, j, spec);
            }
        }
    }
}