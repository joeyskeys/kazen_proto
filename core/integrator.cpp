
#include <iostream>
#include <limits>

#include <tbb/tbb.h>

#include "integrator.h"
#include "material.h"
#include "state.h"
#include "sampling.h"

static RGBSpectrum estamate_direct(const Intersection& isect, const Light* light_ptr, const Integrator& integrator) {
    Vec3f wi;
    float light_pdf, scattering_pdf;
    Intersection isect_tmp;
    
    // Do visibility test in light sampling
    RGBSpectrum light_radiance = light_ptr->sample_l(isect, wi, light_pdf, integrator.accel_ptr);
    RGBSpectrum result_radiance;

    if (light_pdf > 0.f && !light_radiance.is_zero()) {
        RGBSpectrum f = isect.mat->bxdf->f(isect.wo, wi, isect_tmp);
        scattering_pdf = isect.mat->bxdf->pdf(isect.wo, wi);

        if (!f.is_zero()) {
            if (light_ptr->is_delta)
                result_radiance += f * light_radiance / light_pdf;
            else {
                auto weight = power_heuristic(1, light_pdf, 1, scattering_pdf);
                result_radiance += f * light_radiance * weight / light_pdf;
            }
        }
    }

    // TODO: apply MIS after area light is added
}

static RGBSpectrum estamate_one_light(const Intersection& isect, const Integrator& integrator) {
    auto light_cnt = integrator.lights.size();
    auto light_ptr = integrator.lights[randomi(light_cnt - 1)];
    return estamate_direct(isect, light_ptr, integrator);
}

static RGBSpectrum estamate_all_light(const Intersection& isect, const Integrator& integrator) {
    auto light_cnt = integrator.lights.size();
    RGBSpectrum ret{0.f, 0.f, 0.f};

    if (light_cnt == 0)
        return ret;

    for (auto& light_ptr : integrator.lights)
        ret += estamate_direct(isect, light_ptr, integrator);
    return ret / light_cnt;
}

void Integrator::render() {
    auto film_width = film_ptr->width;
    auto film_height = film_ptr->height;
    float depth = 10;

    constexpr static int sample_count = 5;

    auto render_start = get_time();

    //tbb::task_scheduler_init init(1);

    tbb::parallel_for (tbb::blocked_range<size_t>(0, film_ptr->tiles.size()), [&](const tbb::blocked_range<size_t>& r) {
        auto tile_start = get_time();

        for (int t = r.begin(); t != r.end(); ++t) {
            Tile& tile = film_ptr->tiles[t];

            for (int j = 0; j < tile.height; j++) {
                for (int i = 0; i < tile.width; i++) {
                    Intersection isect;
                    RGBSpectrum l{0.f, 0.f, 0.f};
                    RGBSpectrum radiance_total{0.f, 0.f, 0.f};

                    for (int s = 0; s < sample_count; s++) {

                        uint x = tile.origin_x + i;
                        uint y = tile.origin_y + j;

                        auto ray = camera_ptr->generate_ray(tile.origin_x + i, tile.origin_y + j);
                        RGBSpectrum beta{1.f, 1.f, 1.f};
                        bool hit = false;
                        Ray tmp_ray = ray;

                        auto sample_start = get_time();

                        RGBSpectrum radiance_per_sample{0.f, 0.f, 0.f};

                        for (int k = 0; k < depth; k++) {
                            isect.ray_t = std::numeric_limits<float>::max();
                            if (accel_ptr->intersect(ray, isect) && k < depth - 1) {
                                // Add radiance contribution from this shading point
                                radiance_per_sample += beta * estamate_all_light(isect, *this);
                                //radiance_per_sample += beta * estamate_one_light();

                                // Sample material to construct next ray
                                auto mat_ptr = isect.mat;
                                auto wo = -ray.direction;
                                float p;
                                beta *= mat_ptr->calculate_response(isect, ray);

                                ray.origin = isect.position;
                                ray.direction = isect.wi;
                                ray.tmin = 0;
                                ray.tmax = std::numeric_limits<float>::max();

                                hit = true;
                            }
                            else {                
                                auto front_vec = Vec3f{0.f, 0.f, -1.f};
                                auto t = 0.5f * (ray.direction.y() + 1.f);
                                radiance_per_sample += beta * ((1.f - t) * RGBSpectrum{1.f, 1.f, 1.f} + t * RGBSpectrum{0.5f, 0.7f, 1.f});
                                break;
                            }
                        }

                        radiance_total += radiance_per_sample;
                    }

                    radiance_total /= sample_count;

                    tile.set_pixel_color(i, j, radiance_total);
                }
            }
        }

        auto tile_end = get_time();
        auto tile_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tile_end - tile_start);
        std::cout << "sample duration : " << tile_duration.count() << " ms\n";
    });

    auto render_end = get_time();
    auto render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(render_end - render_start);
    std::cout << "render duration : " << render_duration.count() << " ms\n";
}