
#include <iostream>
#include <limits>

#include <tbb/tbb.h>

#include "integrator.h"
#include "material.h"
#include "state.h"
#include "sampling.h"
#include "shading/bsdf.h"

Integrator::Integrator()
    : accel_ptr(nullptr)
    , camera_ptr(nullptr)
    , film_ptr(nullptr)
{}

Integrator::Integrator(Camera* cam_ptr, Film* flm_ptr)
    : camera_ptr(cam_ptr)
    , film_ptr(flm_ptr)
{}

NormalIntegrator::NormalIntegrator()
    : Integrator()
{}

NormalIntegrator::NormalIntegrator(Camera* cam_ptr, Film* flm_ptr)
    : Integrator(cam_ptr, flm_ptr)
{}

RGBSpectrum NormalIntegrator::Li(const Ray& r) const {
    Intersection isect;
    //isect.ray_t = std::numeric_limits<float>::max();
    if (!accel_ptr->intersect(r, isect)) {
        return RGBSpectrum{0};
    }

    auto ret = isect.normal.abs();
    return ret;
}

AmbientOcclusionIntegrator::AmbientOcclusionIntegrator()
    : Integrator()
{}

AmbientOcclusionIntegrator::AmbientOcclusionIntegrator(Camera* cam_ptr,
    Film* flm_ptr)
    : Integrator(cam_ptr, flm_ptr)
{}

RGBSpectrum AmbientOcclusionIntegrator::Li(const Ray& r) const {
    Intersection isect;
    if (!accel_ptr->intersect(r, isect)) {
        return RGBSpectrum{0};
    }

    auto sample = sample_hemisphere().normalized();
    auto shadow_ray_dir = tangent_to_world(sample, isect.normal, isect.tangent, isect.bitangent);
    auto shadow_ray = Ray(isect.position, shadow_ray_dir.normalized());

    float t;
    if (!accel_ptr->intersect(shadow_ray, t)) {
        float cos_theta_v = cos_theta(sample);
        return RGBSpectrum{static_cast<float>(cos_theta_v * M_1_PI)} / (M_1_PI * 0.25f);
    }

    return RGBSpectrum{0};
}

PathIntegrator::PathIntegrator()
    : Integrator()
{}

PathIntegrator::PathIntegrator(Camera* cam_ptr, Film* flm_ptr)
    : Integrator(cam_ptr, flm_ptr)
{}

RGBSpectrum PathIntegrator::Li(const Ray& r) const {
    RGBSpectrum Li{0}, throughput{0};
    float eta = 1.f;
    float bsdf_weight = 1.f;

    Intersection isect;
    Ray ray(r);
    if (!accel_ptr->intersect(ray, isect))
        return Li;

    int depth = 1;
    constexpr int max_depth = 6;
    //while (true) {
    while (depth <= max_depth) {
        OSL::ShaderGlobals sg;
        KazenRenderServices::globals_from_hit(sg, ray, isect);
        auto shader_ptr = (*shaders)[isect.shader_name];
        if (shader_ptr == nullptr)
            throw std::runtime_error(fmt::format("Shader for name : {} does not exist..", isect.shader_name));
        shadingsys->execute(*ctx, *shader_ptr, sg);
        ShadingResult ret;
        bool last_bounce = depth == max_depth;
        process_closure(ret, sg.Ci, RGBSpectrum{1}, last_bounce);

        // Check if hit light
        if (isect.is_light) {
            Li += bsdf_weight * throughput * isect.shape->light->eval(isect, (ray.origin - isect.position).normalized(), isect.normal);
        }

        // Russian roulette
        if (depth >= 3) {
            auto prob = std::min(throughput.max_component() * eta * eta, 0.99f);
            if (prob < random())
                break;
            throughput /= prob;
        }

        // Light sampling
        int sampled_light_idx = randomf() * 0.99999 * lights->size();
        auto light_ptr = lights->at(sampled_light_idx).get();
        Vec3f light_dir;
        float light_pdf;
        // We'are evenly sampling the lights
        auto Ls = light_ptr->sample(isect, light_dir, light_pdf, accel_ptr) / lights->size();
        light_pdf = light_ptr->pdf(isect);
        if (!Ls.is_zero()) {
            float cos_theta_v = dot(light_dir, isect.normal);
            float bsdf_pdf;
            auto f = ret.bsdf.eval(sg, light_dir, bsdf_pdf);
            float light_weight = power_heuristic(1, light_pdf, 1, bsdf_pdf);
            Li += throughput * Ls * f * cos_theta_v * light_weight;
        }

        // BSDF sampling
        float bsdf_pdf;
        auto bsdf_albedo = ret.bsdf.sample(sg, random3f(), isect.wi, bsdf_pdf);
        throughput *= bsdf_albedo;
        // This might have problem
        eta *= 0.95f;

        ray.origin = isect.position;
        ray.direction = isect.wi;
        ray.tmin = 0;
        ray.tmax = std::numeric_limits<float>::max();
        isect.ray_t = std::numeric_limits<float>::max();
        if (!accel_ptr->intersect(ray, isect))
            break;

        if (isect.is_light) {
            light_pdf = isect.shape->light->pdf(isect);
            bsdf_weight = power_heuristic(1, bsdf_pdf, 1, light_pdf);
        }

        depth++;
    }
    
    return Li;
}

/*
void Integrator::render() {
    auto film_width = film_ptr->width;
    auto film_height = film_ptr->height;
    int  max_depth = 6;
    int  min_depth = 3;

    // No refract for now
    float eta = 1.f;

    constexpr static int sample_count = 64;

    auto render_start = get_time();

#define WITH_TBB

#ifdef WITH_TBB
    //tbb::task_scheduler_init init(1);
    tbb::parallel_for (tbb::blocked_range<size_t>(0, film_ptr->tiles.size()), [&](const tbb::blocked_range<size_t>& r) {
#else
    {
#endif
        OSL::PerThreadInfo *thread_info = shadingsys->create_thread_info();
        OSL::ShadingContext *ctx = shadingsys->get_context(thread_info);

#ifdef WITH_TBB
        for (int t = r.begin(); t != r.end(); ++t) {
#else
        for (int t = 0; t < film_ptr->tiles.size(); t++) {
#endif
            auto tile_start = get_time();
            Tile& tile = film_ptr->tiles[t];

            for (int j = 0; j < tile.height; j++) {
                for (int i = 0; i < tile.width; i++) {
                    Intersection isect;
                    RGBSpectrum radiance_total{0.f, 0.f, 0.f};

                    for (int s = 0; s < sample_count; s++) {

                        uint x = tile.origin_x + i;
                        uint y = tile.origin_y + j;

                        auto ray = camera_ptr->generate_ray(tile.origin_x + i, tile.origin_y + j);
                        RGBSpectrum throughput{1.f, 1.f, 1.f};
                        bool hit = false;
                        Ray tmp_ray = ray;

                        auto sample_start = get_time();

                        RGBSpectrum radiance_per_sample{0.f, 0.f, 0.f};
                        float bsdf_weight = 1.f;
                        float bsdf_pdf;
                        Vec3f light_p, light_n;
                        float light_pdf;

                        for (int k = 0; k < max_depth; k++) {
                            isect.ray_t = std::numeric_limits<float>::max();
                            if (accel_ptr->intersect(ray, isect) && k < max_depth - 1) {
                                
                                OSL::ShaderGlobals sg;
                                KazenRenderServices::globals_from_hit(sg, ray, isect);

                                // TODO : move this verification into parsing code
                                auto shader_ptr = (*shaders)[isect.shader_name];
                                if (shader_ptr == nullptr)
                                    throw std::runtime_error(fmt::format("Shader for name : {} does not exist..", isect.shader_name));

                                shadingsys->execute(*ctx, *shader_ptr, sg);
                                ShadingResult ret;
                                bool last_bounce = k == max_depth;
                                process_closure(ret, sg.Ci, RGBSpectrum(1, 1, 1), last_bounce);

                                // Sample BSDF, evaluate light
                                float light_weight = 1.f;
                                if (isect.is_light) {
                                    // something like light->eval(isect.position, light_pdf);
                                    // TODO : GeometryLight is a light or a geometry and what
                                    //        should be stored in lights.
                                    light_pdf = isect.shape->light->pdf(isect);
                                    light_weight = power_heuristic(1, light_pdf, 1, bsdf_pdf);
                                    radiance_per_sample += throughput * light_weight * ret.Le;
                                    break;
                                }
                                //radiance_per_sample += throughput * light_weight * ret.Le;

                                if (k >= min_depth) {
                                    // Perform russian roulette to cut off path
                                    auto probability = std::min(throughput.max_component() * eta * eta, 0.8f);
                                    if (probability < randomf()) {
                                        //std::cout << "break at k = " << k << ";" << std::endl;
                                        break;
                                    }
                                    throughput /= probability;
                                }

                                // build internal pdf
                                ret.bsdf.compute_pdfs(sg, throughput, k >= min_depth);

                                // Sample Light, evaluate BSDF
                                // find a light with sampling
                                int sampled_light_idx = randomf() * lights->size();
                                auto light_ptr = lights->at(sampled_light_idx).get();
                                {
                                    // TODO : Use a OSL emission shader to sample light radiance where
                                    //        some extra magic could happen.
                                    Vec3f light_dir;
                                    auto light_radiance = light_ptr->sample(isect, light_dir, light_pdf, accel_ptr);
                                    auto bsdf_albedo = ret.bsdf.eval(sg, light_dir, bsdf_pdf);
                                    radiance_per_sample += throughput * light_radiance * bsdf_albedo * power_heuristic(1, light_pdf, 1, bsdf_pdf);
                                }

                                // Sample BSDF to construct next ray
                                // Code pattern from pbrt does not suits here..
                                throughput *= ret.bsdf.sample(sg, random3f(), isect.wi, bsdf_pdf);
                                ray.origin = isect.position;
                                ray.direction = isect.wi;
                                ray.tmin = 0;
                                ray.tmax = std::numeric_limits<float>::max();

                                hit = true;
                            }
                            else {
                                // Hit environment
                                auto t = 0.5f * (ray.direction.y() + 1.f);
                                radiance_per_sample += 0.3 * throughput * ((1.f - t) * RGBSpectrum{1.f, 1.f, 1.f} + t * RGBSpectrum{0.5f, 0.7f, 1.f});
                                //std::cout << "miss at k = " << k << ";" << std::endl;
                                break;
                            }
                        }

                        radiance_total += radiance_per_sample;
                    }

                    radiance_total /= sample_count;

                    tile.set_pixel_color(i, j, radiance_total);
                }
            }

            auto tile_end = get_time();
            auto tile_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tile_end - tile_start);
            std::cout << "tile duration : " << tile_duration.count() << " ms\n";
        }

        shadingsys->release_context(ctx);
        shadingsys->destroy_thread_info(thread_info);

#ifdef WITH_TBB
    });
#else
    }
#endif

    auto render_end = get_time();
    auto render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(render_end - render_start);
    std::cout << "render duration : " << render_duration.count() << " ms\n";
}
*/