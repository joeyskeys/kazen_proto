
#include <iostream>
#include <limits>

#include <tbb/tbb.h>

#include "integrator.h"
#include "material.h"
#include "state.h"
#include "sampling.h"
#include "shading/bsdf.h"

static RGBSpectrum estamate_direct(Intersection& isect, const Light* light_ptr, const Integrator& integrator, ShadingResult& sr, OSL::ShaderGlobals& sg, RGBSpectrum& throughput) {
    Vec3f wi;
    float light_pdf, bsdf_pdf;
    Intersection isect_tmp;

    // Light sampling
    RGBSpectrum light_radiance = light_ptr->sample(isect, wi, light_pdf, integrator.accel_ptr);
    RGBSpectrum result_radiance;

    if (light_pdf > 0.f && !light_radiance.is_zero()) {
        //RGBSpectrum f = isect.mat->bxdf->f(isect.wo, wi, isect_tmp);
        //scattering_pdf = isect.mat->bxdf->pdf(isect.wo, wi);
        auto f = sr.bsdf.eval(sg, wi, bsdf_pdf);

        if (!f.is_zero()) {
            if (light_ptr->is_delta)
                result_radiance += f * light_radiance / light_pdf;
            else {
                auto weight = power_heuristic(1, light_pdf, 1, bsdf_pdf);
                result_radiance += f * light_radiance * weight / light_pdf;
            }
        }
    }

    // BSDF sampling
    throughput *= sr.bsdf.sample(sg, random3f(), isect.wi, bsdf_pdf);

    // TODO: apply MIS after area light is added
    return result_radiance;
}

static RGBSpectrum estamate_one_light(Intersection& isect, const Integrator& integrator, ShadingResult& sr, OSL::ShaderGlobals& sg, RGBSpectrum& throughput) {
    auto light_cnt = integrator.lights->size();
    // FIXME : add a wrapper to sample
    Light* light_ptr = nullptr;
    if (light_cnt > 1)
        light_ptr = integrator.lights->at(randomi(light_cnt - 1)).get();
    else
        light_ptr = integrator.lights->at(0).get();

    return estamate_direct(isect, light_ptr, integrator, sr, sg, throughput);
}

static RGBSpectrum estamate_all_light(Intersection& isect, const Integrator& integrator, ShadingResult& sr, OSL::ShaderGlobals& sg, RGBSpectrum& throughput) {
    auto light_cnt = integrator.lights->size();
    RGBSpectrum ret{0.f, 0.f, 0.f};

    if (light_cnt == 0)
        return ret;

    for (auto& light : *(integrator.lights))
        ret += estamate_direct(isect, light.get(), integrator, sr, sg, throughput);
    return ret / light_cnt;
}

Integrator::Integrator()
    : accel_ptr(nullptr)
    , camera_ptr(nullptr)
    , film_ptr(nullptr)
{

}

Integrator::Integrator(Camera* cam_ptr, Film* flm_ptr)
    : camera_ptr(cam_ptr)
    , film_ptr(flm_ptr)
{
    //shadingsys = std::make_unique<OSL::ShadingSystem>(&rend, nullptr, &errhandler);
    //register_closures(shadingsys.get());
}

void Integrator::render() {
    auto film_width = film_ptr->width;
    auto film_height = film_ptr->height;
    int  max_depth = 10;
    int  min_depth = 3;

    // No refract for now
    float eta = 1.f;

    constexpr static int sample_count = 5;

    auto render_start = get_time();

//#define WITH_TBB

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
                                shadingsys->execute(*ctx, *(*shaders)[isect.shader_name], sg);
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
                                }
                                radiance_per_sample += throughput * light_weight * ret.Le;

                                if (k >= min_depth) {
                                    // Perform russian roulette to cut off path
                                    auto probability = std::min(throughput.max_component() * eta * eta, 0.8f);
                                    if (probability < randomf())
                                        break;
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
                                radiance_per_sample += throughput * ((1.f - t) * RGBSpectrum{1.f, 1.f, 1.f} + t * RGBSpectrum{0.5f, 0.7f, 1.f});
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