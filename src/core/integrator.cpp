
#include <iostream>
#include <limits>

#include <tbb/tbb.h>

#include "integrator.h"
#include "core/material.h"
#include "core/state.h"
#include "core/sampling.h"
#include "core/scene.h"
#include "shading/bsdf.h"

Integrator::Integrator()
    : accel_ptr(nullptr)
    , camera_ptr(nullptr)
    , film_ptr(nullptr)
{}

void Integrator::setup(Scene* scene) {
    accel_ptr = scene->accelerator.get();
    lights = &scene->lights;
}

Integrator::Integrator(Camera* cam_ptr, Film* flm_ptr, Recorder* rec)
    : camera_ptr(cam_ptr)
    , film_ptr(flm_ptr)
    , recorder(rec)
{}

NormalIntegrator::NormalIntegrator()
    : Integrator()
{}

NormalIntegrator::NormalIntegrator(Camera* cam_ptr, Film* flm_ptr, Recorder* rec)
    : Integrator(cam_ptr, flm_ptr, rec)
{}

RGBSpectrum NormalIntegrator::Li(const Ray& r, const RecordContext& rctx) const {
    Intersection isect;
    if (!accel_ptr->intersect(r, isect)) {
        return RGBSpectrum{0};
    }

    auto ret = isect.N.abs();
    return ret;
}

AmbientOcclusionIntegrator::AmbientOcclusionIntegrator()
    : Integrator()
{}

AmbientOcclusionIntegrator::AmbientOcclusionIntegrator(Camera* cam_ptr,
    Film* flm_ptr, Recorder* rec)
    : Integrator(cam_ptr, flm_ptr, rec)
{}

RGBSpectrum AmbientOcclusionIntegrator::Li(const Ray& r, const RecordContext& rctx) const {
    Intersection isect;
    if (!accel_ptr->intersect(r, isect)) {
        return RGBSpectrum{0};
    }

    auto sample = sample_hemisphere().normalized();
    auto shadow_ray_dir = tangent_to_world(sample, isect.N, isect.tangent, isect.bitangent);
    auto shadow_ray = Ray(isect.P, shadow_ray_dir.normalized());

    float t;
    if (!accel_ptr->intersect(shadow_ray, t)) {
        float cos_theta_v = cos_theta(sample);
        return RGBSpectrum{static_cast<float>(cos_theta_v * M_1_PI)} / (M_1_PI * 0.25f);
    }

    return RGBSpectrum{0};
}

WhittedIntegrator::WhittedIntegrator()
    : OSLBasedIntegrator()
{}

WhittedIntegrator::WhittedIntegrator(Camera* cam_ptr, Film* flm_ptr, Recorder* rec)
    : OSLBasedIntegrator(cam_ptr, flm_ptr, rec)
{}

void WhittedIntegrator::setup(Scene* scene) {
    Integrator::setup(scene);
    shadingsys = scene->shadingsys.get();
    shaders = &scene->shaders;
    thread_info = shadingsys->create_thread_info();
    ctx = shadingsys->get_context(thread_info);
}

RGBSpectrum WhittedIntegrator::Li(const Ray& r, const RecordContext& rctx) const {
    RGBSpectrum Li{0};
    Intersection isect;
    Ray ray(r);
    OSL::ShaderGlobals sg;

    LightPath p;
    LightPathEvent e_start;
    e_start.type = EStart;
    e_start.event_position = r.origin;
    e_start.ray_direction = r.direction;
    e_start.Li = Li;
    p.record(std::move(e_start));

    if (!accel_ptr->intersect(ray, isect)) {
        p.record(EBackground, isect, RGBSpectrum{0}, Li);
        return Li;
    }

    isect.refined_point = isect.P;

    KazenRenderServices::globals_from_hit(sg, ray, isect);
    auto shader_ptr = (*shaders)[isect.shader_name];
    if (shader_ptr == nullptr)
        throw std::runtime_error(fmt::format("Shader for name : {} does not exist..", isect.shader_name));
    shadingsys->execute(*ctx, *shader_ptr, sg);
    ShadingResult ret;
    process_closure(ret, sg.Ci, RGBSpectrum{1}, false);
    ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);

    if (isect.is_light)
        Li += ret.Le;

    // We don't have specular material for now
    int light_sample_range = lights->size();

    if (isect.is_light)
        --light_sample_range;

    if (light_sample_range > 0) {
        int sampled_light_idx = randomf() * 0.99999 * light_sample_range;
        if (isect.is_light && sampled_light_idx >= isect.light_id)
            ++sampled_light_idx;

        auto light_ptr = lights->at(sampled_light_idx).get();
        Vec3f light_dir;
        float light_pdf, bsdf_pdf;
        auto Ls = light_ptr->sample(isect, light_dir, light_pdf, accel_ptr);

        if (!Ls.is_zero()) {
            float cos_theta_v = dot(light_dir, isect.N);
            auto f = ret.surface.eval(sg, light_dir, bsdf_pdf);
            recorder->print(rctx, fmt::format("cos theta : {}, f : {}, Ls : {}, light_pdf : {}", cos_theta_v, f, Ls, light_pdf));
            Li += (f * Ls * cos_theta_v) * lights->size();
        }

        p.record(EReflection, isect, RGBSpectrum{0}, Li);
    }

    recorder->record(p, rctx);

    return Li;
}

PathIntegrator::PathIntegrator()
    : OSLBasedIntegrator()
{}

PathIntegrator::PathIntegrator(Camera* cam_ptr, Film* flm_ptr, Recorder* rec)
    : OSLBasedIntegrator(cam_ptr, flm_ptr, rec)
{}

void PathIntegrator::setup(Scene* scene) {
    Integrator::setup(scene);
    shadingsys = scene->shadingsys.get();
    shaders = &scene->shaders;
    thread_info = shadingsys->create_thread_info();
    ctx = shadingsys->get_context(thread_info);
}

RGBSpectrum PathIntegrator::Li(const Ray& r, const RecordContext& rctx) const {
    /* *********************************************
     * The rendering equation is:
     *           /
     * Lo = Le + | f * Li * cos dw             (1)
     *           /hemisphere
     * 
     * For one shading point, the radiance is sampled
     * with MIS for one direction, hence generating
     * the ray path. There were methods generating ray
     * trees which is just not more efficient than the
     * current way.
     * 
     * *********************************************/

    RGBSpectrum Li{0}, throughput{1};
    float direct_weight;
    float indirect_weight = 1.f;
    float eta = 0.95f;
    float emission_pdf, light_pdf, bsdf_pdf;

    Intersection isect;
    Ray ray(r);

    constexpr int max_depth = 8;
    constexpr int min_depth = 3;

    // Create light path and add start event
    LightPath p;
    LightPathEvent e_start;
    e_start.type = EStart;
    e_start.event_position = r.origin;
    e_start.ray_direction = r.direction;
    e_start.throughput = throughput;
    e_start.Li = Li;
    p.record(std::move(e_start));
    size_t last_geom_id = -1;
    bool specular_bounce = false;

    for (int depth = 0; depth < max_depth; ++depth) {
        OSL::ShaderGlobals sg;

        if (accel_ptr->intersect(ray, isect)) {
            if (isect.geom_id == last_geom_id) {
                // Bypass self-intersection
                ray.tmin = isect.offset_point2();
                ray.tmax = std::numeric_limits<float>::max();
                isect.ray_t = std::numeric_limits<float>::max();
                continue;
            }
            last_geom_id = isect.geom_id;

            KazenRenderServices::globals_from_hit(sg, ray, isect);
            auto shader_ptr = (*shaders)[isect.shader_name];
            if (shader_ptr == nullptr)
                throw std::runtime_error(fmt::format("Shader for name : {} does not exist..", isect.shader_name));
            shadingsys->execute(*ctx, *shader_ptr, sg);
            ShadingResult ret;
            bool last_bounce = depth == max_depth;
            process_closure(ret, sg.Ci, RGBSpectrum{1}, false);

            /* *********************************************
             * 1. Le calculation
             * We try out pbrt's impl, add emitted contribution
             * at only first hit and specular bounce coz further
             * emission is counted in the direction light sampling
             * part
             * *********************************************/
            //if (isect.is_light)
            if (depth == 0 || specular_bounce)
                Li += throughput * ret.Le;


            ret.surface.compute_pdfs(sg, throughput, depth >= min_depth);

            int light_sample_range = lights->size();

            // Avoid sampling itself if we've hit a geometry light
            if (isect.is_light)
                --light_sample_range;

            Vec3f next_ray_dir;
            float bsdf_sampled_pdf;
            auto sampled_f = ret.surface.sample(sg, random3f(), next_ray_dir, bsdf_sampled_pdf);

            if (light_sample_range > 0) {
                int sampled_light_idx = randomf() * 0.99999 * light_sample_range;

                // Shift the index if we happen to sampled itself
                if (isect.is_light && sampled_light_idx >= isect.light_id)
                    ++sampled_light_idx;

                /* *********************************************
                * 2. Sampling light to get direct light contribution
                * *********************************************/
                auto light_ptr = lights->at(sampled_light_idx).get();
                Vec3f light_dir;
                auto Ls = light_ptr->sample(isect, light_dir, light_pdf, accel_ptr);
                if (!Ls.is_zero()) {
                    float cos_theta_v = dot(light_dir, isect.N);
                    auto f = ret.surface.eval(sg, light_dir, bsdf_pdf);
                    direct_weight = power_heuristic(1, light_pdf, 1, bsdf_pdf);
                    Li += throughput * Ls * f * cos_theta_v * direct_weight * lights->size();
                }

                recorder->print(rctx, fmt::format("Ls sampling light : {}", Ls));

                /* *********************************************
                * 3. Sampling material to get next direction
                * *********************************************/
                float cos_theta_v = dot(next_ray_dir, isect.N);
                Intersection tmpsect;
                if (accel_ptr->intersect(Ray(isect.P, next_ray_dir), tmpsect) && tmpsect.is_light) {
                    Ls = light_ptr->eval(isect, next_ray_dir, tmpsect.P, light_pdf, tmpsect.N);
                    indirect_weight = power_heuristic(1, bsdf_sampled_pdf, 1, light_pdf);
                    Li += throughput * Ls * sampled_f * cos_theta_v * indirect_weight / bsdf_sampled_pdf;
                }

                recorder->print(rctx, fmt::format("Ls sampling bsdf : {}", Ls));
            }

            throughput *= sampled_f;
            if (throughput.is_zero())
                return Li;
            // TODO : add the real eta calculation
            //eta *= 0.95f;

            // Russian Roulette
            // The time of doing the Russian Roulette will determine
            // which part will be discarded.
            // Here we follow the implementation in mitsuba and will
            // discard the L_direct & L_indirect on condition.
            if (depth >= min_depth) {
                auto prob = std::min(throughput.max_component() * eta * eta, 0.99f);
                if (prob < randomf()) {
                    p.record(ERouletteCut, isect, throughput, Li);
                    break;
                }
                throughput /= prob;
            }

            // Construct next ray
            //ray.direction = isect.wi;
            ray.direction = next_ray_dir;
            //isect.refined_point = isect.P + isect.offset_point1();
            isect.refined_point = isect.P;
            ray.origin = isect.refined_point;
            //ray.tmin = isect.offset_point2();
            ray.tmin = epsilon<float>;
            ray.tmax = std::numeric_limits<float>::max();
            isect.ray_t = std::numeric_limits<float>::max();
            
            // LightPath event recording
            if (isect.is_light)
                p.record(EEmission, isect, throughput, Li);
            else
                p.record(EReflection, isect, throughput, Li);
        }
        else {
            // Hit background
            p.record(EBackground, isect, throughput, Li);
            break;
        }
    }

    recorder->record(p, rctx);
    
    return Li;
}

OldPathIntegrator::OldPathIntegrator()
    : OSLBasedIntegrator()
{}

OldPathIntegrator::OldPathIntegrator(Camera* cam_ptr, Film* flm_ptr, Recorder* rec)
    : OSLBasedIntegrator(cam_ptr, flm_ptr, rec)
{}

void OldPathIntegrator::setup(Scene* scene) {
    Integrator::setup(scene);
    shadingsys = scene->shadingsys.get();
    shaders = &scene->shaders;
    thread_info = shadingsys->create_thread_info();
    ctx = shadingsys->get_context(thread_info);
}

RGBSpectrum OldPathIntegrator::Li (const Ray& r, const RecordContext& rctx) const {
    Ray ray(r);
    Intersection isect;

    RGBSpectrum Li{0}, throughput{1};
    float eta = 1.f;
    float direct_weight;
    float indirect_weight = 1.f;
    float light_pdf, bsdf_pdf;

    constexpr int min_depth = 3;
    constexpr int max_depth = 6;
    RGBSpectrum f{1};

    for (int depth = 0; depth <= max_depth; ++depth) {
        OSL::ShaderGlobals sg;

        if (accel_ptr->intersect(ray, isect)) {
            KazenRenderServices::globals_from_hit(sg, ray, isect);
            auto shader_ptr = (*shaders)[isect.shader_name];
            if (shader_ptr == nullptr)
                throw std::runtime_error(fmt::format("Shader for name : {} does not exist..", isect.shader_name));
            shadingsys->execute(*ctx, *shader_ptr, sg);
            ShadingResult ret;
            //bool last_bounce = depth == max_depth;
            process_closure(ret, sg.Ci, RGBSpectrum{1}, false);

            if (isect.is_light && depth > 0) {
                //light_pdf = 1.f / isect.shape->area();
                light_pdf = 1.f;
                indirect_weight = power_heuristic(1, bsdf_pdf, 1, light_pdf);
                auto cos_theta_v = dot(-ray.direction, isect.N);
                Li += throughput * ret.Le * f * cos_theta_v * indirect_weight / bsdf_pdf;
            }

            throughput *= f;
            if (throughput.is_zero())
                return Li;
            eta *= 0.95f;

            // Russian roulette
            if (depth >= min_depth) {
                auto prob = std::min(throughput.max_component() * eta * eta, 0.95f);
                if (prob <= randomf())
                    break;
                throughput /= prob;
            }

            // Intersection with lights
            if (isect.is_light)
                Li += throughput * ret.Le;

            // Sampling light to get direct distribution
            ret.surface.compute_pdfs(sg, throughput, false);
            int light_sample_range = lights->size();

            // Avoid sampling itself if we've hit a geometry light
            if (isect.is_light)
                --light_sample_range;

            if (light_sample_range > 0) {
                int sampled_light_idx = randomf() * 0.99999 * light_sample_range;

                // Shift the index if we happen to sampled itself
                if (isect.is_light && sampled_light_idx >= isect.light_id)
                    ++sampled_light_idx;

                auto light_ptr = lights->at(sampled_light_idx).get();
                Vec3f light_dir;
                auto Ls = light_ptr->sample(isect, light_dir, light_pdf, accel_ptr);
                if (isect.is_light)
                    isect.wi = light_dir;
                if (!Ls.is_zero()) {
                    float cos_theta_v = dot(light_dir, isect.N);
                    auto f = ret.surface.eval(sg, light_dir, bsdf_pdf);
                    direct_weight = power_heuristic(1, light_pdf, 1, bsdf_pdf);
                    Li += throughput * Ls * f * cos_theta_v * direct_weight / light_pdf;
                }
            }

            // Sampling bsdf to get next direction
            f = ret.surface.sample(sg, random3f(), isect.wi, bsdf_pdf);

            // Construct next ray
            ray.direction = isect.wi;
            isect.refined_point = isect.P;
            ray.origin = isect.refined_point;
            ray.tmin = epsilon<float>;
            ray.tmax = std::numeric_limits<float>::max();
            isect.ray_t = std::numeric_limits<float>::max();
        }
        else {
            break;
        }
    }

    return Li;
}