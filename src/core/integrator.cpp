
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

    auto ret = isect.normal.abs();
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

PathIntegrator::PathIntegrator(Camera* cam_ptr, Film* flm_ptr, Recorder* rec)
    : Integrator(cam_ptr, flm_ptr, rec)
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
     * Here with MIS and NEE, the equation is
     * rewrite as:
     * 
     * Lo = Le + mis_weight_1 * L_direct       (2)
     *      + mis_weight_2 * L_indirect
     * 
     * In the context of path tracing, the equation
     * is:
     * 
     * Lo = Le1 + w_1_d * L_1_d + w_1_i * f1 * (3)
     *      (Le2 + w_2_d * L_2_d + f2 *
     *      (Le3 + ...))
     * 
     * In each iteration, we follow the equation 2
     * by seperating the procedural into 3 parts:
     * 
     * 1. Le calculation if we hit emissive object
     *    (delta light cannot be hit but can be
     *    sampled);
     * 2. L_direct calculation by sampling a light
     *    (or all) and get its contribution;
     * 3. L_indirect calculation by sampling material,
     *    update throughtput and forward to next
     *    iteration.
     * 
     * And:
     * 
     * 1. L_direct and L_indirect could be clipped off
     *    in deeper iterations by Russian Roulette;
     * 2. throughtput is used to represent the accumulated
     *    effect of each hit event a.k.a "f" in the equations
     *    with initial value 1.
     * 
     * *********************************************/

    RGBSpectrum Li{0}, throughput{1};
    float direct_weight;
    float indirect_weight = 1.f;
    float eta = 1.f;
    float emission_pdf, light_pdf, bsdf_pdf;

    Intersection isect;
    Ray ray(r);

    constexpr int max_depth = 6;
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

    for (int depth = 0; depth < max_depth; ++depth) {
        OSL::ShaderGlobals sg;

        if (accel_ptr->intersect(ray, isect)) {
            KazenRenderServices::globals_from_hit(sg, ray, isect);
            auto shader_ptr = (*shaders)[isect.shader_name];
            if (shader_ptr == nullptr)
                throw std::runtime_error(fmt::format("Shader for name : {} does not exist..", isect.shader_name));
            shadingsys->execute(*ctx, *shader_ptr, sg);
            ShadingResult ret;
            bool last_bounce = depth == max_depth;
            process_closure(ret, sg.Ci, RGBSpectrum{1}, last_bounce);

            /* *********************************************
             * 1. Le calculation
             * *********************************************/
            if (isect.is_light)
                Li += throughput * indirect_weight * ret.Le;

            // Russian Roulette
            // The time of doing the Russian Roulette will determine
            // which part will be discarded.
            // Here we follow the implementation in mitsuba and will
            // discard the L_direct & L_indirect on condition.
            // The implementation in pbrt-v3 do it in the final step
            // of the iteration which will discard the indirect contribution
            // of this iteration.
            if (depth >= min_depth) {
                auto prob = std::min(throughput.max_component() * eta * eta, 0.99f);
                if (prob < random()) {
                    p.record(ERouletteCut, sg, throughput, Li);
                    break;
                }
                throughput /= prob;
            }

            /* *********************************************
             * 2. L_direct calculation by sampling light
             * *********************************************/
            ret.surface.compute_pdfs(sg, throughput, depth >= min_depth);

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
                if (!Ls.is_zero()) {
                    float cos_theta_v = dot(light_dir, isect.normal);
                    auto f = ret.surface.eval(sg, light_dir, bsdf_pdf);
                    direct_weight = power_heuristic(1, light_pdf, 1, bsdf_pdf);
                    Li += throughput * Ls * f * cos_theta_v * direct_weight;
                }
            }

            /* *********************************************
             * 3. L_indirect calculation by sampling material
             * *********************************************/
            RGBSpectrum f{0};
            if (isect.is_light)
                f = ret.surface.eval(sg, isect.wi, bsdf_pdf);
            else
                f = ret.surface.sample(sg, random3f(), isect.wi, bsdf_pdf);

            // Add mis weight into throughput?
            throughput *= f;

            if (throughput.is_zero())
                return Li;
            // TODO : add the real eta calculation
            eta *= 0.95f;

            // Construct next ray
            ray.direction = isect.wi;
            ray.origin = isect.position + ray.direction * epsilon<float>;
            ray.tmin = 0;
            ray.tmax = std::numeric_limits<float>::max();
            isect.ray_t = std::numeric_limits<float>::max();
            
            // LightPath event recording
            if (isect.is_light)
                p.record(EEmission, sg, throughput, Li);
            else
                p.record(EReflection, sg, throughput, Li);
        }
        else {
            // Hit background
            p.record(EBackground, sg, throughput, Li);
            break;
        }
    }

    recorder->record(p, rctx);
    
    return Li;
}
