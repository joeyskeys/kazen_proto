
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

void PathIntegrator::setup(Scene* scene) {
    Integrator::setup(scene);
    shadingsys = scene->shadingsys.get();
    shaders = &scene->shaders;
    thread_info = shadingsys->create_thread_info();
    ctx = shadingsys->get_context(thread_info);
}

RGBSpectrum PathIntegrator::Li(const Ray& r) const {
    RGBSpectrum Li{0}, throughput{1};
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
            Li += bsdf_weight * throughput * ret.Le;
            break;
        }

        // Russian roulette
        if (depth >= 3) {
            auto prob = std::min(throughput.max_component() * eta * eta, 0.99f);
            if (prob < random())
                break;
            throughput /= prob;
        }

        // Build internal pdfs
        ret.bsdf.compute_pdfs(sg, throughput, depth >= 3);

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
