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
    , sampler_ptr(nullptr)
{}

void Integrator::setup(Scene* scene) {
    accel_ptr = scene->accelerator.get();
    lights = &scene->lights;
}

Light* Integrator::get_random_light(const float& xi, float& pdf) const {
    const auto cnt = lights->size();
    assert(cnt > 0);
    if (cnt == 0)
        return nullptr;

    pdf = 1. / cnt;
    auto idx = std::min(static_cast<size_t>(xi * cnt), cnt - 1);
    return lights->at(idx).get();
}

Integrator::Integrator(Camera* cam_ptr, Film* flm_ptr, Sampler* spl_ptr, Recorder* rec)
    : camera_ptr(cam_ptr)
    , film_ptr(flm_ptr)
    , sampler_ptr(spl_ptr)
    , recorder(rec)
{}

NormalIntegrator::NormalIntegrator()
    : Integrator()
{}

NormalIntegrator::NormalIntegrator(Camera* cam_ptr, Film* flm_ptr, Sampler* spl_ptr, Recorder* rec)
    : Integrator(cam_ptr, flm_ptr, spl_ptr, rec)
{}

RGBSpectrum NormalIntegrator::Li(const Ray& r, const RecordContext& rctx) const {
    Intersection isect;
    if (!accel_ptr->intersect(r, isect)) {
        return RGBSpectrum{0};
    }

    //auto ret = isect.N.abs();
    auto ret = base::abs(isect.shading_normal);
    return ret;
}

AmbientOcclusionIntegrator::AmbientOcclusionIntegrator()
    : Integrator()
{}

AmbientOcclusionIntegrator::AmbientOcclusionIntegrator(Camera* cam_ptr,
    Film* flm_ptr, Sampler* spl_ptr, Recorder* rec)
    : Integrator(cam_ptr, flm_ptr, spl_ptr, rec)
{}

RGBSpectrum AmbientOcclusionIntegrator::Li(const Ray& r, const RecordContext& rctx) const {
    Intersection isect;
    if (!accel_ptr->intersect(r, isect)) {
        return RGBSpectrum{0};
    }

    auto sample = base::normalize(sample_hemisphere());
    auto shadow_ray_dir = local_to_world(sample, isect.shading_normal);
    auto shadow_ray = Ray(isect.P, base::normalize(shadow_ray_dir));

    float t;
    if (!accel_ptr->intersect(shadow_ray, t)) {
        float cos_theta_v = cos_theta(sample);
        return RGBSpectrum{static_cast<float>(cos_theta_v * M_1_PI)} / (M_1_PI * 0.25f);
    }

    return RGBSpectrum{0};
}

void OSLBasedIntegrator::setup(Scene* scene) {
    Integrator::setup(scene);
    shadingsys = scene->shadingsys.get();
    shaders = &scene->shaders;
    background_shader = &scene->background_shader;
    thread_info = shadingsys->create_thread_info();
    ctx = shadingsys->get_context(thread_info);
}

WhittedIntegrator::WhittedIntegrator()
    : OSLBasedIntegrator()
{}

WhittedIntegrator::WhittedIntegrator(Camera* cam_ptr, Film* flm_ptr, Sampler* spl_ptr, Recorder* rec)
    : OSLBasedIntegrator(cam_ptr, flm_ptr, spl_ptr, rec)
{}

RGBSpectrum WhittedIntegrator::Li(const Ray& r, const RecordContext& rctx) const {
    RGBSpectrum Li{0};
    Intersection isect;
    Ray ray(r);
    OSL::ShaderGlobals sg, lighting_sg;
    RGBSpectrum throughput{1};

    LightPath p;
    LightPathEvent e_start;
    e_start.type = EStart;
    e_start.event_position = r.origin;
    e_start.ray_direction = r.direction;
    e_start.Li = Li;
    p.record(std::move(e_start));

    while (true) {
        if (!accel_ptr->intersect(ray, isect)) {
            p.record(EBackground, isect, throughput, Li);
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
            Li += throughput * ret.Le;

        BSDFSample sample;
        auto sampled_f = ret.surface.sample(sg, sample, sampler_ptr->random4f());
        if (sample.mode != ScatteringMode::Specular) {
            auto light_cnt = lights->size();
            if (light_cnt == 0)
                return RGBSpectrum(0.f);
            int sampled_light_idx = std::min(static_cast<size_t>(sampler_ptr->randomf() * light_cnt), light_cnt - 1);

            auto light_ptr = lights->at(sampled_light_idx).get();
            Vec3f light_dir;
            float light_pdf, bsdf_pdf;

            LightRecord lrec{isect.P};
            light_ptr->sample(lrec);
            lrec.shading_pt = isect.P;
            light_dir = -lrec.get_light_dir();
            KazenRenderServices::globals_from_lightrec(lighting_sg, lrec);
            auto light_shader = (*shaders)[light_ptr->shader_name];
            if (light_shader == nullptr)
                throw std::runtime_error(fmt::format("Light shader for name : {} does not exist..", light_ptr->shader_name));
            shadingsys->execute(*ctx, *light_shader, sg);
            ShadingResult lighting_ret;
            process_closure(lighting_ret, sg.Ci, RGBSpectrum{1}, false);
            lighting_ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);
            light_ptr->prepare(lighting_ret.Le);

            auto Ls = light_ptr->eval(isect, lrec.get_light_dir(), sampler_ptr->random3f()) / lrec.pdf;

            if (!base::is_zero(Ls)) {
                //float cos_theta_v = dot(light_dir, isect.N);
                float cos_theta_v = dot(light_dir, isect.shading_normal);
                sample.wo = isect.to_local(light_dir);
                auto f = ret.surface.eval(sg, sample);
                //recorder->print(rctx, fmt::format("cos theta : {}, f : {}, Ls : {}, light_pdf : {}", cos_theta_v, f, Ls, light_pdf));
                Li += throughput * (f * Ls * cos_theta_v) * lights->size();
            }

            break;
        }
        else {
            if (sampler_ptr->randomf() < 0.99f) {
                throughput *= sampled_f;
                ray = Ray(isect.P, isect.to_world(sample.wo));
            }
            else
                break;
        }
    }

    p.record(EReflection, isect, RGBSpectrum{0}, Li);
    recorder->record(p, rctx);

    return Li;
}

PathMatsIntegrator::PathMatsIntegrator()
    : OSLBasedIntegrator()
{}

PathMatsIntegrator::PathMatsIntegrator(Camera* cam_ptr, Film* flm_ptr, Sampler* spl_ptr, Recorder* rec)
    : OSLBasedIntegrator(cam_ptr, flm_ptr, spl_ptr, rec)
{}

RGBSpectrum PathMatsIntegrator::Li(const Ray& r, const RecordContext& rctx) const {
    Intersection its;
    if (!accel_ptr->intersect(r, its))
        return 0.f;

    RGBSpectrum Li{0}, throughput{1};
    Ray ray(r);
    int depth = 1;
    float eta = 0.95f;
    BSDFSample sample;
    OSL::ShaderGlobals sg;
    
    while (true) {
        KazenRenderServices::globals_from_hit(sg, ray, its);
        auto shader_ptr = (*shaders)[its.shader_name];
        ShadingResult ret;
        shadingsys->execute(*ctx, *shader_ptr, sg);
        process_closure(ret, sg.Ci, RGBSpectrum{1}, false);
        ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);

        if (its.is_light) {
            //Li += throughput * ret.Le;
            auto Ls = lights->at(its.light_id)->eval(its, -ray.direction, sampler_ptr->random3f());
            Li += throughput * Ls;
        }

        //float prob = std::min(base::max_component(throughput) * eta * eta, 0.99f);
        float prob = 0.98f;
        auto rand = sampler_ptr->randomf();
        if (rand >= prob)
            return Li;
        throughput /= prob;

        auto f = ret.surface.sample(sg, sample, sampler_ptr->random4f());
        throughput *= f;
        ray = Ray(its.P, its.to_world(sample.wo));
        if (!accel_ptr->intersect(ray, its))
            return Li;
    }
}

PathEmsIntegrator::PathEmsIntegrator()
    : OSLBasedIntegrator()
{}

PathEmsIntegrator::PathEmsIntegrator(Camera* cam_ptr, Film* flm_ptr, Sampler* spl_ptr, Recorder* rec)
    : OSLBasedIntegrator(cam_ptr, flm_ptr, spl_ptr, rec)
{}

RGBSpectrum PathEmsIntegrator::Li(const Ray& r, const RecordContext& rctx) const {
    Intersection its;
    if (!accel_ptr->intersect(r, its))
        return 0.f;

    RGBSpectrum Li{0}, throughput{1};
    Ray ray(r);
    int depth = 1;
    float eta = 0.95f;
    bool is_specular = true;
    BSDFSample sample;
    OSL::ShaderGlobals sg;
    float pdf = 0.f;
    float light_pdf, bsdf_pdf;

    LightPath p;

    while (true) {
        KazenRenderServices::globals_from_hit(sg, ray, its);
        auto shader_ptr = (*shaders)[its.shader_name];
        ShadingResult ret;
        shadingsys->execute(*ctx, *shader_ptr, sg);
        process_closure(ret, sg.Ci, RGBSpectrum{1}, false);
        ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);

        if (its.is_light) {
            //Li += throughput * ret.Le * is_specular;
            auto Ls = lights->at(its.light_id)->eval(its, -ray.direction, sampler_ptr->random3f());
            Li += throughput * Ls * is_specular;
            p.record(EEmission, its, throughput, Li);
        }

        auto sampled_f = ret.surface.sample(sg, sample, sampler_ptr->random4f());
        if (sample.mode != ScatteringMode::Specular) {
            is_specular = false;
            auto light_cnt = lights->size();
            if (light_cnt == 0)
                return RGBSpectrum(0.f);

            auto light_ptr = get_random_light(sampler_ptr->randomf(), pdf);
            auto light_shader_ptr = (*shaders)[light_ptr->shader_name];
            ShadingResult light_ret;
            shadingsys->execute(*ctx, *light_shader_ptr, sg);
            process_closure(light_ret, sg.Ci, RGBSpectrum{1}, false);
            light_ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);
            light_ptr->prepare(light_ret.Le);

            Vec3f light_dir;
            auto Ls = light_ptr->sample(its, light_dir, light_pdf, accel_ptr);

            if (!base::is_zero(Ls)) {
                //float cos_theta_v = dot(light_dir, isect.N);
                float cos_theta_v = dot(light_dir, its.shading_normal);
                sample.wo = its.to_local(light_dir);
                auto f = ret.surface.eval(sg, sample);
                //recorder->print(rctx, fmt::format("cos theta : {}, f : {}, Ls : {}, light_pdf : {}", cos_theta_v, f, Ls, light_pdf));
                Li += throughput * f * Ls * cos_theta_v / pdf;
            }
        }
        else {
            is_specular = true;
        }

        if (depth >= 3) {
            float prob = std::min(base::max_component(throughput) * eta * eta, 0.99f);
            if (sampler_ptr->randomf() >= prob) {
                p.record(ERouletteCut, its, throughput, Li);
                break;
            }
            throughput /= prob;
        }

        // sampled incorrect bsdf value when hitting light
        throughput *= sampled_f;
        p.record(EReflection, its, throughput, Li);
        if (base::is_zero(throughput))
            break;
        eta *= 0.9;
        ray = Ray(its.P, its.to_world(sample.wo));
        if (!accel_ptr->intersect(ray, its))
            break;

        depth += 1;
    }

    recorder->record(p, rctx);
    return Li;
}

PathIntegrator::PathIntegrator()
    : OSLBasedIntegrator()
{}

PathIntegrator::PathIntegrator(Camera* cam_ptr, Film* flm_ptr, Sampler* spl_ptr, Recorder* rec)
    : OSLBasedIntegrator(cam_ptr, flm_ptr, spl_ptr, rec)
{}

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

    Intersection its;
    if (!accel_ptr->intersect(r, its)) {
        KazenRenderServices::globals_from_miss(sg, ray, its);
        return process_bg_closure(background_shader);
    }

    RGBSpectrum Li{0}, throughput{1};
    float eta = 0.95f;
    float lpdf, mpdf = 1.f;
    float mis_weight = 1.f;
    BSDFSample bsdf_sample;
    //bsdf_sample.pdf = 0;
    bool last_bounce_specular = true;

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

    for (int depth = 1; depth < max_depth; ++depth) {
        OSL::ShaderGlobals sg;

        KazenRenderServices::globals_from_hit(sg, ray, its);
        auto shader_ptr = (*shaders)[its.shader_name];
        if (shader_ptr == nullptr)
            throw std::runtime_error(fmt::format("Shader for name : {} does not exist..", its.shader_name));
        shadingsys->execute(*ctx, *shader_ptr, sg);
        ShadingResult ret, light_ret;
        process_closure(ret, sg.Ci, RGBSpectrum{1}, false);
        ret.surface.compute_pdfs(sg, throughput, false);

        float pdf;
        auto light_ptr = get_random_light(sampler_ptr->randomf(), pdf);
        auto light_shader = (*shaders)[light_ptr->shader_name];
        if (light_shader == nullptr)
            throw std::runtime_error(fmt::format("Light shader for name : {} does not existj..", light_ptr->shader_name));
        shadingsys->execute(*ctx, *light_shader, sg);
        process_closure(light_ret, sg.Ci, RGBSpectrum{1}, false);
        light_ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);
        light_ptr->prepare(light_ret.Le);

        /* *********************************************
            * 1. Le calculation
            * We try out pbrt's impl, add emitted contribution
            * at only first hit and specular bounce coz further
            * emission is counted in the direction light sampling
            * part
            * *********************************************/
        if (its.is_light) {
        //if (last_bounce_specular) {
            auto Ls = light_ptr->eval(its, bsdf_sample.wo, its.P);
            // We have a problem here with interface design...
            //lpdf = light_ptr->pdf(its, );
            mis_weight = last_bounce_specular ? 1.f : power_heuristic(1, mpdf, 1, lpdf);
            Li += mis_weight * throughput * Ls;
            //Li += throughput * ret.Le;
        }

        //auto sampled_f = ret.surface.sample(sg, bsdf_sample);

        /* *********************************************
        * 2. Sampling light to get direct light contribution
        * *********************************************/
        Vec3f light_dir;
        auto Ls = light_ptr->sample(its, light_dir, lpdf, accel_ptr);
        if (!base::is_zero(Ls)) {
            float cos_theta_v = dot(light_dir, its.shading_normal);
            if (cos_theta_v > 0.) {
                bsdf_sample.wo = its.to_local(light_dir);
                auto f = ret.surface.eval(sg, bsdf_sample);
                mis_weight = power_heuristic(1, lpdf, 1, bsdf_sample.pdf);
                mis_weight = std::isnan(mis_weight) ? 0.f : mis_weight;
                Li += mis_weight * throughput * Ls * f * cos_theta_v / pdf;
            }
        }

        // Russian Roulette
        // The time of doing the Russian Roulette will determine
        // which part will be discarded.
        // Here we follow the implementation in mitsuba and will
        // discard the L_direct & L_indirect on condition.
        if (depth >= min_depth) {
            auto prob = std::min(base::max_component(throughput) * eta * eta, 0.99f);
            if (prob < sampler_ptr->randomf()) {
                p.record(ERouletteCut, its, throughput, Li);
                return Li;
            }
            throughput /= prob;
        }

        /* *********************************************
        * 3. Sampling material to get next direction
        * *********************************************/
        auto f = ret.surface.sample(sg, bsdf_sample, sampler_ptr->random4f());
        last_bounce_specular = bsdf_sample.mode == ScatteringMode::Specular;
        throughput *= f;
        if (base::is_zero(throughput))
            return Li;

        mpdf = bsdf_sample.pdf;
        ray = Ray(its.P, its.to_world(bsdf_sample.wo));

        if (!accel_ptr->intersect(ray, its)) {
            KazenRenderServices::globals_from_miss(sg, ray, its);
            return Li + throughput * process_bg_closure(background_shader);
        }

        // LightPath event recording
        if (its.is_light)
            p.record(EEmission, its, throughput, Li);
        else
            p.record(EReflection, its, throughput, Li);

        depth += 1;
    }

    recorder->record(p, rctx);
    
    return Li;
}
