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
    //assert(cnt > 0);
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

RGBSpectrum NormalIntegrator::Li(const Ray& r, const RecordContext* rctx) const {
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

RGBSpectrum AmbientOcclusionIntegrator::Li(const Ray& r, const RecordContext* rctx) const {
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
    /*
    shadingsys = scene->shadingsys.get();
    shaders = &scene->shaders;
    background_shader = scene->background_shader;
    thread_info = shadingsys->create_thread_info();
    ctx = shadingsys->get_context(thread_info);
    */
    shading_ctx.engine_ptr = &shading_engine;
    shading_engine.osl_shading_sys = scene->shadingsys.get();
    shading_engine.osl_thread_info = scene->shadingsys->create_thread_info();
    shading_engine.osl_shading_ctx = scene->shadingsys->get_context(
        shading_engine.osl_thread_info);
    shading_engine.background_shader = scene->background_shader;
    shading_engine.shaders = &scene->shaders;

    
    shading_ctx.accel = scene->accelerator.get();
    //shading_ctx.shaders = shaders;
}

WhittedIntegrator::WhittedIntegrator()
    : OSLBasedIntegrator()
{}

WhittedIntegrator::WhittedIntegrator(Camera* cam_ptr, Film* flm_ptr, Sampler* spl_ptr, Recorder* rec)
    : OSLBasedIntegrator(cam_ptr, flm_ptr, spl_ptr, rec)
{}

RGBSpectrum WhittedIntegrator::Li(const Ray& r, const RecordContext* rctx) const {
    RGBSpectrum Li{0};
    Intersection isect;
    shading_ctx.isect_i = &isect;
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

        KazenRenderServices::globals_from_hit(sg, ray, isect);
        shading_engine.execute(isect.shader_name, sg);
        ShadingResult ret;
        process_closure(ret, sg.Ci, RGBSpectrum{1}, false);
        ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);
        shading_ctx.sg = &sg;

        if (isect.is_light)
            Li += throughput * ret.Le;

        BSDFSample sample;
        shading_ctx.closure_sample = &sample;
        auto sampled_f = ret.surface.sample(&shading_ctx, sampler_ptr);
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
            shading_engine.execute(light_ptr->shader_name, sg);
            ShadingResult lighting_ret;
            process_closure(lighting_ret, sg.Ci, RGBSpectrum{1}, false);
            lighting_ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);
            light_ptr->prepare(lighting_ret.Le);

            auto Ls = light_ptr->eval(isect, lrec.get_light_dir(), sampler_ptr->random3f()) / lrec.pdf;

            if (!base::is_zero(Ls)) {
                //float cos_theta_v = dot(light_dir, isect.N);
                float cos_theta_v = dot(light_dir, isect.shading_normal);
                sample.wo = isect.to_local(light_dir);
                auto f = ret.surface.eval(&shading_ctx);
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

RGBSpectrum PathMatsIntegrator::Li(const Ray& r, const RecordContext* rctx) const {
    Intersection its;
    shading_ctx.isect_i = &its;
    if (!accel_ptr->intersect(r, its))
        return 0.f;

    RGBSpectrum Li{0}, throughput{1};
    Ray ray(r);
    int depth = 1;
    float eta = 0.95f;
    BSDFSample sample;
    shading_ctx.closure_sample = &sample;
    OSL::ShaderGlobals sg;
    
    while (true) {
        KazenRenderServices::globals_from_hit(sg, ray, its);
        ShadingResult ret;
        shading_engine.execute(its.shader_name, sg);
        process_closure(ret, sg.Ci, RGBSpectrum{1}, false);
        ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);
        shading_ctx.sg = &sg;

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

        auto f = ret.surface.sample(&shading_ctx, sampler_ptr);
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

RGBSpectrum PathEmsIntegrator::Li(const Ray& r, const RecordContext* rctx) const {
    Intersection its;
    shading_ctx.isect_i = &its;
    if (!accel_ptr->intersect(r, its))
        return 0.f;

    RGBSpectrum Li{0}, throughput{1};
    Ray ray(r);
    int depth = 1;
    float eta = 0.95f;
    bool is_specular = true;
    BSDFSample sample;
    shading_ctx.closure_sample = &sample;
    OSL::ShaderGlobals sg;
    float pdf = 0.f;
    float light_pdf, bsdf_pdf;

    LightPath p;

    while (true) {
        KazenRenderServices::globals_from_hit(sg, ray, its);
        ShadingResult ret;
        shading_engine.execute(its.shader_name, sg);
        process_closure(ret, sg.Ci, RGBSpectrum{1}, false);
        ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);
        shading_ctx.sg = &sg;

        if (its.is_light) {
            //Li += throughput * ret.Le * is_specular;
            auto Ls = lights->at(its.light_id)->eval(its, -ray.direction, sampler_ptr->random3f());
            Li += throughput * Ls * is_specular;
            p.record(EEmission, its, throughput, Li);
        }

        auto sampled_f = ret.surface.sample(&shading_ctx, sampler_ptr);
        if (sample.mode != ScatteringMode::Specular) {
            is_specular = false;
            auto light_cnt = lights->size();
            if (light_cnt == 0)
                return RGBSpectrum(0.f);

            auto light_ptr = get_random_light(sampler_ptr->randomf(), pdf);
            ShadingResult light_ret;
            shading_engine.execute(light_ptr->shader_name, sg);
            process_closure(light_ret, sg.Ci, RGBSpectrum{1}, false);
            light_ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);
            light_ptr->prepare(light_ret.Le);

            Vec3f light_dir;
            auto Ls = light_ptr->sample(its, light_dir, light_pdf, accel_ptr);

            if (!base::is_zero(Ls)) {
                //float cos_theta_v = dot(light_dir, isect.N);
                float cos_theta_v = dot(light_dir, its.shading_normal);
                sample.wo = its.to_local(light_dir);
                auto f = ret.surface.eval(&shading_ctx);
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

RGBSpectrum PathIntegrator::Li(const Ray& r, const RecordContext* rctx) const {
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
    shading_ctx.isect_i = &its;
    OSL::ShaderGlobals sg;
    shading_ctx.sg = &sg;
    if (!accel_ptr->intersect(r, its)) {
        if (shading_engine.background_shader) {
            shading_engine.execute(shading_engine.background_shader, sg);
            KazenRenderServices::globals_from_miss(sg, r, its);
            return process_bg_closure(sg.Ci);
        }
        else
            return 0;
    }

    RGBSpectrum Li{0}, throughput{1};
    float eta = 0.95f;
    float lpdf, mpdf = 1.f;
    float mis_weight = 1.f;
    BSDFSample bsdf_sample;
    BSSRDFSample bssrdf_sample;
    shading_ctx.closure_sample = &bsdf_sample;
    //bsdf_sample.pdf = 0;
    bool last_bounce_specular = true;

    Ray ray(r);
    shading_ctx.ray = &ray;

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

    bool single_emissive_surface = true;

    for (int depth = 1; depth < max_depth; ++depth) {
        KazenRenderServices::globals_from_hit(sg, ray, its);
        shading_engine.execute(its.shader_name, sg);
        ShadingResult ret, light_ret, bssrdf_ret;
        process_closure(ret, sg.Ci, RGBSpectrum{1}, false);
        ret.surface.compute_pdfs(sg, throughput, false);
        ret.bssrdf.compute_pdfs(sg, throughput, false);
        ret.bssrdf.precompute(&shading_ctx);

        float pdf;
        auto light_ptr = get_random_light(sampler_ptr->randomf(), pdf);
        if (light_ptr) {
            // Now we don't explicitly need a light in scene
            if (light_ptr->shader_name == "")
                throw std::runtime_error("Sampled light doesn't have a shader");
            shading_engine.execute(light_ptr->shader_name, sg);
            process_closure(light_ret, sg.Ci, RGBSpectrum{1}, false);
            light_ret.surface.compute_pdfs(sg, RGBSpectrum{1}, false);
            light_ptr->prepare(light_ret.Le);
        }

        /* *********************************************
            * 1. Le calculation
            * We try out pbrt's impl, add emitted contribution
            * at only first hit and specular bounce coz further
            * emission is counted in the direction light sampling
            * part
            * Update: not anymore, if we hit emissive surface we
            * add its contribution directly
            * *********************************************/
        if (its.is_light) {
        //if (last_bounce_specular) {
            if (!single_emissive_surface || base::dot(ray.direction, its.shading_normal) < 0) {
                auto Ls = light_ptr->eval(its, bsdf_sample.wo, its.P);
                // We have a problem here with interface design...
                //lpdf = light_ptr->pdf(its, );
                mis_weight = last_bounce_specular ? 1.f : power_heuristic(1, mpdf, 1, lpdf);
                Li += mis_weight * throughput * Ls;
                p.record(EEmission, its, throughput, Li);
            }
        }

        //auto sampled_f = ret.surface.sample(sg, bsdf_sample);

        /* *********************************************
        * 2. Sampling light to get direct light contribution
        * *********************************************/
        Vec3f light_dir;
        if (light_ptr) {
            // TODO: remove the contribution from itself it the sampled light
            // is the hit object
            auto Ls = light_ptr->sample(its, light_dir, lpdf, accel_ptr);
            if (!base::is_zero(Ls)) {
                float cos_theta_v = dot(light_dir, its.shading_normal);
                if (cos_theta_v > 0.) {
                    bsdf_sample.wo = its.to_local(light_dir);
                    auto f = ret.surface.eval(&shading_ctx);
                    mis_weight = power_heuristic(1, lpdf, 1, bsdf_sample.pdf);
                    mis_weight = std::isnan(mis_weight) ? 0.f : mis_weight;
                    Li += mis_weight * throughput * Ls * f * cos_theta_v / pdf;
                }
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
                break;
            }
            throughput /= prob;
        }

        /* *********************************************
        * 3. Sampling material to get next direction
        * *********************************************/
        auto f = ret.surface.sample(&shading_ctx, sampler_ptr);
        last_bounce_specular = bsdf_sample.mode == ScatteringMode::Specular;
        throughput *= f;
        mpdf = bsdf_sample.pdf;
        ray = Ray(its.P, its.to_world(bsdf_sample.wo));

        // Update ray differential
        ray.origin_dx = ray.origin + its.dpdx;
        ray.origin_dy = ray.origin + its.dpdy;
        ray.direction_dx = ray.direction_dy = ray.direction;
        
        //p.record(EReflection, its, throughput, Li, sp, ray.direction);
        if (base::is_zero(throughput))
            break;

        // Account for subsurface scattering if applicable
        // Here's a major design difference between pbrt-v3 and appleseed.
        // The former use bssrdf to sample a out-going point and get the bsdf,
        // use that bsdf to get the next direction. While in appleseed the
        // out-going direction is sampled directly in bssrdf.
        // Update: will take the path of appleseed, do osl execution in the
        // bssrdf and update related fields, code now is kinda messy..
        if (ret.bssrdf) {
            shading_ctx.closure_sample = &bssrdf_sample;
            auto f = ret.bssrdf.sample(&shading_ctx, sampler_ptr);
            if (f.is_zero() || bssrdf_sample.pdf == 0) break;

            throughput *= f / bssrdf_sample.pdf;
            light_ptr = get_random_light(sampler_ptr->randomf(), pdf);
            auto Ls = light_ptr->eval(shading_ctx.isect_o, bssrdf_sample.wo, bssrdf_sample.po);
            Li += throughput * Ls;

            if (base::is_zero(bssrdf_sample.brdf_f) || bssrdf_sample.brdf_pdf == 0.)
                break;
            throughput *= bssrdf_sample.brdf_f / bssrdf_sample.brdf_pdf;
            // Probably have handle specular bounce here?

            // how to convert this to world space
            ray = Ray(bssrdf_sample.po,
                shading_ctx.isect_o.frame.to_world(bssrdf_sample.wo));
        }

        if (!accel_ptr->intersect(ray, its)) {
            if (shading_engine.background_shader) {
                KazenRenderServices::globals_from_miss(sg, ray, its);
                shading_engine.execute(shading_engine.background_shader, sg);
                Li += throughput * process_bg_closure(sg.Ci);
            }
            p.record(EBackground, its, throughput, Li);
            break;
        }

        depth += 1;
    }

    recorder->record(p, rctx);
    
    return Li;
}
