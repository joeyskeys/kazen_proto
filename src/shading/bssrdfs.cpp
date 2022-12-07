
#include "base/utils.h"
#include "base/vec.h"
#include "shading/bsdf.h"
#include "shading/bssrdfs.h"
#include "shading/context.h"

static bool find_po(ShadingContext* ctx, bssrdf_profile_sample_func& profile_func, const Vec4f& rand) {
    auto dipole_params = reinterpret_cast<KpDipoleParams*>(ctx->data);
    auto bssrdf_sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);

    // Choose a projection axis
    Frame frame;
    float u = rand[0];
    if (u < 0.5f) {
        frame = Frame(ctx->frame.s, ctx->frame.t, ctx->frame.n);
        u = u * 2;
    }
    else if (u < 0.75f) {
        frame = Frame(ctx->frame.t, ctx->frame.n, ctx->frame.s);
        u = (u - 0.5f) * 4;
    }
    else {
        frame = Frame(ctx->frame.n, ctx->frame.s, ctx->frame.t);
        u = (u - 0.75f) * 4;
    }

    // A simple strategy to sample a channel
    // Not enough random number here available, so reuse one from frame sampling
    // But in the coming bsdf sampling, we still need another Vec3 random numbers
    size_t ch = 3 * 0.99999f * u;
    const float disk_radius = profile_func(ctx.data, ch, rand[1]);

    if (disk_radius == 0.f)
        return false;

    if (disk_radius >= dipole_params->max_radius)
        return false;

    auto phi = constants::two_pi<float>() * rand[2];
    auto disk_point = Vec3f{disk_radius * std::cos(phi), 0.f, disk_radius * std::sin(phi)};


    auto h = std::sqrt(square(dipole_params->max_radius) - square(disk_radius));
    auto hn = h * frame.n;
    auto entry_pt = disk_point + hn;
    auto ray_dir = base::normalize(-hn);

    const static int max_intersection_cnt = 10;
    int found_intersection = 0;
    Intersection isects[max_intersection_cnt];
    auto start_pt = entry_pt;
    for (int i = 0; i < max_intersection_cnt; i++) {
        Ray r(start_pt, ray_dir);
        ctx->accel->intersect(r, isects[i]);
        start_pt = isects[i].P;
        ++found_intersection;
    }

    // Randomly chose one intersection
    if (found_intersection == 0)
        return false;
    else if (found_intersection == 1) {
        bssrdf_sample->po = isects[0].P;
        return true;
    }
    else {
        uint idx = rand[3] * max_intersection_cnt * 0.999999f;
        bssrdf_sample->po = isects[idx];
        return true;
    }
}

static inline float sample_standard_dipole_func(const void* data, uint32_t ch, const float u) {
    auto dipole_params = reinterpret_cast<KpDipoleParams*>(data)
    // TODO : We need to sample a channel, use fixed first channel for now
    return sample_exponential_distribution(dipole_params->sigma_tr[ch], u);
}

static RGBSpectrum sample_dipole(ShadingContext* ctx, bssrdf_profile_sample_func& profile_sample_func,
    const Vec4f& rand)
{
    auto found_po = find_po(ctx, profile_samplefunc, rand);
    if (!found_po)
        return 0;

    auto bssrdf_sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);
    // We need to unify the interfaces now
    auto f = bssrdf_sample->sampled_brdf->sample(ctx, base::head<3>(rand));
    return f;
}

static RGBSpectrum eval_standard_dipole_func(const void* data, const Vec3f& pi,
    const Vec3f& wi, const Vec3f& po, const Vec3f& wo)
{
    auto params = reinterpret_cast<KpDipoleParams*>(data);
    const float sqr_radius = base::length_squared(pi - po);
    if (sqr_radius > square(params->max_radius))
        return 0;

    const float Fdr = fresnel_internel_diffuse_reflectance(params->eta);
    const float A = (1. + Fdr) / (1. - Fdr);

    // Here's a design decision to consider:
    // Each vector arithmetic operation could be a small for loop (or not if
    // vectorization library is used), write code in this way saves typing but
    // could be a penalty to performance.
    auto sigma_a = base::to_vec3(params->sigma_a);
    auto sigma_s = base::to_vec3(params->sigma_s);
    auto sigma_s_prime = sigma_s * (1. - params->g);
    auto sigma_t_prime = sigma_s_prime + sigma_a;
    auto sigma_tr = base::to_vec3(params->sigma_tr);

    auto zr = 1. / sigma_t_prime;
    auto zv = -zr * (1.f + (4.f / 3.f) * A);

    auto dr = base::sqrt(sqr_radius + zr * zr);
    auto dv = base::sqrt(sqr_radius + zv * zv);

    auto rcp_dr = 1. / dr;
    auto rcp_dv = 1. / dv;
    auto sigma_tr_dr = sigma_tr * dr;
    auto sigma_tr_dv = sigma_tr * dv;
    auto kr = zr * (sigma_tr_dr + 1.f) * base::square(rcp_dr);
    auto kv = zv * (sigma_tr_dv + 1.f) * base::square(rcp_dv);
    auto er = base::exp(-sigma_tr_dr) * rcp_dr;
    auto ev = base::exp(-sigma_tr_dv) * rcp_dv;
    
    return constants::one_div_pi<float>() * 0.25 * (kr * er - kv * ev);
}

static RGBSpectrum eval_dipole(ShadingContext* ctx, bssrdf_profile_eval_func& profile_eval_func)
{
    auto dipole_params = reinterpret_cast<KpDipoleParams*>(ctx->data);
    auto bssrdf_sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);
    auto ret = profile_eval_func(ctx->data, Vec3f(0.f), bssrdf_sample->wi,
        bssrdf_sample->po, bssrdf_sample->wo);

    // We have a problem here, since bssrdf involes "two shading point", we'll have
    // two normals and two directions(for wi & wo). Should we still use the shading
    // space representation of directions?
    auto cos_on = std::min(std::abs(cos(bssrdf_sample->wo)), 1.f);
    auto fo = fresnel_trans_dielectric(dipole_params->eta, cos_on);
    auto cos_in = std::min(std::abs(cos(ctx->isect_i.wi)), 1.f);
    auto fi = fresnel_trans_dielectric(dipole_params->eta, cos_on);

    // Noramlization factor
    // checkout "A Quantized-Diffusion Model for Rendering Translucent Materials" page 6
    // equation 14
    auto c = 1.f - fresnel_first_moment_x2(dipole_params->eta);

    return ret * fo * fi / c;
}

RGBSpectrum KpDipole::eval(ShadingContext* ctx) {
    return eval_dipole(ctx, eval_standard_dipole_func)
}

RGBSpectrum KpDipole::sample(ShadingContext* ctx, const Vec4f& rand) {
    return sample_dipole(ctx, sample_standard_dipole_func, rand);
}