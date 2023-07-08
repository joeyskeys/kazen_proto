#include <functional>

#include <OSL/oslexec.h>
#include <boost/math/constants/constants.hpp>

#include "base/mis.h"
#include "base/utils.h"
#include "base/vec.h"
#include "shading/bsdf.h"
#include "shading/bssrdfs.h"
#include "shading/context.h"
#include "shading/renderservices.h"

namespace constants = boost::math::constants;


/*********************************************
 * According to [1], light could separated into three components:
 * reduced-intensity, single-scattering and multiple-scattering components:
 * S = S^(0) + S^(1) + S_d
 * As for bssrdf model, the first term is irrelevant, so we only consider
 * the last two terms.
 * "Single scattering occurs only when the refracted incoming and outgoing
 * rays intersect", from [2], this condition
 * is hard to meet and my understanding is that most renderers don't 
 * compute this term. It leaves us only the third term.
 * As for the third term, "A better dipole" giveed a normalized version
 * (equation 1):
 * S_d = 1 / pi *    F_t(wi)   *     R_d      * F_t(wo) / (1 - 2 * C1 / eta)
 *                ------------   ------------   ----------------------------
 *                fresnel term   spatial term          directional term
 * Here three parts are multiplied directly coz we've made a assumption here
 * that spatial term and directional term are independent thus the name
 * "separable".
 * 
 * As in the dipole like models, R_d is approximated by diffusion term and 
 * seems the most common BSSRDF models are based dipole methods, I saw quite
 * a lot paper refer to it directly as diffusion term.
 * 
 * Since evaluation BSSRDF by parameters like sigma_t, sigma_a is quite not
 * intuitive, [3] and [4] gives methods to evaluate Rd in a analytical way
 * and allow us to compute alpha_prime first, and then use it to compute
 * simga_t and sigma_a. But my question is that you need to compute Rd to
 * get S, but now you evaluate Rd first and then use the result to compute
 * Rd again. Appleseed applies this method. Methods told in [3] section 4
 * uses the Rd approximation eqution in [2] section 2.4 to compute
 * alpha_prime(even not analytically) and then use the main equation to
 * calculate Rd again... not sure if this is reasonable.
 * 
 * As I dig deep enough, I saw all those equations in papers back in the 50s.
 * Those equations were derived firstly in the physics community by studying
 * the wave and particles, pure physics. Introduced into medical community
 * when studying the light propagation in the skin, blood or similar tissues.
 * Finally introduced into CG community(more or less like a summary and explaining). 
 * I could not really understand all these equations before I acutally dive
 * into the physics.
 * 
 * [1] A Better Dipole
 *     http://www.eugenedeon.com/wp-content/uploads/2014/04/betterdipole.pdf
 * 
 * [2] A Practical Model for Subsurface Light Transport
 *     https://graphics.stanford.edu/papers/bssrdf/bssrdf.pdf
 * 
 * [3] A Rapid Hierarchical Rendering Technique for Translucent Materials
 *     http://graphics.ucsd.edu/~henrik/papers/fast_bssrdf/fast_bssrdf.pdf
 * 
 * [4] Texture mapping for the Better Dipole model
 *     https://graphics.pixar.com/library/TexturingBetterDipole/paper.pdf
 * 
 *********************************************/

using rd_func = std::function<float(const float)>;

RGBSpectrum compute_alpha_prime(const rd_func& rd_f, const RGBSpectrum& rd) {
    // This function is basically copied from [4] section 3, appleseed uses
    // it too.
    // Basically a bisection to estimate the alpha_prime since the rd_func
    // is monotonic.
    RGBSpectrum ap;
    for (int i = 0; i < 3; i++) {
        int iter_cnt = 20;
        float x0 = 0.f, x1 = 1.f;
        float xmid, x;

        for (int j = 0; j < iter_cnt; j++) {
            float xmid = 0.5f * (x0 + x1);
            x = rd_f(xmid);
            x < rd[i] ? x0 = xmid : x1 = xmid;
        }
        ap[i] = 0.5f * (x0 + x1);
    }

    return ap;
}

// The following two class are basically copied from appleseed.
// The Rd compute func for better dipole model will need to store
// C1 and C2, like a closure. Make them classes will help to unify
// the interface.
// Opaque to user, so put them here directly.
class ComputeRdStandardDipole {
public:
    explicit ComputeRdStandardDipole(const float eta) {
        // Using equations from [2] page 3
        const float Fdr = fresnel_internel_diffuse_reflectance(eta);
        m_a = (1.f + Fdr) / (1.f - Fdr);
    }

    float compute_rd(const float alpha_prime) const {
        // [3] eq.15
        const float sqrt_3ap = std::sqrt(3.f * (1.f - alpha_prime));
        return (0.5f * alpha_prime) * (1.f + std::exp(-(4.f / 3.f) * m_a * sqrt_3ap))
            * std::exp(-sqrt_3ap);
    }

private:
    float m_a;
};

class ComputeRdBetterDipole {
public:
    explicit ComputeRdBetterDipole(const float eta)
        : m_two_c1(fresnel_first_moment_x2(eta))
        , m_three_c2(fresnel_second_moment_x3(eta))
    {}

    float compute_rd(const float alpha_prime) const {
        // [4] page 2.
        float cphi = 0.25f * (1.f - m_two_c1);
        float ce = 0.5f * (1.f - m_three_c2);
        float four_a = (1.f + m_three_c2) / cphi;
        float mu_tr_d = std::sqrt((1.f - alpha_prime) * (2.f - alpha_prime) / 3.f);
        const float myexp = std::exp(-four_a * mu_tr_d);
        return 0.5f * square(alpha_prime)
                    * std::exp(-std::sqrt(3.f * (1.f - alpha_prime) / (2.f - alpha_prime)))
                    * (ce * (1.f + myexp) + cphi / mu_tr_d * (1.f - myexp));
    }

private:
    const float m_two_c1;
    const float m_three_c2;
};

using compute_alpha_prime_func = std::function<RGBSpectrum(const RGBSpectrum&)>;

static void dipole_precompute(ShadingContext* ctx, const compute_alpha_prime_func& f) {
    auto params = reinterpret_cast<KpDipoleParams*>(ctx->data);
    auto sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);
    sample->sigma_tr = 1.f / base::to_vec3(params->mfp);
    sample->alpha_prime = f(base::to_vec3(params->Rd));
    sample->sigma_t_prime = sample->sigma_tr / base::sqrt(3.f * (1.f -
        sample->alpha_prime));
    sample->sigma_t = sample->sigma_t_prime / (1 - params->g);
    sample->sigma_s_prime = sample->alpha_prime * sample->sigma_t_prime;
    sample->sigma_s = sample->sigma_s_prime / (1 - params->g);
    sample->sigma_a = sample->sigma_t_prime - sample->sigma_s_prime;
}

using eval_profile_func = std::function<RGBSpectrum(ShadingContext*,
    const Vec3f&, const Vec3f&, const Vec3f&, const Vec3f&)>;

static RGBSpectrum separable_bssrdf_eval(
    ShadingContext* ctx,
    const eval_profile_func& eval_profile,
    const Vec3f& pi,
    const Vec3f& wi,
    const Vec3f& po,
    const Vec3f& wo)
{
    auto params = reinterpret_cast<const KpDipoleParams*>(ctx->data);
    auto Rd = eval_profile(ctx, pi, wi, po, wo);
    const float cos_o = std::min(std::abs(cos_theta(wo)), 1.f);
    auto Fo = fresnel_trans_dielectric(params->eta, cos_o);
    const float cos_i = std::min(std::abs(cos_theta(wi)), 1.f);
    auto Fi = fresnel_trans_dielectric(params->eta, cos_i);
    const float C = 1.f - fresnel_first_moment_x2(params->eta);

    return constants::one_div_pi<float>() * Rd * Fo * Fi / C;
}

static RGBSpectrum standard_dipole_profile_eval(
    ShadingContext* ctx,
    const Vec3f& pi,
    const Vec3f& wi,
    const Vec3f& po,
    const Vec3f& wo)
{
    auto params = reinterpret_cast<KpDipoleParams*>(ctx->data);
    auto sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);
    const float radius_sqr = base::length_squared(pi - po);
    // Following two variable calculation is redundent..
    const float Fdr = fresnel_internel_diffuse_reflectance(params->eta);
    const float A = (1.f + Fdr) / (1.f - Fdr);

    const auto sigma_a = sample->sigma_a;
    const auto sigma_s = sample->sigma_s;
    const auto sigma_s_prime = sigma_s * (1.f - params->g);
    const auto sigma_t_prime = sigma_s_prime + sigma_a;
    const auto alpha_prime = sample->alpha_prime;
    const auto sigma_tr = sample->sigma_tr;

    // We have
    //   zr = 1 / sigma_t_prime
    //   zv = -zr - 2 * zb
    //      = -zr - 4 * A * D
    //
    // where
    //
    //   D = 1 / (3 * sigma_t_prime)
    // 
    // So
    //
    //   zv = -zr - 4 * A / (3 * simga_t_prime)
    //      = -zr - zr * 4/3 * A
    //      = -zr * (1 + 4/3 * A)
    const auto zr = 1.f / sigma_t_prime;
    const auto zv  = -zr * (1.f + (4.f / 3.f) * A);

    // c^2 = a^2 + b^2
    // Calculate the third edge length of a triangle
    const auto dr = base::sqrt(radius_sqr + zr * zr);
    const auto dv = base::sqrt(radius_sqr + zv * zv);
    
    const auto rcp_dr = 1.f / dr;
    const auto rcp_dv = 1.f / dv;
    const auto sigma_tr_dr = sigma_tr * dr;
    const auto sigma_tr_dv = sigma_tr * dv;
    const auto kr = zr * (sigma_tr_dr + 1.f) * base::square(rcp_dr);
    const auto kv = zv * (sigma_tr_dv + 1.f) * base::square(rcp_dv);
    const auto er = base::exp(-sigma_tr_dr) * rcp_dr;
    const auto ev = base::exp(-sigma_tr_dv) * rcp_dv;
    constexpr auto one_div_four_pi = constants::one_div_two_pi<float>() / 2.f;
    return alpha_prime * one_div_four_pi * (kr * er - kv * ev);
}

static RGBSpectrum better_dipole_profile_eval(
    ShadingContext* ctx,
    const Vec3f& pi,
    const Vec3f& wi,
    const Vec3f& po,
    const Vec3f& wo)
{
    auto params = reinterpret_cast<KpDipoleParams*>(ctx->data);
    auto sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);
    const float sqr_radius = base::length_squared(pi - po);
    if (sqr_radius > square(params->max_radius))
        return 0;

    const float two_c1 = fresnel_first_moment_x2(params->eta);
    const float three_c2 = fresnel_second_moment_x3(params->eta);
    const float A = (1.f + three_c2) / (1.f - two_c1);
    const float cphi = 0.25f * (1.f - two_c1);
    const float ce = 0.5f * (1.f - three_c2);

    const auto sigma_a = sample->sigma_a;
    const auto sigma_s = sample->sigma_s;
    const auto sigma_s_prime = sigma_s * (1.f - params->g);
    const auto sigma_t_prime = sigma_s_prime + sigma_a;
    const auto alpha_prime = sample->alpha_prime;
    const auto sigma_tr = sample->sigma_tr;

    const auto D = (2.f * sigma_a + sigma_s_prime) / (3.f *
        base::square(sigma_t_prime));
    const auto zr = 1.f / sigma_t_prime;
    const auto zv = -zr - 4.f * A * D;
    const auto dr = base::sqrt(sqr_radius + zr * zr);
    const auto dv = base::sqrt(sqr_radius + zv * zv);

    const auto rcp_dr = 1.f / dr;
    const auto rcp_dv = 1.f / dv;
    const auto sigma_tr_dr = sigma_tr * dr;
    const auto sigma_tr_dv = sigma_tr * dv;
    const auto cphi_over_D = cphi / D;
    const auto kr = ce * zr * (sigma_tr_dr + 1.f) * base::square(rcp_dr) + cphi_over_D;
    const auto kv = ce * zv * (sigma_tr_dv + 1.f) * base::square(rcp_dv) + cphi_over_D;
    const auto er = base::exp(-sigma_tr_dr) * rcp_dr;
    const auto ev = base::exp(-sigma_tr_dv) * rcp_dv;
    constexpr auto one_div_four_pi = constants::one_div_two_pi<float>() / 2.f;
    return base::square(alpha_prime) * one_div_four_pi * (kr * ev - kv * ev);
}

static float separable_bssrdf_pdf(ShadingContext* ctx, const bssrdf_profile_pdf_func& pdf_func, const float r) {
    auto params = reinterpret_cast<KpDipoleParams*>(ctx->data);
    auto bssrdf_sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);
    float pdf = 0.f;
    
    // Calculate the main pdf of sampled axis
    const float dot_nn = std::abs(base::dot(bssrdf_sample->frame.n, ctx->isect_o.frame.n));
    pdf = bssrdf_sample->axis_prob * bssrdf_sample->pt_prob * dot_nn;
    if (pdf < 0.000001f)
        return 0.f;

    // Mis with other axises
    const auto d = ctx->isect_o.P - ctx->isect_i->P;
    const auto& u = bssrdf_sample->frame.s;
    const auto& v = bssrdf_sample->frame.t;
    const float du = base::length(base::project(d, u));
    const float dv = base::length(base::project(d, v));

    const float dot_un = std::abs(base::dot(u,
        ctx->isect_o.frame.to_world(ctx->isect_o.wo)));
    const float dot_vn = std::abs(base::dot(v,
        ctx->isect_o.frame.to_world(ctx->isect_o.wo)));

    const float pdf_u = pdf_func(ctx, du) * dot_un;
    const float pdf_v = pdf_func(ctx, dv) * dot_vn;

    // TODO : make sure which spaces are the normals actually
    // in.
    // TODO : make this piece of code more intuitive by using enums
    float mis_weight = 1.f;
    switch (bssrdf_sample->sampled_axis) {
        case 0: {
            // Sampled N, unchanged
            mis_weight = mis_power2(pdf, 0.25f * pdf_u, 0.25f * pdf_v);
            break;
        }
        case 1: {
            // Sampled S, T as x axis, N as y axis
            mis_weight = mis_power2(pdf, 0.25f * pdf_u, 0.5f * pdf_v);
            break;
        }
        case 2: {
            // Sampled T, N as x axis, S as y axis
            mis_weight = mis_power2(pdf, 0.5f * pdf_u, 0.25f * pdf_v);
            break;
        }
    }

    return pdf / mis_weight / bssrdf_sample->sample_cnt;
}

static float dipole_profile_pdf(ShadingContext* ctx, const float r) {
    auto params = reinterpret_cast<KpDipoleParams*>(ctx->data);
    auto sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);
    if (r > params->max_radius)
        return 0.f;

    float pdf = 0.f;
    for (uint32_t i = 0; i < 3; i++) {
        // TODO : channel pdf is WRONG
        // Now we are not sampling channel with the right
        // weights
        const float channel_pdf = 0.333334f;
        const float sigma_tr = sample->sigma_tr[i];
        pdf += channel_pdf * exponential_distribution_pdf(r, sigma_tr);
    }

    pdf /= constants::two_pi<float>() * r;

    return pdf;
}

/*********************************************
 * Importance sampling for BSSRDF is a big topic by itself. [1] gave a
 * very intuitive summary for it and introduces their own method, which
 * is adopted in appleseed.
 * Here we use the method introduced in [2] as the default method first.
 * Then we'll implement all the methods briefed in [1] and then do a
 * comparison between them.
 * 
 * [1] BSSRDF Importance Sampling Slides
 * https://pdfs.semanticscholar.org/90da/5211ce2a6f63d50b8616736c393aaf8bf4ca.pdf
 * [2] BSSRDF Importance Sampling
 * https://library.imageworks.com/pdfs/imageworks-library-BSSRDF-sampling.pdf
 *********************************************/

static bool find_po(ShadingContext* ctx, const bssrdf_profile_sample_func& profile_func, Sampler* sampler) {
    auto dipole_params = reinterpret_cast<KpDipoleParams*>(ctx->data);
    auto bssrdf_sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);
    auto rand = sampler->random4f();

    // Sample a channel and a disk radius
    size_t ch = 3 * 0.99999f * rand[0];
    const float disk_radius = profile_func(ctx->closure_sample, ch, rand[1]);

    if (disk_radius == 0.f)
        return false;

    if (disk_radius >= dipole_params->max_radius)
        return false;

    // Choose a projection axis
    auto isect_frame = ctx->isect_i->frame;
    float u = rand[2];
    if (u < 0.5f) {
        bssrdf_sample->frame = Frame(isect_frame.s, isect_frame.t, isect_frame.n);
        u = u * 2;
        bssrdf_sample->axis_prob = 0.5f;
        bssrdf_sample->sampled_axis = 0;
    }
    else if (u < 0.75f) {
        bssrdf_sample->frame = Frame(isect_frame.t, isect_frame.n, isect_frame.s);
        u = (u - 0.5f) * 4;
        bssrdf_sample->axis_prob = 0.25f;
        bssrdf_sample->sampled_axis = 1;
    }
    else {
        bssrdf_sample->frame = Frame(isect_frame.n, isect_frame.s, isect_frame.t);
        u = (u - 0.75f) * 4;
        bssrdf_sample->axis_prob = 0.25f;
        bssrdf_sample->sampled_axis = 2;
    }

    // Sample a phi angle to to get the disk point location on xz plane
    auto phi = constants::two_pi<float>() * rand[3];
    auto disk_point = Vec3f{disk_radius * std::cos(phi), 0.f, disk_radius * std::sin(phi)};
    bssrdf_sample->pt_prob = dipole_profile_pdf(ctx, disk_radius);

    auto h = std::sqrt(square(dipole_params->max_radius) - square(disk_radius));
    auto hn = h * bssrdf_sample->frame.n;
    auto entry_pt = ctx->isect_i->P + ctx->isect_i->frame.to_world(disk_point) + hn;
    auto ray_dir = base::normalize(-hn);

    const static int max_intersection_cnt = 10;
    int found_intersection = 0;
    Intersection isects[max_intersection_cnt];

    std::vector<uint32_t> isect_indice;
    isect_indice.reserve(10);

    auto start_pt = entry_pt;
    float tmax = 2.f * h;
    for (int i = 0; i < max_intersection_cnt; i++) {
        Ray r(start_pt, ray_dir, epsilon<float>, tmax);
        if (!ctx->accel->intersect(r, isects[i]) )
            break;
        if (isects[i].shape == ctx->isect_i->shape) {
            ++found_intersection;
            isect_indice.push_back(i);
        }
        start_pt = isects[i].P;
        tmax -= isects[i].ray_t;
        if (tmax < epsilon<float>)
            break;
    }

    // Randomly chose one intersection
    uint idx = 0;
    if (found_intersection == 0)
        return false;
    else if (found_intersection > 1) {
        idx = sampler->randomf() * found_intersection * 0.999999f;
    }

    idx = isect_indice[idx];
    bssrdf_sample->po = isects[idx].P;
    bssrdf_sample->sampled_shader = (*(ctx->engine_ptr->shaders))[isects[idx].shader_name];
    bssrdf_sample->sample_cnt = found_intersection;
    ctx->isect_o = isects[idx];

    return true;
}

static inline float sample_standard_dipole_func(void* sp, uint32_t ch, const float u) {
    auto sample = reinterpret_cast<BSSRDFSample*>(sp);
    return sample_exponential_distribution(sample->sigma_tr[ch], u);
}

static RGBSpectrum separable_bssrdf_sample(
    ShadingContext* ctx,
    const bssrdf_profile_sample_func& profile_sample_func,
    const eval_profile_func& profile_eval_func,
    Sampler* sampler)
{
    // TODO : look into the channel sampling code in as
    // and find why it relates to the spectrum
    auto found = find_po(ctx, profile_sample_func, sampler);
    if (!found)
        return 0;

    auto bssrdf_sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);
    auto original_data = ctx->data;

    OSL::ShaderGlobals sg;
    KazenRenderServices::globals_from_hit(sg, *(ctx->ray), ctx->isect_o);
    ctx->engine_ptr->execute(bssrdf_sample->sampled_shader, sg);
    ShadingResult ret;
    process_closure(ret, sg.Ci, 1, false);
    ret.surface.compute_pdfs(sg, 1, false);
    bssrdf_sample->sampled_closure = ret.surface;

    BSDFSample bsdf_sample;
    auto original_sample = ctx->closure_sample;
    ctx->closure_sample = &bsdf_sample;

    auto original_sg = ctx->sg;
    ctx->sg = &sg;

    bssrdf_sample->brdf_f = bssrdf_sample->sampled_closure.sample(ctx, sampler);
    bssrdf_sample->brdf_pdf = bsdf_sample.pdf;
    ctx->isect_o.wo = bsdf_sample.wo;

    ctx->data = original_data;
    ctx->closure_sample = original_sample;
    ctx->sg = original_sg;

    bssrdf_sample->pdf = separable_bssrdf_pdf(ctx, dipole_profile_pdf,
        base::length(ctx->isect_i->P - ctx->isect_o.P));

    return separable_bssrdf_eval(ctx, profile_eval_func, ctx->isect_i->P,
        ctx->isect_i->wi, ctx->isect_o.P, ctx->isect_o.wo);
}

void KpStandardDipole::precompute(ShadingContext* ctx) {
    auto params = reinterpret_cast<KpDipoleParams*>(ctx->data);
    ComputeRdStandardDipole compute_rd_standard_dipole(params->eta);
    // This code is a mess
    // TODO: Figure out how to remove one of the bind/lambda
    //auto compute_rd_func = std::bind(&ComputeRdStandardDipole::compute_rd, &compute_rd_standard_dipole,
        //std::placeholders::_1);
    auto compute_rd_func = [&](const float a) {
        return compute_rd_standard_dipole.compute_rd(a);
    };
    auto alpha_func = std::bind(compute_alpha_prime, compute_rd_func, std::placeholders::_1);
    dipole_precompute(ctx, alpha_func);
}

RGBSpectrum KpStandardDipole::eval(ShadingContext* ctx) {
    //return eval_dipole(ctx, eval_standard_dipole_func);
    return separable_bssrdf_eval(ctx, standard_dipole_profile_eval,
        ctx->isect_i->P, ctx->isect_i->wi, ctx->isect_o.P, ctx->isect_o.wo);
}

RGBSpectrum KpStandardDipole::sample(ShadingContext* ctx, Sampler* sampler) {
    //return sample_dipole(ctx, sample_standard_dipole_func,
        //eval_standard_dipole_func, rng);
    return separable_bssrdf_sample(ctx, sample_standard_dipole_func,
        standard_dipole_profile_eval, sampler);
}

void KpBetterDipole::precompute(ShadingContext* ctx) {
    auto params = reinterpret_cast<KpDipoleParams*>(ctx->data);
    ComputeRdBetterDipole compute_rd_better_dipole(params->eta);
    auto compute_rd_func = [&](const float a) {
        return compute_rd_better_dipole.compute_rd(a);
    };
    auto alpha_func = std::bind(compute_alpha_prime, compute_rd_func, std::placeholders::_1);
    dipole_precompute(ctx, alpha_func);
}

RGBSpectrum KpBetterDipole::eval(ShadingContext* ctx) {
    return separable_bssrdf_eval(ctx, better_dipole_profile_eval,
        ctx->isect_i->P, ctx->isect_i->wi, ctx->isect_o.P, ctx->isect_o.wo);
}

RGBSpectrum KpBetterDipole::sample(ShadingContext* ctx, Sampler* sampler) {
    return separable_bssrdf_sample(ctx, sample_standard_dipole_func,
        better_dipole_profile_eval, sampler);
}