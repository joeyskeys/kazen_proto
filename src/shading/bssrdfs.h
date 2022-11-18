#pragma once

#include <boost/math/constants/constants.hpp>
#include <OSL/oslclosure.h>

#include "shading/bssrdfs.h"
#include "shading/fresnel.h"

using OSL::TypeDesc;

namespace constants = boost::math::constants;

/* BSSRDF is new subject, starting with pbrt and appleseed references:
 * https://www.pbr-book.org/3ed-2018/Volume_Scattering/The_BSSRDF
 * http://library.imageworks.com/pdfs/imageworks-library-BSSRDF-sampling.pdf
 * https://www.solidangle.com/research/s2013_bssrdf_slides.pdf
 */

struct DipoleParams {
    OSL::Vec3 N;
    OSL::Vec3 sigma_a;
    OSL::Vec3 sigma_s;
    OSL::Vec3 sigma_tr;
    float max_radius, eta, g;
};

// Diffusion approximation of subsurface reflection
// Checkout "A Practical Model for Subsurface Light Transport" page 3 formular 5
// https://graphics.stanford.edu/papers/bssrdf/bssrdf.pdf
static RGBSpectrum Sd(const Vec3f& pi, const Vec3f& wi, const Vec3f& po,
    const Vec3f& wo, const DipoleParams* params,
    std::function<RGBSpectrum(const Vec3f&, const Vec3f&,
        const Vec3f&, const Vec3f&, const DipoleParams*)>& eval_profile_func)
{
    auto Rd = eval_profile_func(pi, wi, po, wo, params);
    auto n = base::to_vec3(params->N);
    auto cos_ni = std::abs(base::dot(n, wi));
    auto fi = fresnel_trans_dielectric(params->eta, cos_ni);
    auto cos_no = std::abs(base::dot(n, wo));
    auto fo = fresnel_trans_dielectric(params->eta, cos_no);
    auto c = 1.f - fresnel_first_moment_x2(params->eta);

    // The equation used is from "A better dipole" page 2 equation 1
    // http://www.eugenedeon.com/wp-content/uploads/2014/04/betterdipole.pdf
    // Acorrding to the paper is contains the normalization factor(the / c).
    // Fresnel approximation calculation is different, original one is not 
    // tested and the implementation here is mostly copied from appleseed
    return Rd * fi * fo / c;
}

static RGBSpectrum standard_dipole_eval_profile(const Vec3f& pi, const Vec3f& wi,
    const Vec3f& po, const Vec3f& wo, const DipoleParams* params)
{
    const float sqr_radius = base::length_squared(pi - po);

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
    auto zv = -zr * (1.f + (4.f / 3.f) * a);

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