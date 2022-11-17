#pragma once

#include <OSL/oslclosure.h>

#include "shading/bssrdfs.h"

using OSL::TypeDesc;

/* BSSRDF is new subject, starting with pbrt and appleseed references:
 * https://www.pbr-book.org/3ed-2018/Volume_Scattering/The_BSSRDF
 * http://library.imageworks.com/pdfs/imageworks-library-BSSRDF-sampling.pdf
 * https://www.solidangle.com/research/s2013_bssrdf_slides.pdf
 */

struct DipoleParams {
    OSL::Vec3 N;
    float max_radius, eta;
};

// Diffusion approximation of subsurface reflection
// Checkout "A Practical Model for Subsurface Light Transport" page 3 formular 5
// https://graphics.stanford.edu/papers/bssrdf/bssrdf.pdf
static RGBSpectrum Sd(const Vec3f& pi, const Vec3f& wi, const Vec3f& po,
    const Vec3f& wo, const void* data,
    std::function<RGBSpectrum(const Vec3f&, const Vec3f&,
        const Vec3f&, const Vec3f&, void*)>& eval_profile_func)
{
    auto params = reinterpret_cast<DipoleParams*>(data);
    auto Rd = eval_profile_func(pi, wi, po, wo, data);
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
