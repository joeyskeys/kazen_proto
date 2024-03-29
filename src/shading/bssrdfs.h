#pragma once

#include <functional>

#include <boost/math/constants/constants.hpp>
#include <OSL/genclosure.h>

#include "core/sampler.h"
#include "shading/bsdf.h"
#include "shading/bssrdfs.h"
#include "shading/context.h"
#include "shading/fresnel.h"

using OSL::TypeDesc;

namespace constants = boost::math::constants;

/* BSSRDF is a new subject, starting with pbrt and appleseed references:
 * https://www.pbr-book.org/3ed-2018/Volume_Scattering/The_BSSRDF
 * http://library.imageworks.com/pdfs/imageworks-library-BSSRDF-sampling.pdf
 * https://www.solidangle.com/research/s2013_bssrdf_slides.pdf
 */

struct KpDipoleParams {
    OSL::Vec3 N;
    OSL::Vec3 Rd;
    OSL::Vec3 mfp;
    float max_radius, eta, g;
};

// Diffusion approximation of subsurface reflection
// Checkout "A Practical Model for Subsurface Light Transport" page 3 formular 5
// https://graphics.stanford.edu/papers/bssrdf/bssrdf.pdf
static RGBSpectrum Sd(const Vec3f& pi, const Vec3f& wi, const Vec3f& po,
    const Vec3f& wo, const KpDipoleParams* params,
    std::function<RGBSpectrum(const Vec3f&, const Vec3f&,
        const Vec3f&, const Vec3f&, const KpDipoleParams*)>& eval_profile_func)
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

struct KpStandardDipole {
    static void precompute(ShadingContext*);
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(KpDipoleParams, N),
            CLOSURE_VECTOR_PARAM(KpDipoleParams, Rd),
            CLOSURE_VECTOR_PARAM(KpDipoleParams, mfp),
            CLOSURE_FLOAT_PARAM(KpDipoleParams, max_radius),
            CLOSURE_FLOAT_PARAM(KpDipoleParams, eta),
            CLOSURE_FLOAT_PARAM(KpDipoleParams, g),
            CLOSURE_FINISH_PARAM(KpDipoleParams)
        };

        shadingsys.register_closure("kp_standard_dipole", KpStandardDipoleID, params, nullptr, nullptr);
    }
};

struct KpBetterDipole {
    static void precompute(ShadingContext*);
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(KpDipoleParams, N),
            CLOSURE_VECTOR_PARAM(KpDipoleParams, Rd),
            CLOSURE_VECTOR_PARAM(KpDipoleParams, mfp),
            CLOSURE_FLOAT_PARAM(KpDipoleParams, max_radius),
            CLOSURE_FLOAT_PARAM(KpDipoleParams, eta),
            CLOSURE_FLOAT_PARAM(KpDipoleParams, g),
            CLOSURE_FINISH_PARAM(KpDipoleParams)
        };

        shadingsys.register_closure("kp_better_dipole", KpBetterDipoleID, params, nullptr, nullptr);
    }
};

// This function type is used as a profile sampling function prototype, called
// in bssrdf sample function
using bssrdf_profile_eval_func = std::function<RGBSpectrum(void*, const Vec3f&,
    const Vec3f&, const Vec3f&, const Vec3f&)>;
using bssrdf_profile_sample_func = std::function<float(void*, uint32_t, const float)>;
using bssrdf_profile_pdf_func = std::function<float(ShadingContext*, const float)>;

using bssrdf_precompute_func = std::function<void(ShadingContext*)>;
using bssrdf_eval_func = std::function<RGBSpectrum(ShadingContext*)>;
using bssrdf_sample_func = std::function<RGBSpectrum(ShadingContext*, Sampler*)>;

inline bssrdf_precompute_func get_bssrdf_precompute_func(ClosureID id) {
    static std::array<bssrdf_precompute_func, 2> precompute_functions {
        KpStandardDipole::precompute,
        KpBetterDipole::precompute
    };
    return precompute_functions[id - KpStandardDipoleID];
}

inline bssrdf_eval_func get_bssrdf_eval_func(ClosureID id) {
    static std::array<bssrdf_eval_func, 2> eval_functions {
        KpStandardDipole::eval,
        KpBetterDipole::eval
    };
    return eval_functions[id - KpStandardDipoleID];
}

inline bssrdf_sample_func get_bssrdf_sample_func(ClosureID id) {
    static std::array<bssrdf_sample_func, 2> sample_functions {
        KpStandardDipole::sample,
        KpBetterDipole::sample
    };
    return sample_functions[id - KpStandardDipoleID];
}