#pragma once

#include <cmath>

#include <boost/math/constants/constants.hpp>

#include "base/utils.h"
#include "base/vec.h"

using base::Vec2f;

namespace constants = boost::math::constants;

/*
inline float stretch_roughness(
    const Vec3f&    m,
    const float     sin_theta_v,
    const float     ax,
    const float     ay)
{
    if (ax == ay || sin_theta_v == 0.f)
        return 1.f / square(ax);

    const float cos_phi_2_ax_2 = square(m.x() / (sin_theta_v * ax));
    const float sin_phi_2_ay_2 = square(m.z() / (sin_theta_v * ay));
    return cos_phi_2_ax_2 + sin_phi_2_ay_2;
}

float beckmann_ndf(
    const Vec3f&    m,
    const float     ax,
    const float     ay)
{
    const float cos_theta_v = m.y();
    if (cos_theta_v == 0.f)
        return 0.f;
    
    const float cos_theta_2 = square(cos_theta_v);
    const float sin_theta_v = std::sqrt(std::max(0.f, 1.f - cos_theta_2));
    const float cos_theta_4 = square(cos_theta_2);
    const float tan_theta_2 = (1.f - cos_theta_2) / cos_theta_2;

    const float A = stretch_roughness(m, sin_theta_v, ax, ay);

    return std::exp(-tan_theta_2 * A) / (constants::pi<float>() * ax * ay * cos_theta_4);
}

float GGX_ndf(
    const Vec3f&    m,
    const float     ax,
    const float     ay)
{
    const float cos_theta_v = m.y();
    if (cos_theta_v == 0.f)
        return square(ax) * constants::one_div_pi<float>();

    const float cos_theta_2 = square(cos_theta_v);
    const float sin_theta_v = std::sqrt(std::max(0.f, 1.f - cos_theta_2));
    const float cos_theta_4 = square(cos_theta_2);
    const float tan_theta_2 = (1.f - cos_theta_2) / cos_theta_2;

    const float A = stretch_roughness(m, sin_theta_v, ax, ay);

    const float tmp = 1.f + tan_theta_2 * A;
    return 1.f / (constants::pi<float>() * ax * ay * cos_theta_4 * square(tmp));
}
*/

Vec3f BeckmannMDF(const Vec2f& sample, float alpha);

float BeckmannPDF(const Vec3f& m, float alpha);

float G1(const Vec3f& wh, const Vec3f& wv, float alpha);

/***********************************************************
 * Some understanding about microfacet related stuffs:
 * 
 * 1. Distruction function used to sample the direction of 
 *    microfacet normal. Two major distribution functions
 *    are Beckmann distribution and Trowbridge-Reitz distribution
 *    (a.k.a GGX, checkout
 *    https://pharr.org/matt/blog/2022/05/06/trowbridge-reitz).
 *    Both of these two model came from physics related fields and
 *    then applied in CG(GGX was a unintentional reinvention).
 * 
 * 2. Shadowing & masking function is decided by the nature of
 *    microfacet. Two major microfacet models are v-cavity mmodel
 *    and smith model. Currently the popular implementation of 4
 *    steps:
 *    i. stretch
 *    ii. sample p22
 *    iii. rotate
 *    iiii. unstretch
 *    v. compute normal
 *    is for smith model, which seems more popular among the community.
 *    checkout https://hal.inria.fr/hal-00996995v2/document
 * 
 * 3. G1 stands for masking function, and the G term in the microfacet
 *    formula stands for shadowing-masking function. If we're using smith
 *    model then shadowing masking are non-correlated and G = G1(wi) * G1(wo).
 * 
 * 4. Lambda function is an auxiliary function which measures invisible
 *    masked microfacet area per visible microfacet area(from
 *    https://pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models).
 *    Shadowing-masking functions are expressed in terms of it. In the
 *    implementation: G1 = 1 / (1 + lambda(w))
 * 
 * 5. http://hal.inria.fr/hal-00996995/en
 *    https://hal.inria.fr/hal-00996995v2/file/supplemental1.pdf
 *    https://hal.inria.fr/hal-00996995v2/file/supplemental2.pdf are must
 *    read!
 * 
 ***********************************************************/

struct GGXDist {
    static float D(const float tan2m) {
        // Impl here are copied from OpenShadingLanguage testrender,
        // identical to the impl in pbrt-v3.
        // Appleseed have a extra stretched_roughness in it, need to
        // do some further investigate.
        auto tmp = 1 + tan2m;
        return 1.f / (constants::pi<float>() * tmp * tmp);
    }

    static inline float lambda(const float a2) {
        return 0.5f * (-1.f + sqrtf(1.f + 1.f / a2));
    }

    static Vec2f sample_slope(const float cos_theta, const Vec2f& sample) {
        Vec2f slope;

        // sample slope x
        float c = cos_theta < 1e-6f ? 1e-6f : cos_theta;
        float Q = (1 + c) * sample[0] - c;
        float num = c * sqrtf((1 - c) * (1 + c)) - Q * sqrtf((1 - Q) * (1 + Q));
        float den = (Q - c) * (Q + c);
        float eps = 1.f / 4294967296.0f;
        den = fabsf(den) < eps ? copysignf(eps, den) : den;
        slope[0] = num / den;

        // sample slope y
        float Ru = 1 - 2 * sample[1];
        float u2 = fabsf(Ru);
        float z = (u2 * (u2 * (u2 * 0.27385f - 0.73369f) + 0.46341f)) /
                  (u2 * (u2 * (u2 * 0.093073f + 0.309420f) - 1.0f) + 0.597999f);
        slope[1] = copysignf(1.0f, Ru) * z * sqrtf(1.0f + slope.x() * slope.x());

        return slope;
    }
};

struct BeckmannDist {
    static float D(const float tan2m) {
        return 1.f / constants::pi<float>() * std::exp(-tan2m);
    }

    static inline float lambda(const float a2) {
        const float a = sqrtf(a2);
        return a < 1.6f ? (1.f - 1.259f * a + 0.396f * a2) / (3.535f * a + 2.181f * a2) : 0.f;
    }

    static Vec2f sample_slope(const float cos_theta_i, const Vec2f& sample) {
        // Impl here are copied from OpenShadingLanguage testrender,
        float ct = cos_theta_i < 1e-6f? 1e-6f : cos_theta_i;
        float tan_theta_i = sqrtf(1 - ct * ct) / ct;
        float cot_theta_i = 1 / tan_theta_i;

        // sample slope x
        // compute a coarse approximation using the approximation:
        // exp(-ierf(x)^2) ~= 1 - x * x
        // solve y = 1 + b + K * (1 - b * b)
        float c = std::erf(cot_theta_i);
        float K = tan_theta_i * constants::one_div_root_pi<float>();
        float y_approx = sample[0] * (1.f + c + K * (1 - c * c));
        float y_exact = sample[0] * (1.f + c + K * std::exp(-cot_theta_i * cot_theta_i));
        float b = K > 0 ? (0.5f - sqrtf(K * (K - y_approx + 1.f) + 0.25f)) / K : y_approx - 1.f;

        // perform newton step to refine toward the true root
        float inv_erf = 1.f / std::erf(b);
        float value = 1.f + b + K * std::exp(-inv_erf * inv_erf) - y_exact;

        // check if we are close enough already
        // this also avoids NaNs as we get close to the root
        Vec2f slope;
        if (fabsf(value) > 1e-6f) {
            b -= value / (1 - inv_erf * tan_theta_i); // newton step 1
            inv_erf = 1.f / std::erf(b);
            value  = 1.0f + b + K * OIIO::fast_exp(-inv_erf * inv_erf) - y_exact;
            b -= value / (1 - inv_erf * tan_theta_i); // newton step 2
            // compute the slope from the refined value
            slope[0] = 1.f / std::erf(b);
        } else {
            // we are close enough already
            slope[0] = inv_erf;
        }

        /* sample slope Y */
        slope[1] = 1.f / std::erf(2.0f * sample[1] - 1.0f);

        return slope;
    }
};