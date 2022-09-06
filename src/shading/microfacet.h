#pragma once

#include <cmath>

#include <boost/math/constants/constants.hpp>

#include "base/utils.h"
#include "base/vec.h"

using base::Vec2f;

namespace constants = boost::math::constants;

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
 *    iv. unstretch
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

static inline float stretched_roughness(
    const Vec3f& m,
    const float sin_theta_v,
    const float xalpha,
    const float yalpha)
{
    // Check "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
    // page 88, equation 87.
    // https://jcgt.org/published/0003/02/03/paper.pdf
    // Code here is mostly copied from appleseed:
    // https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/foundation/math/microfacet.cpp
    if (xalpha == yalpha)
        return 1.f / square(xalpha);
    const float cos_phi_2_ax_2 = square(m.x() / (sin_theta_v * xalpha));
    const float sin_phi_2_ay_2 = square(m.z() / (sin_theta_v * yalpha));
    return cos_phi_2_ax_2 + sin_phi_2_ay_2;
}

inline float projected_roughness(
    const Vec3f&    m,
    const float     sin_theta_v,
    const float     xalpha,
    const float     yalpha)
{
    // Check "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
    // page 86, equation 80, same document as above
    // Code here is mostly copied from appleseed, same as above
    if (xalpha == yalpha || sin_theta_v == 0.f)
        return xalpha;

    const float cos_phi_2_ax_2 = square((m.x() * xalpha) / sin_theta_v);
    const float sin_phi_2_ay_2 = square((m.z() * yalpha) / sin_theta_v);
    return std::sqrt(cos_phi_2_ax_2 + sin_phi_2_ay_2);
}

struct GGXDist {
    static float D(const float tan2m_a) {
        auto tmp = 1 + tan2m_a;
        return 1.f / (constants::pi<float>() * tmp * tmp);
    }

    static inline float lambda(const float a_rcp) {
        const float a2_rcp = square(a_rcp);
        return (-1.f + std::sqrt(1.f + a2_rcp)) * 0.5f;
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
    static float D(const float tan2m_a) {
        return constants::one_div_pi<float>() * std::exp(-tan2m_a);
    }

    static inline float lambda(const float a_rcp) {
        const float a = 1.f / a_rcp;
        if (a < 1.6f) {
            const float a2 = square(a);
            return (1.f - 1.259f * a + 0.396f * a2) / (3.535f * a + 2.181f * a2);
        }
        return 0.f;
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

template <typename MDF>
class MicrofacetInterface {
public:
    MicrofacetInterface(const Vec3f& i, const float xa, const float ya)
        : wi(i)
        , xalpha(xa)
        , yalpha(ya)
    {}

    // The design of returning pdf rather than sampled direction is more convinient
    // for error checking
    Vec3f sample_m(const Vec3f& rand) const {
        // 1. stretch wi
        Vec3f stretched{wi[0] * xalpha, wi[1], wi[2] * yalpha};
        auto cos_theta_i = cos_theta(wi);
        if (cos_theta_i < 0.f)
            stretched = -stretched;
        // normalize
        stretched = base::normalize(stretched);
        // get polar coordinates
        float theta = 0.f, phi = 0.f;
        if (cos_theta_i < 0.9999f) {
            theta = acos(cos_theta_i);
            phi = atan2(wi[2], wi[0]);
        }

        // 2. sample slope
        Vec2f slope = MDF::sample_slope(cos_theta_i, base::head<2>(rand));

        // 3. rotate
        const float cos_phi_v = cos(phi);
        const float sin_phi_v = sin(phi);
        slope = Vec2f{
            cos_phi_v * slope[0] - sin_phi_v * slope[1],
            sin_phi_v * slope[0] + cos_phi_v * slope[1]
        };

        // 4. unstretch and normalize
        const Vec3f m{-slope[0] * xalpha, 1.f, -slope[1] * yalpha};
        return base::normalize(m);
    }

    float pdf(const Vec3f& m) const {
        // Check "Importance Sampling Microfacet-Based BSDFs using the Distribution
        // of Visible Normals" page 4 equation 2.
        // https://hal.inria.fr/hal-00996995v2/document
        // Code here is mostly copied from appleseed:
        // https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/foundation/math/microfacet.cpp
        // Here we use the exact visible normal distribution function as the pdf
        // function.
        const float cos_theta_v = cos_theta(wi);
        if (cos_theta_v == 0.f)
            return 0.f;

        return G1(wi, xalpha, yalpha) * std::abs(base::dot(wi, m)) *
            D(m, xalpha, yalpha) / std::abs(cos_theta_v);
    }

    inline float lambda(const Vec3f& m) const {
        float cos_theta_v = cos_theta(m);
        if (cos_theta_v == 0.f)
            return 0.f;

        const float sin_theta_v = std::sqrt(std::max(0.f, 1.f - square(cos_theta_v)));
        if (sin_theta_v == 0.f)
            return 0.f;

        const float alpha = projected_roughness(m, sin_theta_v, xalpha, yalpha);
        const float tan_theta_v = std::abs(sin_theta_v / cos_theta_v);
        const float a_rcp = alpha * tan_theta_v;

        return MDF::lambda(a_rcp);
    }

    virtual float G(const Vec3f& wo) const {
        return 1.f / (lambda(wi, xalpha, yalpha) + lambda(wo, xalpha, yalpha) + 1.f);
    }

    inline float G1(const Vec3f& w) const {
        return 1.f / (lambda(w, xalpha, yalpha) + 1.f);
    }

    inline float D(const Vec3f& m) const {
        const float cos_theta_v = cos_theta(m);
        if (cos_theta_v == 0.f) {
            if constexpr (std::is_same_v<MDF, BeckmannDist>)
                return 0.f;
            else // currently only two possible distributions: beckmann & trowbridge
                return square(xalpha) * constants::one_div_pi<float>();
        }
        
        const float cos_theta_2 = square(cos_theta_v);
        const float sin_theta_v = std::sqrt(std::max(0.f, 1.f - cos_theta_2));
        const float cos_theta_4 = square(cos_theta_2);
        const float tan_theta_2 = (1.f - cos_theta_2) / cos_theta_2;

        const float A = stretched_roughness(m, sin_theta_v, xalpha, yalpha);

        // Code structure here takes reference from "Understanding the Masking-Shadowing
        // Function in Microfacet-Based BRDFs" page 88 equation 87
        return MDF::D(tan_theta_2 * A) / (cos_theta_4 * xalpha * yalpha);
    }

public:
    Vec3f wi;
    float xalpha, yalpha;
};

class DisneyMicrofacetInterface : public MicrofacetInterface<GGXDist> {
public:
    float G(const Vec3f& wo) const override {
        return G1(wo) * G1(wi);
    }
};
