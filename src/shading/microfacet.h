#pragma once

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
 ***********************************************************/

struct GGXDist {
    static float F(const float m2) {

    }

    static float lambda(const float a2) {

    }

    static Vec2f sample_slope(const float cos_theta, const Vec2f sample) {

    }
};

struct BeckmannDist {
    static float F(const float m2) {

    }

    static float lambda(const float a2) {

    }

    static Vec2f sample_slope(const float cos_theta, const Vec2f sample) {

    }
};