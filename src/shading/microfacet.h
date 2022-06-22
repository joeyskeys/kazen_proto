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