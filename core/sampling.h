#pragma once

#include <cmath>

#include "vec.h"

Vec3f sample_hemisphere(const Vec2f& u) {
    float theta = u.x * M_PI_2;
    float phi = u.y * M_PI * 2.f;
    float v_cos_theta = std::cos(theta);
    float v_sin_theta = std::sin(theta);

    return Vec3f{v_sin_theta * std::cos(phi), v_cos_theta, v_sin_theta * std::sin(phi)};
}