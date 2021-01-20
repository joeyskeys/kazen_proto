#pragma once

#include <cmath>
#include <random>

#include "base/vec.h"

inline Vec3f sample_hemisphere(const Vec2f& u) {
    float theta = u.x() * M_PI_2;
    float phi = u.y() * M_PI * 2.f;
    float v_cos_theta = std::cos(theta);
    float v_sin_theta = std::sin(theta);

    return normalize(Vec3f{v_sin_theta * std::cos(phi), v_cos_theta, v_sin_theta * std::sin(phi)});
}

inline float randomf() {
    static std::uniform_real_distribution<double> dist(0.f, 1.f);
    static std::mt19937 gen;
    return dist(gen);
}

inline auto random2f() {
    return Vec2f{randomf(), randomf()};
}

inline auto random3f() {
    return Vec3f{randomf(), randomf(), randomf()};
}