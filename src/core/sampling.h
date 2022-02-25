#pragma once

#include <cmath>
#include <random>

#include "base/vec.h"

inline float randomf() {
    static std::uniform_real_distribution<double> dist(0.f, 1.f);
    static std::mt19937 gen;
    return dist(gen);
}

inline auto random2f() {
    return Vec2f{randomf(), randomf()};
}

inline Vec3f sample_hemisphere(const Vec2f& u) {
    float theta = u.x() * M_PI_2;
    float phi = u.y() * M_PI * 2.f;
    float v_cos_theta = std::cos(theta);
    float v_sin_theta = std::sin(theta);

    return normalize(Vec3f{v_sin_theta * std::cos(phi), v_cos_theta, v_sin_theta * std::sin(phi)});
}

inline Vec3f sample_hemisphere() {
    return sample_hemisphere(random2f());
}

inline Vec3f sample_hemisphere_with_exponent(const Vec2f& u, const float exponent) {
    float phi = u.x() * M_PI * 2.f;
    float v_cos_theta = std::pow(u.y(), 1 / (exponent + 1));
    float v_sin_theta2 = 1 - cos_theta * cos_theta;
    float v_sin_theta = v_sin_theta2 > 0 ? sqrtf(v_sin_theta2) : 0;

    return normalize(Vec3f{v_cos_theta * std::cos(phi), v_cos_theta, v_sin_theta * std::sin(phi)});
}

inline Vec3f sample_hemisphere_with_exponent(const float exponent) {
    return sample_hemisphere_with_exponent(random2f(), exponent);
}

inline auto random3f() {
    return Vec3f{randomf(), randomf(), randomf()};
}

inline int randomi(int range) {
    static std::uniform_int_distribution<> dist(0, range);
    static std::mt19937 gen;
    return dist(gen);
}

inline float power_heuristic(int nf, float f_pdf, int ng, float g_pdf) {
    float f = nf * f_pdf, g = ng * g_pdf;
    return (f * f) / (f * f + g * g);
}