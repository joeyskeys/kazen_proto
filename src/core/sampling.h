#pragma once

#include <cmath>
#include <random>

#include <boost/math/constants/constants.hpp>

#include "base/vec.h"

namespace constants = boost::math::constants;

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
    float v_sin_theta2 = 1 - v_cos_theta * v_cos_theta;
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

inline Vec2f to_uniform_disk(const Vec2f& sample) {
    auto theta = sample.x() * constants::two_pi<float>();
    auto r = std::sqrt(sample.y());
    return Vec2f(r * std::cos(theta), r * std::sin(theta));
}

inline float to_uniform_disk_pdf(const Vec2f& sample) {
    auto r = Vec2f(sample.x(), sample.y()).length();
    if (r > 1)
        return 0;
    else
        return constants::one_div_pi<float>();
}

inline Vec3f to_uniform_sphere(const Vec2f& sample) {
    auto phi = sample.x() * constants::two_pi<float>();
    auto theta = std::acos(1. - sample.y());
    auto sin_theta_val = std::sin(theta);
    return Vec3f(sin_theta_val * std::cos(phi), sin_theta_val * std::sin(phi), std::cos(theta));
}

inline float to_uniform_sphere_pdf(const Vec3f& v) {
    return 0.25 * constants::one_div_pi<float>();
}

inline Vec3f to_uniform_hemisphere(const Vec2f& sample) {
    float phi = sample.x() * constants::two_pi<float>();
    float theta = std::acos(1. - sample.y());
    auto sin_theta_val = std::sin(theta);
    return Vec3f(sin_theta_val * std::cos(phi), sin_theta_val * std::sin(phi), std::cos(theta));
}

inline float to_uniform_hemisphere(const Vec3f& v) {
    return v.z() > 0 ? constants::half_pi<float>() : 0;
}

inline Vec3f to_cosine_hemisphere(const Vec2f& sample) {
    auto pt = to_uniform_disk(sample);
    float z = std::sqrt(1. - pt.length_squared());
    if (z == 0)
        z = 1e-10f;
    return Vec3f(pt.x(), pt.y(), z);
}

inline float to_cosine_hemisphere_pdf(const Vec3f& v) {
    return v.z() * constants::one_div_pi<float>();
}