#pragma once

#include <algorithm>

#include <boost/math/constants/constants.hpp>

#include "vec.h"

namespace constants = boost::math::constants;

template <typename T>
T to_radian(const T degree) {
    return static_cast<T>(degree / 180. * constants::pi<float>());
}

template <typename T>
T to_degree(const T radian) {
    return static_cast<T>(radian / constants::pi<float>() * 180.);
}

template <typename T>
inline T square(const T& a) {
    return a * a;
}

/* Shading related utilities.
 *
 * Kazen uses a right-handed coordinate system, all the calculations follow
 * the same convension, calculated in the tangent space.
 *
 *                         | y
 *                         |
 *                         |
 *                         |
 *                         |
 *                         |
 *                         |
 *                         |___________________ x
 *                        /
 *                       /
 *                      /
 *                     /
 *                    /
 *                   /
 *                  / z
 *
 * params:
 *  w     : The vector to be calculated in tangent space, usually wi or wo.
 *
 *  theta : The angle between the vector and the normal, normal is the
 *          unit vector alone y axis in the tangent space.
 *
 *  phi   : The angle between xz plane projected vector from the original
 *          one and x axis;
 *
 * rules:
 *  1. To avoid ambiguous, all the variables start with a v.
 */

inline Vec3f world_to_tangent(const Vec3f& w, const Vec3f& n, const Vec3f& t, const Vec3f& b) {
    return Vec3f{dot(w, t), dot(w, n), dot(w, b)};
}

inline Vec3f tangent_to_world(const Vec3f& w, const Vec3f& n, const Vec3f& t, const Vec3f& b) {
    return Vec3f{
        t.x() * w.x() + n.x() * w.y() + b.x() * w.z(),
        t.y() * w.x() + n.y() * w.y() + b.y() * w.z(),
        t.z() * w.x() + n.z() * w.y() + b.z() * w.z()
    };
}

inline Vec3f tangent_to_world(const Vec3f& w, const Vec3f& n) {
    auto t = (fabsf(w.x()) > .01f ? Vec3f(w.z(), 0, -w.x()) : Vec3f(0, -w.z(), w.y())).normalized();
    auto b = cross(w, t);
    return tangent_to_world(w, n, t, b);
}

inline float cos_theta(const Vec3f& w) {
    return w.y();
}

inline float cos_2_theta(const Vec3f& w) {
    return w.y() * w.y();
}

inline float abs_cos_theta(const Vec3f& w) {
    return std::abs(w.y());
}

inline float sin_2_theta(const Vec3f& w) {
    return std::max(0.f, 1.f - cos_2_theta(w));
}

inline float sin_theta(const Vec3f& w) {
    return std::sqrt(sin_2_theta(w));
}

inline float tan_theta(const Vec3f& w) {
    return sin_theta(w) / cos_theta(w);
}

inline float tan_2_theta(const Vec3f& w) {
    return sin_2_theta(w) / cos_2_theta(w);
}

inline float cos_phi(const Vec3f& w) {
    float v_sin_theta = sin_theta(w);
    return (v_sin_theta == 0.f) ? 1.f : std::clamp(w.x() / v_sin_theta, -1.f, 1.f);
}

inline float sin_phi(const Vec3f& w) {
    float v_sin_theta = sin_theta(w);
    return (v_sin_theta == 0.f) ? 0 : std::clamp(w.z() / v_sin_theta, -1.f, 1.f);
}

inline float cos_2_phi(const Vec3f& w) {
    auto v_cos_phi = cos_phi(w);
    return v_cos_phi * v_cos_phi;
}

inline float sin_2_phi(const Vec3f& w) {
    auto v_sin_phi = sin_phi(w);
    return v_sin_phi * v_sin_phi;
}

inline Vec3f reflect(const Vec3f& wo, const Vec3f& n) {
    return -wo + 2 * dot(wo, n) * n;
}

inline bool refract(const Vec3f& wi, const Vec3f& n, const float eta, Vec3f& wt) {
    float v_cos_theta_i = dot(n, wi);
    float v_sin_2_theta_i = std::max(0.f, 1.f - v_cos_theta_i * v_cos_theta_i);
    float v_sin_2_theta_t = eta * eta * v_sin_2_theta_i;

    if (v_sin_2_theta_t >= 1.f) return false;
    float v_cos_theta_t = std::sqrt(1.f - v_sin_2_theta_t);
    wt = eta * -wi + (eta * v_cos_theta_i - v_cos_theta_t) * n;
    return true;
}

inline bool same_hemisphere(const Vec3f& w, const Vec3f& n) {
    return w.y() * n.y() > 0.f;
}
