#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>

#include <boost/math/constants/constants.hpp>

#include "vec.h"

using base::Vec3f;

namespace constants = boost::math::constants;
namespace fs = std::filesystem;

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

inline Vec3f world_to_local(const Vec3f& w, const Vec3f& n, const Vec3f& t, const Vec3f& b) {
    return Vec3f{dot(w, t), dot(w, n), dot(w, b)};
}

inline Vec3f local_to_world(const Vec3f& w, const Vec3f& n, const Vec3f& t, const Vec3f& b) {
    return Vec3f{
        t.x() * w.x() + n.x() * w.y() + b.x() * w.z(),
        t.y() * w.x() + n.y() * w.y() + b.y() * w.z(),
        t.z() * w.x() + n.z() * w.y() + b.z() * w.z()
    };
}

inline Vec3f local_to_world(const Vec3f& w, const Vec3f& n) {
    auto t = (fabsf(w.x()) > .01f ? Vec3f(w.z(), 0, -w.x()) : normalize(Vec3f(0, -w.z(), w.y())));
    auto b = cross(w, t);
    return local_to_world(w, n, t, b);
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
    float tmp = 1 - w[1] * w[1];
    if (tmp < 0.f)
        return 0.f;
    return std::sqrt(tmp) / w[1];
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

inline Vec3f reflect(const Vec3f& wi) {
    return Vec3f(-wi.x(), wi.y(), -wi.z());
}

inline Vec3f reflect(const Vec3f& wi, const Vec3f& n) {
    return -wi + 2 * dot(wi, n) * n;
}

inline Vec3f refract(const Vec3f& wi, const Vec3f& n, float eta) {
    float cos_theta_i = dot(n, wi);
    if (cos_theta_i < 0)
        eta = 1.f / eta;
    auto cos_theta_o2 = 1 - (1 - cos_theta_i * cos_theta_i) * (eta * eta);
    if (cos_theta_o2 <= 0.f)
        return Vec3f(0.f);
    auto sign = cos_theta_i >= 0.f ? 1.f : -1.f;
    return n * (-cos_theta_i * eta + sign * sqrt(cos_theta_o2)) + wi * eta;
}

inline float fresnel(float cos_theta_i, float ext_ior, float int_ior) {
    float etai = ext_ior, etat = int_ior;

    if (ext_ior == int_ior)
        return 0.f;

    if (cos_theta_i < 0.f) {
        std::swap(etai, etat);
        cos_theta_i = -cos_theta_i;
    }

    float eta = etai / etat,
        sin_theta_t2 = eta * eta * (1. - cos_theta_i * cos_theta_i);

    if (sin_theta_t2 > 1.f)
        return 1.f;

    float cos_theta_t = std::sqrt(1.f - sin_theta_t2);
    float Rs = (etai * cos_theta_i - etat * cos_theta_t)
        / (etai * cos_theta_i + etat * cos_theta_t);
    float Rp = (etat * cos_theta_i - etai * cos_theta_t)
        / (etat * cos_theta_i + etai * cos_theta_t);

    return (Rs * Rs + Rp * Rp) / 2.f;
}

inline bool same_hemisphere(const Vec3f& w, const Vec3f& n) {
    return w.y() * n.y() > 0.f;
}

struct Frame {
    Vec3f s, t, n;

    Frame() {}

    Frame(const Vec3f& s, const Vec3f& t, const Vec3f& n)
        : s(s), t(t), n(n) {}

    Frame(const Vec3f& n) : n(n) {
        if (std::abs(n[0]) > std::abs(n[1])) {
            float inv_len = 1.f / std::sqrt(n[0] * n[0] + n[2] * n[2]);
            t = Vec3f(n[2] * inv_len, 0.f, -n[0] * inv_len);
        }
        else {
            float inv_len = 1.f / std::sqrt(n[1] * n[1] + n[2] * n[2]);
            t = Vec3f(0.f, n[2] * inv_len, -n[1] * inv_len);
        }
        s = cross(t, n);
    }

    Vec3f to_local(const Vec3f& v) const {
        return Vec3f(dot(v, s), dot(v, n), dot(v, t));
    }

    Vec3f to_world(const Vec3f& v) const {
        return s * v[0] + n * v[1] + t * v[2];
    }
};

// File IO utilities
inline std::string load_file(const fs::path& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.good()) {
        std::cerr << "Failed open file : " << path << std::endl;
        return std::string();
    }

    size_t file_size = fs::file_size(path);
    std::string buffer(file_size, 0);
    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();

    return buffer;
}