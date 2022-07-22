#pragma once

#include <cmath>

#include "base/vec.h"

using base::Vec3f;

// simple version of fresnel function for now
inline float fresnel_dielectric(float cosi, float eta) {
    if (eta == 0)
        return 1;

    if (cosi < 0.f) eta = 1.f / eta;
    float c = fabsf(cosi);
    float g = eta * eta - 1 + c * c;
    if (g > 0) {
        g = sqrtf(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        return 0.5f * A * A * (1 + B * B);
    }
    return 1.f;
}

inline float fresnel_refraction(const Vec3f I, const Vec3f N, float eta, Vec3f& T) {
    // compute refracted direction and fresnel term
    // return value will be 0 if TIR occurs
    // NOTE: I is the incoming ray direction (points toward the surface, normalized)
    //       N is the surface normal (points toward the incoming ray origin, normalized)
    //       T is the outgoing refracted direction (points away from the surface)
    float cosi = -dot(I, N);
    // check which side of the surface we are on
    Vec3f Nn;
    float neta;
    if (cosi > 0) {
        neta = 1 / eta;
        Nn = N;
    }
    else {
        cosi = -cosi;
        neta = eta;
        Nn = -N;
    }

    float arg = 1.f - (neta * neta * (1.f - cosi * cosi));
    if (arg >= 0) {
        float dnp = sqrt(arg);
        float nK = (neta * cosi) - dnp;
        T = I * neta + Nn * nK;
        return 1 - fresnel_dielectric(cosi, eta);
    }

    T = Vec3f(0);
    return 0;
}

// Detailed fresnel functions
// This part of code is copied from appleseed with a little bit modification
// Checkout https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/foundation/math/fresnel.h
inline float fresnel_refl_dielectric_p(const float eta, const float cos_theta_i, const float cos_theta_t);

inline float fresnel_refl_dielectric_s(const float eta, const float cos_theta_i, const float cos_theta_t);

inline float fresnel_refl_dielectric(const float eta, const float cos_theta_i, const float cos_theta_t);

inline float fresnel_refl_dielectric(const float eta, const float cos_theta_i);

inline float fresnel_trans_dielectric(const float eta, const float cos_theta_i);

// Implementations
float fresnel_refl_dielectric_p(const float eta, const float cos_theta_i, const float cos_theta_t) {
    assert(cos_theta_i >= 0.f && cos_theta_i <= 1.f);
    assert(cos_theta_t >= 0.f && cos_theta_t <= 1.f);
    assert(cos_theta_i > 0.f || cos_theta_i > 0.f);

    float fac = eta;
    fac *= cos_theta_t;

    float denom = cos_theta_i;
    float ret = denom;
    ret -= fac;
    denom += fac;
    ret /= denom;
    ret *= ret;
    return ret;
}

float fresnel_refl_dielectric_s(const float eta, const float cos_theta_i, const float cos_theta_t) {
    assert(cos_theta_i >= 0.f && cos_theta_i <= 1.f);
    assert(cos_theta_t >= 0.f && cos_theta_t <= 1.f);
    assert(cos_theta_i > 0.f || cos_theta_t > 0.f);

    float fac = eta;
    fac *= cos_theta_i;

    float denom = cos_theta_t;
    float ret = denom;
    ret -= fac;
    denom += fac;
    ret /= denom;
    ret *= ret;
    return ret;
}

float fresnel_refl_dielectric(const float eta, const float cos_theta_i, const float cos_theta_t) {
    assert(cos_theta_i >= 0.f && cos_theta_i <= 1.f);
    assert(cos_theta_t >= 0.f && cos_theta_t <= 1.f);
    
    if (cos_theta_i == 0.f && cos_theta_t == 0.f)
        return 0;

    auto p = fresnel_refl_dielectric_p(eta, cos_theta_i, cos_theta_t);
    auto s = fresnel_refl_dielectric_s(eta, cos_theta_i, cos_theta_t);
    return 0.5f * (p + s);
}

float fresnel_refl_dielectric(const float eta, const float cos_theta_i) {
    const float sin_theta_i2 = 1.f - square(cos_theta_i);
    const float sin_theta_t2 = sin_theta_i2 * square(eta);
    const float cos_theta_t2 = 1.f - sin_theta_t2;

    if (cos_theta_t2 < 0.f) // Total internal reflection
        return 1.f;
    else {
        const float cos_theta_t = std::sqrt(cos_theta_t2);
        return fresnel_refl_dielectric(eta, cos_theta_i, cos_theta_t);
    }
}

float fresnel_trans_dielectric(const float eta, const float cos_theta_i) {
    const float sin_theta_i2 = 1.f - square(cos_theta_i);
    const float sin_theta_t2 = sin_theta_i2 * square(eta);
    const float cos_theta_t2 = 1.f - sin_theta_t2;

    if (cos_theta_t2 < 0.f)
        return 0.f;
    else {
        const float cos_theta_t = std::sqrt(cos_theta_t2);
        return 1 - fresnel_refl_dielectric(eta, cos_theta_i, cos_theta_t);
    }
}
