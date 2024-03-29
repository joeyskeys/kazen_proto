#pragma once

#include <cmath>

#include "base/vec.h"
#include "base/utils.h"
#include "core/spectrum.h"

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

inline float fresnel_first_moment_x2(const float eta) {
    return
        eta < 1.f
            ? 0.919317 + eta * (-3.4793 + eta * (6.75335 + eta * (-7.80989 + eta * (4.98554 - eta * 1.36881))))
            : -9.23372 + eta * (22.2272 + eta * (-20.9292 + eta * (10.2291 + eta * (-2.54396 + eta * 0.254913))));
}

inline float fresnel_second_moment_x3(const float eta) {
    const float rcp_eta = 1.f / eta;
    return
        eta < 1.f
            ? 0.828421 + eta * (-2.62051 + eta * (3.36231 + eta * (-1.95284 + eta * (0.236494 + eta * 0.145787))))
            : -1641.1 + (((135.926 * rcp_eta) - 656.175) * rcp_eta + 1376.53) * rcp_eta
              + eta * (1213.67 + eta * (-568.556 + eta * (164.798 + eta * (-27.0181 + eta * 1.91826)))); 
}

inline float fresnel_internel_diffuse_reflectance(const float eta) {
    const float rcp_eta = 1. / eta;
    const float rcp_eta2 = square(rcp_eta);
    // The eta > 1 part is same as the equation presented in the paper
    // "A Practical Model for Subsurface Light Transport" page 3 equation for
    // Fdr
    // This implementation is copied from appleseed and dunno where the
    // eta < 1 part is from
    return
        eta < 1
            ? -0.4399 + 0.7099 * rcp_eta - 0.3319 * rcp_eta2 + 0.0636 * rcp_eta * rcp_eta2
            : -1.4399 * rcp_eta2 + 0.7099 * rcp_eta + 0.6681 + 0.0636 * eta;
}

inline float schlick_weight(float coso) {
    return std::pow(std::clamp(1. - coso, 0., 1.), 5.);
}

inline float fresnel_schlick(const float eta, const float cos_theta_v) {
    auto F0 = square((1. - eta) / (1. + eta));
    return F0 + (1. - F0) * pow(1. - cos_theta_v, 5.);
}

inline RGBSpectrum fresnel_schlick(const RGBSpectrum F0, const float coso) {
    return base::lerp(F0, RGBSpectrum{1}, RGBSpectrum{schlick_weight(coso)});
}
