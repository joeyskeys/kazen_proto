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