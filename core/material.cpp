
#include "material.h"
#include "sampling.h"

#include <cmath>

float BxDF::pdf(const Vec3f& wo, const Vec3f& wi) const {
    // Different from pbrt, do not check same hemisphere here.
    return abs_cos_theta(wi) * M_1_PI;
}

RGBSpectrum BxDF::sample_f(const Vec3f& wo, Vec3f& wi, const Vec2f& u, float& p) const {
    wi = sample_hemisphere(u);
    if (wo.y() < 0.f) wi.z() *= -1.f;
    return f(wo, wi);
}

RGBSpectrum LambertianBxDF::f(const Vec3f& wo, const Vec3f& wi) const {
    return color * M_1_PI;
}

RGBSpectrum 