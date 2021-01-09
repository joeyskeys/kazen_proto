
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

RGBSpectrum Material::calculate_response(Intersection& isect, const Ray& ray) const {
    Vec3f wo = -ray.direction;
    Vec3f wi;
    RGBSpectrum spec;

    Vec2f u = random2f();
    float pdf = 0.f;
    float p = 1.f;

    spec = bxdf->sample_f(wo, wi, u, pdf);

    return spec;
}
