#include <cmath>

#include "material.h"
#include "sampling.h"

float BxDF::pdf(const Vec3f& wo, const Vec3f& wi) const {
    // Different from pbrt, do not check same hemisphere here.
    return abs_cos_theta(wi) * M_1_PI;
}

RGBSpectrum BxDF::sample_f(const Vec3f& wo, Vec3f& wi, const Intersection& isect, const Vec2f& u, float& p) const {
    wi = sample_hemisphere(u);
    //wi = normalize(Vec3f{0.f, 1.f, 0.f} + random3f());
    if (wo.y() < 0.f) wi.y() *= -1.f;
    p = pdf(wo, wi);
    return f(wo, wi, isect);
}

RGBSpectrum LambertianBxDF::f(const Vec3f& wo, const Vec3f& wi, const Intersection& isect) const {
    //return color * M_1_PI;
    return color;
}

RGBSpectrum MetalBxDF::f(const Vec3f& wo, const Vec3f& wi, const Intersection& isect) const {
    return color * 0.95;
}

RGBSpectrum MetalBxDF::sample_f(const Vec3f& wo, Vec3f& wi, const Intersection& isect, const Vec2f& u, float& p) const {
    wi = reflect(wo, Vec3f{0.f, 1.f, 0.f});
    return f(wo, wi, isect);
}

RGBSpectrum Material::calculate_response(Intersection& isect, Ray& ray) const {
    Vec3f wo = world_to_tangent(-ray.direction, isect.normal, isect.tangent, isect.bitangent);
    Vec3f wi;
    RGBSpectrum spec;

    Vec2f u = random2f();
    float pdf = 0.f;
    float p = 1.f;

    auto f = bxdf->sample_f(wo, wi, isect, u, pdf);
    spec = f / pdf;

    isect.wo = -ray.direction;
    isect.wi = tangent_to_world(wi, isect.normal, isect.tangent, isect.bitangent);

    return f;
}
