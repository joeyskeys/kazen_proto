#include <cmath>

#include "material.h"
#include "sampling.h"

float BxDF::pdf(const Vec3f& wo, const Vec3f& wi) const {
    // Different from pbrt, do not check same hemisphere here.
    //return abs_cos_theta(wi) * M_1_PI;

    // 1 / (2 * Pi)
    return  wi.y() * M_1_PI;
}

RGBSpectrum BxDF::sample_f(const Vec3f& wo, Vec3f& wi, const Intersection& isect, const Vec2f& u, float& p) const {
    wi = sample_hemisphere(u);
    //wi = normalize(Vec3f{0.f, 1.f, 0.f} + random3f());

    // sample hemisphere directly sample from the upper hemisphere
    //if (wo.y() < 0.f) wi.y() *= -1.f;

    p = pdf(wo, wi);
    return f(wo, wi, isect);
}

RGBSpectrum LambertianBxDF::f(const Vec3f& wo, const Vec3f& wi, const Intersection& isect) const {
    return color * M_1_PI;
}

RGBSpectrum MetalBxDF::f(const Vec3f& wo, const Vec3f& wi, const Intersection& isect) const {
    return color * 0.95;
}

float MetalBxDF::pdf(const Vec3f& wo, const Vec3f& wi) const {
    return 1;
}

RGBSpectrum MetalBxDF::sample_f(const Vec3f& wo, Vec3f& wi, const Intersection& isect, const Vec2f& u, float& p) const {
    wi = reflect(wo, Vec3f{0.f, 1.f, 0.f});
    p = 1.f;
    return f(wo, wi, isect);
}

RGBSpectrum Material::calculate_response(Intersection& isect, Ray& ray) const {
    Vec3f wo = world_to_tangent(-ray.direction, isect.N, isect.tangent, isect.bitangent);
    Vec3f wi;

    Vec2f u = random2f();
    float pdf;

    auto f = bxdf->sample_f(wo, wi, isect, u, pdf);

    // LLVM and GCC behave differently if wi.y() multiplied directly..
    f *= wi.y();

    auto spec = f / pdf;

    isect.wo = -ray.direction;
    isect.wi = tangent_to_world(wi, isect.N, isect.tangent, isect.bitangent);

    return spec;
}
