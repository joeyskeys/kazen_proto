
#include "material.h"
#include "sampling.h"

#include <cmath>

float BxDF::pdf(const Vec3f& wo, const Vec3f& wi) const {
    // Different from pbrt, do not check same hemisphere here.
    return abs_cos_theta(wi) * M_1_PI;
}

RGBSpectrum BxDF::sample_f(const Vec3f& wo, Vec3f& wi, const Vec2f& u, float& p) const {
    wi = sample_hemisphere(u);
    //wi = normalize(Vec3f{0.f, 1.f, 0.f} + random3f());
    if (wo.y() < 0.f) wi.y() *= -1.f;
    p = pdf(wo, wi);
    return f(wo, wi);
}

RGBSpectrum LambertianBxDF::f(const Vec3f& wo, const Vec3f& wi) const {
    //return color * M_1_PI;
    return color;
}

RGBSpectrum Material::calculate_response(Intersection& isect, Ray& ray) const {
    Vec3f wo = world_to_tangent(-ray.direction, isect.normal, isect.tangent, isect.bitangent);
    Vec3f wi;
    RGBSpectrum spec;

    Vec2f u = random2f();
    float pdf = 0.f;
    float p = 1.f;

    auto f = bxdf->sample_f(wo, wi, u, pdf);
    spec = f / pdf;

    std::cout << "backface : " << isect.backface << std::endl;
    std::cout << "normal : " << isect.normal;
    std::cout << "tangent : " << isect.tangent;
    std::cout << "bitangent : " << isect.bitangent;
    isect.wo = -ray.direction;
    std::cout << "local wi : " << wi;
    isect.wi = tangent_to_world(wi, isect.normal, isect.tangent, isect.bitangent);
    std::cout << "world wi : " << isect.wi;
    std::cout << "local wo : " << wo;
    std::cout << "world wo : " << -ray.direction;

    //return spec;
    //return RGBSpectrum{0.5f, 0.5f, 0.5f};
    return f;
}
