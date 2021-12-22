

#include "light.h"

RGBSpectrum PointLight::sample(const Intersection& isect, Vec3f& wi, float& pdf, const HitablePtr scene) const {
    // TODO : how to sample delta light?
    // Visibility test
    if (scene->occluded(position, isect.position))
        return RGBSpectrum{0.f, 0.f, 0.f};

    auto length_sqr = (position - isect.position).length_squared();
    pdf = 1.f;

    // length squared falloff
    return radiance / length_sqr;
}

RGBSpectrum PointLight::eval(const Intersection& isect, const Vec3f& wi, const Vec3f& n) const {
    auto length_sqr = (position - isect.position).length_squared();
    return radiance / length_sqr;
}

void* PointLight::address_of(const std::string& name) {
    if (name == "radiance")
        return &radiance;
    else if (name == "position")
        return &position;
    return nullptr;
}

RGBSpectrum GeometryLight::sample(const Intersection& isect, Vec3f& wi, float& pdf, const HitablePtr scene) const {
    Vec3f p, n;
    geometry->sample(p, n, pdf);

    // Visibility test
    if (scene->occluded(p, isect.position))
        return RGBSpectrum{0.f, 0.f, 0.f};
    
    wi = (p - isect.position).normalized();

    return eval(isect, wi, n);
}

RGBSpectrum GeometryLight::eval(const Intersection& isect, const Vec3f& wi, const Vec3f& n) const {
    auto cos_theta = dot(wi, n);
    return cos_theta < 0.f ? radiance : 0.f;
}