

#include "light.h"

RGBSpectrum PointLight::sample(const Intersection& isect, Vec3f& light_dir, float& pdf, const HitablePtr scene) const {
    // TODO : how to sample delta light?
    // Visibility test
    if (scene->occluded(position, isect.P))
        return RGBSpectrum{0.f, 0.f, 0.f};

    auto connected_vec = position - isect.P;
    light_dir = connected_vec.normalized();
    auto length_sqr = connected_vec.length_squared() * 0.1;
    pdf = 1.f;

    // length squared falloff
    return radiance / length_sqr;
}

RGBSpectrum PointLight::eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& n) const {
    auto length_sqr = (position - isect.P).length_squared();
    return radiance / length_sqr;
}

void* PointLight::address_of(const std::string& name) {
    if (name == "radiance")
        return &radiance;
    else if (name == "position")
        return &position;
    return nullptr;
}

RGBSpectrum GeometryLight::sample(const Intersection& isect, Vec3f& light_dir, float& pdf, const HitablePtr scene) const {
    Vec3f p, n;
    geometry->sample(p, n, pdf);
    light_dir = (p - isect.P).normalized();

    // Visibility test
    auto shadow_ray = Ray(isect.P, light_dir);
    // self-intersection also possible here
    shadow_ray.tmin = epsilon<float>;
    if (scene->occluded(shadow_ray, geometry->geom_id))
        return RGBSpectrum{0.f, 0.f, 0.f};
    
    return eval(isect, -light_dir, n);
}

float GeometryLight::pdf(const Intersection& isect) const {
    return 0.1f;
}

RGBSpectrum GeometryLight::eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& n) const {
    auto cos_theta = dot(light_dir, n);
    return cos_theta > 0.f ? radiance : 0.f;
}