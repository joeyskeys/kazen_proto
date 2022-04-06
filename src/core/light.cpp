

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

RGBSpectrum PointLight::eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& pt_sample,
    float& pdf, const Vec3f& n) const {
    auto length_sqr = (position - isect.P).length_squared();
    pdf = 1.f;
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
    if (scene->occluded(shadow_ray, geometry->geom_id, n))
        return RGBSpectrum{0.f, 0.f, 0.f};
    
    // PDF must unified in unit, solid angle or area
    // Since we're using solid angle for bsdf, convert light pdf to
    // solid angle measured too
    // The result is still not correct..
    auto cos_theta_v = dot(-light_dir, n);
    if (cos_theta_v <= 0.f)
        return 0.f;
    auto length_sqr = (isect.P - p).length_squared();
    pdf = pdf * length_sqr / cos_theta_v;

    return eval(isect, -light_dir, p, pdf, n) / pdf;
}

RGBSpectrum GeometryLight::eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& pt_sample,
    float& pdf, const Vec3f& n) const {
    return radiance * intensity;
}