

#include "core/light.h"

Ray LightRecord::get_shadow_ray() {
    return Ray(lighting_pt, (shading_pt - lighting_pt).normalized());
}

Vec3f LightRecord::get_light_dir() {
    return (shading_pt - lighting_pt).normalized();
}

LightRecord PointLight::sample() const {
    // Not implemented yet
    throw std::runtime_error("Not implemented");
}

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

RGBSpectrum PointLight::eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& pt_sample) const {
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

LightRecord GeometryLight::sample() const {
    LightRecord lrec;
    geometry->sample(lrec.lighting_pt, lrec.n, lrec.uv, lrec.pdf);
    lrec.pdf *= (lrec.shading_pt - lrec.lighting_pt).length_squared() / dot(lrec.get_light_dir(), lrec.n);
    lrec.area = geometry->area();
    return lrec;
}

RGBSpectrum GeometryLight::sample(const Intersection& isect, Vec3f& light_dir, float& pdf, const HitablePtr scene) const {
    Vec3f p, n;
    Vec2f uv;
    geometry->sample(p, n, uv, pdf);
    auto light_vec = p - isect.P;
    light_dir = light_vec.normalized();

    // Visibility test
    auto shadow_ray = Ray(isect.P, light_dir);
    // self-intersection also possible here
    shadow_ray.tmin = epsilon<float>;
    shadow_ray.tmax = light_vec.length() - epsilon<float>;
    if (scene->occluded(shadow_ray, geometry->geom_id, n))
        return RGBSpectrum{0.f, 0.f, 0.f};
    
    // PDF must unified in unit, solid angle or area
    // Since we're using solid angle for bsdf, convert light pdf to
    // solid angle measured too
    auto cos_theta_v = dot(-light_dir, n);
    if (cos_theta_v <= 0.f)
        return 0.f;
    auto length_sqr = (isect.P - p).length_squared();
    pdf = pdf * length_sqr / cos_theta_v;

    return eval(isect, -light_dir, p) / pdf;
}

RGBSpectrum GeometryLight::eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& pt_sample) const {
    return radiance;
}

float GeometryLight::pdf(const Intersection& isect, const Vec3f& p, const Vec3f& n) const {
    auto light_vec = isect.P - p;
    return 1. / geometry->area() * light_vec.length_squared() / dot(light_vec.normalized(), n);
}