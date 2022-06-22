

#include "core/light.h"

Ray LightRecord::get_shadow_ray() {
    return Ray(lighting_pt, base::normalize(shading_pt - lighting_pt));
}

Vec3f LightRecord::get_light_dir() {
    return base::normalize(shading_pt - lighting_pt);
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
    light_dir = base::normalize(connected_vec);
    auto length_sqr = base::length_squared(connected_vec) * 0.1;
    pdf = 1.f;

    // length squared falloff
    return radiance / length_sqr;
}

RGBSpectrum PointLight::eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& pt_sample) const {
    auto length_sqr = base::length_squared(position - isect.P);
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
    lrec.pdf *= base::length_squared(lrec.shading_pt - lrec.lighting_pt) / dot(lrec.get_light_dir(), lrec.n);
    lrec.area = geometry->area();
    return lrec;
}

RGBSpectrum GeometryLight::sample(const Intersection& isect, Vec3f& light_dir, float& pdf, const HitablePtr scene) const {
    Vec3f p, n;
    Vec2f uv;
    geometry->sample(p, n, uv, pdf);
    auto light_vec = p - isect.P;
    light_dir = base::normalize(light_vec);

    // Visibility test
    auto shadow_ray = Ray(isect.P, light_dir);
    // self-intersection also possible here
    shadow_ray.tmin = epsilon<float>;
    shadow_ray.tmax = base::length(light_vec) - epsilon<float>;
    if (scene->occluded(shadow_ray, geometry->geom_id, n))
        return RGBSpectrum{0.f, 0.f, 0.f};
    
    // PDF must unified in unit, solid angle or area
    // Since we're using solid angle for bsdf, convert light pdf to
    // solid angle measured too
    auto cos_theta_v = dot(-light_dir, n);
    if (cos_theta_v <= 0.f)
        return 0.f;
    auto length_sqr = base::length_squared(isect.P - p);
    pdf = pdf * length_sqr / cos_theta_v;

    return eval(isect, -light_dir, p) / pdf;
}

RGBSpectrum GeometryLight::eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& pt_sample) const {
    return radiance;
}

float GeometryLight::pdf(const Intersection& isect, const Vec3f& p, const Vec3f& n) const {
    auto light_vec = isect.P - p;
    return 1. / geometry->area() * base::length_squared(light_vec) / dot(base::normalize(light_vec), n);
}