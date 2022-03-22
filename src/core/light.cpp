

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

RGBSpectrum PointLight::eval(const Intersection& isect, const Vec3f& light_dir, float& pdf, const HitablePtr scene) const {
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

    // Visibility test (Done in eval)
    //auto shadow_ray = Ray(isect.P, light_dir);
    // self-intersection also possible here
    //shadow_ray.tmin = epsilon<float>;
    //if (scene->occluded(shadow_ray, geometry->geom_id))
        //return RGBSpectrum{0.f, 0.f, 0.f};
    
    return eval(isect, -light_dir, pdf, scene);
}

RGBSpectrum GeometryLight::eval(const Intersection& isect, const Vec3f& light_dir, float& pdf, const HitablePtr scene) const {
    Intersection tmpsect;
    if (!scene->intersect(Ray(isect.P, light_dir), tmpsect) || tmpsect.geom_id != geometry->geom_id)
        return 0.f;

    if (dot(tmpsect.N, isect.wi) >= 0)
        return 0.f;

    pdf = 1.f / geometry->area();
    return radiance;
}