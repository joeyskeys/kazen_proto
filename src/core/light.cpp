

#include "light.h"

RGBSpectrum PointLight::sample_l(const Intersection& isect, Vec3f& wi, float& pdf, const HitablePtr scene) const {
    // Visibility test
    if (scene->occluded(position, isect.position))
        return RGBSpectrum{0.f, 0.f, 0.f};

    auto connected_vec = position - isect.position;
    wi = normalize(connected_vec);
    auto length = connected_vec.length();
    pdf = 1.f;

    Ray r(isect.position, wi);
    Intersection isect_tmp;
    if (scene->intersect(r, isect_tmp) && isect.ray_t < length)
        return RGBSpectrum(0.f, 0.f, 0.f);

    // length squared falloff
    return radiance / (length * length);
}

void* PointLight::address_of(const std::string& name) {
    if (name == "radiance")
        return &radiance;
    else if (name == "position")
        return &position;
    return nullptr;
}