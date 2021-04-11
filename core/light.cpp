

#include "light.h"

RGBSpectrum PointLight::sample_l(const Intersection& isect, const Vec2f& u, Vec3f& wi, float& pdf, cosnt HitablePtr scene) const {
    auto connected_vec = position - isect.position;
    wi = normalize(connected_vec);
    auto length = connected_vec.length();
    pdf = 1.f;

    Ray r(isect.position, wi);
    Intersection isect;
    if (scene.intersect(r, isect) && isect.ray_t < length)
        return RGBSpectrum(0.f, 0.f, 0.f);

    // length squared falloff
    return light_radiance / (length * length);
}