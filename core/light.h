#pragma once

#include "spectrum.h"
#include "material.h"

class VisibilityTester {
public:

};

class Light {
public:
    Light(RGBSpectrum l=RGBSpectrum(1.f, 1.f, 1.f), bool d=true)
        : light_radiance(l)
        , is_delta(d)
    {}

    virtual RGBSpectrum sample_l(const Intersection& isect, const Vec2f& u, Vec3f& wi, float& pdf, const HitablePtr scene) const = 0;
    virtual float       pdf(const Intersection& isect, const Vec3f& wi) const {
        return 1.f;
    }

    // members
    RGBSpectrum light_radiance;
    bool is_delta;
};

class PointLight : public Light {
public:
    PointLight(const RGBSpectrum& l, const Vec3f& pos)
        : Light(l, true)
        , position(pos)
    {}

    RGBSpectrum sample_l(const Intersection& isect, const Vec2f& u, Vec3f& wi, float& pdf, const HitablePtr scene) const;

    Vec3f position;
}