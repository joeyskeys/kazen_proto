#pragma once

#include "base/vec.h"
#include "base/utils.h"
#include "spectrum.h"
#include "ray.h"

class BxDF {
public:
    virtual RGBSpectrum f(const Vec3f& wo, const Vec3f& wi) const = 0;
    virtual float       pdf(const Vec3f& wo, const Vec3f& wi) const;
    virtual RGBSpectrum sample_f(const Vec3f& wo, Vec3f& wi, const Vec2f& u, float& p) const;
};

class LambertianBxDF : public BxDF {
public:
    LambertianBxDF(const RGBSpectrum& c)
        : color(c)
    {}

    RGBSpectrum f(const Vec3f& wo, const Vec3f& wi) const override;

private:
    RGBSpectrum color;
};

using BxDFPtr = BxDF*;

struct Intersection;

class Material {
public:
    RGBSpectrum calculate_response(Intersection& isect, const Ray& ray) const;

    BxDFPtr bxdf;
};

using MaterialPtr = Material*;

struct Intersection {
    Vec3f p;
    Vec3f n;
    Vec3f ng;
    Vec3f t;
    Vec3f b;
    Vec3f bary;
    Vec2f uv;
    MaterialPtr mat;
};