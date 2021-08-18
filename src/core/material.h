#pragma once

#include "base/vec.h"
#include "base/utils.h"
#include "core/intersection.h"
#include "core/spectrum.h"
#include "core/ray.h"

class BxDF {
public:
    virtual RGBSpectrum f(const Vec3f& wo, const Vec3f& wi, const Intersection& isect) const = 0;
    virtual float       pdf(const Vec3f& wo, const Vec3f& wi) const;
    virtual RGBSpectrum sample_f(const Vec3f& wo, Vec3f& wi, const Intersection& isect, const Vec2f& u, float& p) const;
};

class LambertianBxDF : public BxDF {
public:
    LambertianBxDF(const RGBSpectrum& c)
        : color(c)
    {}

    RGBSpectrum f(const Vec3f& wo, const Vec3f& wi, const Intersection& isect) const override;

private:
    RGBSpectrum color;
};

class MetalBxDF : public BxDF {
public:
    MetalBxDF(const RGBSpectrum& c)
        : color(c)
    {}

    RGBSpectrum f(const Vec3f& wo, const Vec3f& wi, const Intersection& isect) const override;
    float       pdf(const Vec3f& wo, const Vec3f& wi) const override;
    RGBSpectrum sample_f(const Vec3f& wo, Vec3f& wi, const Intersection& isect, const Vec2f& u, float& p) const override;

private:
    RGBSpectrum color;
};

using BxDFPtr = BxDF*;

class Material {
public:
    RGBSpectrum calculate_response(Intersection& isect, Ray& ray) const;

    BxDFPtr bxdf;
};
