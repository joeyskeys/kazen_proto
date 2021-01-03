#pragma once

#include "base/vec.h"
#include "base/utils.h"
#include "spectrum.h"

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

class Material {
    RGBSpectrum calculate_response(const Vec3f& wo, float& p) const;
    RGBSpectrum calculate_response(const Vec3f& wo, Vec3f& wi, float& p) const;
};

using MaterialPtr = Material*;