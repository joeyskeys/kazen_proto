#pragma once

#include "base/dictlike.h"
#include "core/spectrum.h"
#include "core/material.h"
#include "core/hitable.h"

class VisibilityTester {
public:

};

class Light : public DictLike {
public:
    Light(RGBSpectrum l=RGBSpectrum(1.f, 1.f, 1.f), bool d=true)
        : radiance(l)
        , is_delta(d)
    {}

    virtual RGBSpectrum sample(const Intersection& isect, Vec3f& wi, float& pdf, const HitablePtr scene) const = 0;
    virtual float       pdf(const Intersection& isect, const Vec3f& wi) const {
        return 1.f;
    }
    virtual RGBSpectrum eval(const Intersection& isect, const Vec3f& wi, const Vec3f& n) const = 0;

    // members
    RGBSpectrum radiance;
    bool is_delta;
};

class PointLight : public Light {
public:
    PointLight()
        : Light(RGBSpectrum(1, 1, 1), true)
        , position(Vec3f(0, 5, 0))
    {}
    
    PointLight(const RGBSpectrum& l, const Vec3f& pos)
        : Light(l, true)
        , position(pos)
    {}

    RGBSpectrum sample(const Intersection& isect, Vec3f& wi, float& pdf, const HitablePtr scene) const override;
    RGBSpectrum eval(const Intersection& isect, const Vec3f& wi, const Vec3f& n) const override;

    void* address_of(const std::string& name) override;
    
public:
    Vec3f position;
};

class GeometryLight : public Light {
public:
    GeometryLight()
        : Light(RGBSpectrum(1, 1, 1), false)
        , geometry(nullptr)
    {}

    GeometryLight(const RGBSpectrum& l, std::unique_ptr<Shape>g)
        : Light(l, false)
        , geometry(g)
    {}

    RGBSpectrum sample(const Intersection& isect, Vec3f& wi, float& pdf, const HitablePtr scene) const override;
    RGBSpectrum eval(const Intersection& isect, const Vec3f& wi, const Vec3f& n) const override;

public:
    std::shared_ptr<Shape> geometry;
};