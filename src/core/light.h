#pragma once

#include "base/dictlike.h"
#include "core/shape.h"
#include "core/spectrum.h"
#include "core/material.h"
#include "core/hitable.h"

class Light : public DictLike {
public:
    Light(uint id, RGBSpectrum l=RGBSpectrum(1.f, 1.f, 1.f), bool d=true)
        : light_id(id)
        , radiance(l)
        , is_delta(d)
    {}

    virtual RGBSpectrum sample(const Intersection& isect, Vec3f& light_dir, float& pdf, const HitablePtr scene) const = 0;
    virtual float       pdf(const Intersection& isect) const {
        return 1.f;
    }
    virtual RGBSpectrum eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& n) const = 0;

    // members
    RGBSpectrum radiance;
    bool is_delta;
    uint light_id;
};

class PointLight : public Light {
public:
    PointLight(uint id)
        : Light(id, RGBSpectrum(1, 1, 1), true)
        , position(Vec3f(0, 5, 0))
    {}
    
    PointLight(uint id, const RGBSpectrum& l, const Vec3f& pos)
        : Light(id, l, true)
        , position(pos)
    {}

    RGBSpectrum sample(const Intersection& isect, Vec3f& light_dir, float& pdf, const HitablePtr scene) const override;
    RGBSpectrum eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& n) const override;

    void* address_of(const std::string& name) override;
    
public:
    Vec3f position;
};

class GeometryLight : public Light {
public:
    GeometryLight(uint id)
        : Light(id, RGBSpectrum(1, 1, 1), false)
        , geometry(nullptr)
    {}

    GeometryLight(uint id, const RGBSpectrum& l, std::shared_ptr<Shape>g)
        : Light(id, l, false)
        , geometry(g)
    {}

    RGBSpectrum sample(const Intersection& isect, Vec3f& light_dir, float& pdf, const HitablePtr scene) const override;
    float       pdf(const Intersection& isect) const override;
    RGBSpectrum eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& n) const override;

public:
    std::shared_ptr<Shape> geometry;
};