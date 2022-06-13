#pragma once

#include "base/dictlike.h"
#include "core/shape.h"
#include "core/spectrum.h"
#include "core/material.h"
#include "core/hitable.h"

struct LightRecord {
    // This struct is like a parameter pack in Nori and Mitsuba
    // renderers
    Vec3f shading_pt;
    Vec3f lighting_pt;
    Vec3f n;
    Vec2f uv;
    float pdf;
    float area;

    Ray     get_shadow_ray();
    Vec3f   get_shadow_ray_dir();
};

class Light : public DictLike {
public:
    Light(uint id, const std::string& sname, bool d=true)
        : light_id(id)
        , is_delta(d)
    {}

    virtual void        prepare(const RGBSpectrum& r) {
        radiance = r;
    }

    virtual LightRecord sample() const = 0;
    virtual RGBSpectrum sample(const Intersection& isect, Vec3f& light_dir, float& pdf, const HitablePtr scene) const = 0;
    virtual RGBSpectrum eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& pt_sample) const = 0;
    virtual float       pdf(const Intersection& isect, const Vec3f& p, const Vec3f& n) const {
        return 1.f;
    }

    // members
    std::string shader_name;
    RGBSpectrum radiance;
    bool is_delta;
    uint light_id;
};

class PointLight : public Light {
public:
    PointLight(uint id)
        : Light(id, "")
        , position(Vec3f(0, 5, 0))
    {}
    
    PointLight(uint id, const std::string& sname, const Vec3f& pos)
        : Light(id, sname)
        , position(pos)
    {}

    LightRecord sample() const;
    RGBSpectrum sample(const Intersection& isect, Vec3f& light_dir, float& pdf, const HitablePtr scene) const override;
    RGBSpectrum eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& pt_sample) const override;

    void* address_of(const std::string& name) override;
    
public:
    Vec3f position;
};

class GeometryLight : public Light {
public:
    GeometryLight(uint id)
        : Light(id, "", false)
        , geometry(nullptr)
    {}

    GeometryLight(uint id, const std::string& sname, std::shared_ptr<Shape>g)
        : Light(id, sname, false)
        , geometry(g)
    {}

    LightRecord sample() const;
    RGBSpectrum sample(const Intersection& isect, Vec3f& light_dir, float& pdf, const HitablePtr scene) const override;
    RGBSpectrum eval(const Intersection& isect, const Vec3f& light_dir, const Vec3f& pt_sample) const override;
    float       pdf(const Intersection& isect, const Vec3f& p, const Vec3f& n) const override;

public:
    std::shared_ptr<Shape> geometry;
};