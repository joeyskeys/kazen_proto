#pragma once

#include "ray.h"
#include "transform.h"
#include "aabb.h"

class Intersection;

class HitableInterface {
public:
    // Intersection method
    virtual bool intersect(const Ray& r, Intersection& isect) const = 0;
    virtual bool intersect(const Ray& r, float& t) const = 0;

    inline bool occluded(const Vec3f& p1, const Vec3f& p2) const {
        auto vec_p1p2 = (p2 - p1).normalized();
        Ray r(p1 + vec_p1p2 * 0.00001f, vec_p1p2);
        float ray_t;

        if (intersect(r, ray_t) && ray_t < vec_p1p2.length())
            return true;
        return false;
    }

    inline bool occluded(const Ray& r, size_t& dest_geom_id, Vec3f& light_normal) const {
        Intersection isect;
        if (!intersect(r, isect) || isect.geom_id != dest_geom_id)
            return true;
        
        light_normal = isect.N;
        return false;
    }

    // Return the bounding box in world space
    virtual AABBf bbox() const = 0;
};

class Hitable : public HitableInterface {
public:
    Hitable() {}

    Hitable(const Transform& l2w)
        : local_to_world(l2w)
        , world_to_local(l2w.inverse())
    {}

    AABBf bbox() const override {
        return bound;
    }

    virtual void print_bound() const {
        std::cout << bbox() << std::endl;
    }

    virtual void print_info() const {
        std::cout << "Not implemented." << std::endl;
    }

    void translate(const Vec3f& t) {
        local_to_world.translate(t);
        world_to_local.translate(-t);
    }

public:
    Transform local_to_world, world_to_local;
    AABBf bound;
};

using HitablePtr = Hitable*;