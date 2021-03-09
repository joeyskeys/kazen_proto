#pragma once

#include "ray.h"
#include "transform.h"
#include "aabb.h"

class Intersection;

class HitableInterface {
public:
    // Intersection method
    virtual bool intersect(const Ray& r, Intersection& isect) const = 0;

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

protected:
    Transform local_to_world, world_to_local;
    AABBf bound;
};

using HitablePtr = Hitable*;