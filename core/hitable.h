#pragma once

#include "ray.h"
#include "transform.h"
#include "aabb.h"

class Intersection;

class HitableInterface {
public:
    virtual bool intersect(Ray& r, Intersection& isect) const = 0;
    virtual bool bbox(AABB& box) const = 0;
};

class Hitable : public HitableInterface {
public:
    Hitable(const Transform& l2w)
        : local_to_world(l2w)
        , world_to_local(l2w.inverse())
    {}

protected:
    Transform local_to_world, world_to_local;
};

using HitablePtr = Hitable*;