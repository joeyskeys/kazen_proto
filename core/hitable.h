#pragma once

#include "intersection.h"
#include "ray.h"
#include "transform.h"

class HitableInterface {
public:
    virtual bool intersect(const Ray& r, Intersection& isect) const = 0;
};

class Hitable : HitableInterface {
public:
    Hitable(const Transform& l2w)
        : local_to_world(l2w)
        , world_to_local(l2w.inverse())
    {}

protected:
    Transform local_to_world, world_to_local;
}

using HitablePtr = Hitable*;