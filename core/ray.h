#pragma once

#include "base/vec.h"

class Ray {
public:
    Ray(Vec3f o, Vec3f d)
        : origin(o)
        , direction(d)
    {}

    inline Vec3f at(const float t) const {
        return origin + t * direction;
    }

    // Members
    Vec3f   origin;
    Vec3f   direction;
    float   time;
};
