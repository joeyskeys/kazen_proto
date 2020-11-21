#pragma once

#include "base/vec.h"

class Ray {
public:
    Ray(Vec3f o, Vec3f d)
        : origin(o)
        , direction(d) {
    }

    Vec3f at(const float t) const {
        return o + t * d;
    }

    // Members
    Vec3f   origin;
    Vec3f   direction;
    T       time;
};
