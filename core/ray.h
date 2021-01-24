#pragma once

#include <limits>

#include "base/vec.h"

class Ray {
public:
    Ray(Vec3f o, Vec3f d, const float time=0.f, const float tmin=0.f, const float tmax=std::numeric_limits<float>::max())
        : origin(o)
        , direction(d.normalized())
        , tmin(tmin)
        , tmax(tmax)
        , time(time)
    {}

    inline Vec3f at(const float t) const {
        return origin + t * direction;
    }

    // Members
    Vec3f   origin;
    Vec3f   direction;
    float   tmin;
    float   tmax;
    float   time;
};
