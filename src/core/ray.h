#pragma once

#include <limits>

#include "base/dual.h"
#include "base/vec.h"

class Ray {
public:
    //Ray(Vec3f o, Vec3f d, const float time=0.f, const float tmin=0.f, const float tmax=std::numeric_limits<float>::max())
    Ray(Dual2V3f o, Dual2V3f d, const float time=0.f, const float tmin=0.f, const float tmax=std::numeric_limits<float>::max())
        : origin(o)
        //, direction(d.normalized())
        , direction(normalize(d))
        , tmin(tmin)
        , tmax(tmax)
        , time(time)
    {}

    inline Vec3f at(const float t) const {
        //return origin + t * direction;
        return origin.val() + direction.val() * t;
    }

    inline Dual2V3f at(const Dual2f t) const {
        return origin + direction * t;
    }

    // Members
    //Vec3f   origin;
    //Vec3f   direction;
    Dual2V3f    origin;
    Dual2V3f    direction;
    float       tmin;
    float       tmax;
    float       time;
};
