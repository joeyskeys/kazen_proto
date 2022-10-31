#pragma once

#include <limits>

#include "base/basic_types.h"
#include "base/vec.h"

using base::Vec3f;

class Ray {
public:
    Ray(Vec3f o, Vec3f d, const float time=0.f, const float tmin=epsilon<float>, const float tmax=std::numeric_limits<float>::max())
    //Ray(Dual2V3f o, Dual2V3f d, const float time=0.f, const float tmin=0.f, const float tmax=std::numeric_limits<float>::Max())
        : origin(o)
        , direction(normalize(d))
        , tmin(tmin)
        , tmax(tmax)
        , time(time)
    {
        origin_dx = origin;
        origin_dy = origin;
    }

    inline Vec3f at(const float t) const {
        return origin + t * direction;
        //return origin.val() + direction.val() * t;
    }

    /*
    inline Dual2V3f at(const Dual2f t) const {
        return origin + direction * t;
    }
    */

    // Members
    Vec3f       origin;
    Vec3f       origin_dx;
    Vec3f       origin_dy;
    Vec3f       direction;
    Vec3f       direction_dx;
    Vec3f       direction_dy;
    float       tmin;
    float       tmax;
    float       time;
};

bool plane_intersect(const Ray& r, const Vec3f& center, const Vec3f& n, float& t);