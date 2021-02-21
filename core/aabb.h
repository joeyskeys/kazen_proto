#pragma once

#include <algorithm>

#include "base/vec.h"

class AABB {
public:
    AABB() {}
    AABB(const Vec3f& min, const Vec3f& max)
        : min(min)
        , max(max)
    {}

    bool intersect(Ray& r, double tmin, double tmax) {
        auto inv_dir = 1.f / r.direction;
        for (int i = 0; i < 3; i++) {
            /*
            auto t0 = std::min((min[i] - r.origin[i]) * inv_dir[i],
                (max[i] - r.origin[i]) * inv_dir[i]);
            auto t1 = std::max((min[i] - r.origin[i]) * inv_dir[i],
                (max[i] - r.origin[i]) * inv_dir[i]);
            tmin = std::max(t0, tmin);
            tmax = std::min(t1, tmax);
            if (tmax < tmin)
                return false;
            */
            auto t0 = (min[i] - r.origin[i]) * inv_dir[i];
            auto t1 = (max[i] - r.origin[i]) * inv_dir[i];
            if (inv_dir[i] < 0.f)
                std::swap(t0, t1);
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin)
                return false;
        }

        return true;
    }

    Vec3f min, max;
};