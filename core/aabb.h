#pragma once

#include <algorithm>

#include "base/vec.h"

template <typename T>
class AABB {
public:
    AABB() {}
    AABB(const Vec3<T>& min_vert, const Vec3<T>& max_vert)
        : min(min_vert)
        , max(max_vert)
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

    AABB union(const AABB& rhs) {
        auto new_min = min(min, rhs.min);
        auto new_max = max(max, rhs.max);
        return AABB{new_min, new_max};
    }

    AABB union(const Vec3<T>& rhs) {
        auto new_min = min(min, rhs);
        auto new_max = max(max, rhs);
        return AABB{new_min, new_max};
    }

    Vec3<T> min, max;
};

template <typename T>
inline AABB<T> union(const AABB<T>& a, const AABB<T>& b) {
    return a.union(b);
}

template <typename T>
inline AABB<T> union(const AABB<T>& a, const Vec3<T>& b) {
    return a.union(b);
}

using AABBf = AABB<float>;
using AABBd = AABB<double>;