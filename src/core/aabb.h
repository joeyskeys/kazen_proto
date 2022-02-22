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

    bool intersect(const Ray& r) const {
        auto tmin = r.tmin;
        auto tmax = r.tmax;
        //auto inv_dir = 1.f / r.direction;
        auto inv_dir = 1.f / r.direction.val();
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

            //auto t0 = (min[i] - r.origin[i]) * inv_dir[i];
            auto t0 = (min[i] - r.origin.val()[i]) * inv_dir[i];
            //auto t1 = (max[i] - r.origin[i]) * inv_dir[i];
            auto t1 = (max[i] - r.origin.val()[i]) * inv_dir[i];

            // inf situation
            if (inv_dir[i] != inv_dir[i]) {
                //if (r.origin[i] > min[i] && r.origin[i] < max[i]) {
                if (r.origin.val()[i] > min[i] && r.origin.val()[i] < max[i]) {
                    return false;
                }
                else
                    continue;
            }

            if (inv_dir[i] < 0.f)
                std::swap(t0, t1);
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (r.tmax <= r.tmin) {
                return false;
            }
        }

        return true;
    }

    AABB bound_union(const AABB& rhs) const {
        auto new_min = vec_min(min, rhs.min);
        auto new_max = vec_max(max, rhs.max);
        return AABB{new_min, new_max};
    }

    AABB bound_union(const Vec3<T>& rhs) const {
        auto new_min = vec_min(min, rhs);
        auto new_max = vec_max(max, rhs);
        return AABB{new_min, new_max};
    }

    friend std::ostream& operator <<(std::ostream& os, const AABB& b) {
        os << "min : " << b.min;
        os << "max : " << b.max << std::endl;

        return os;
    }

    Vec3<T> min, max;
};

template <typename T>
inline AABB<T> bound_union(const AABB<T>& a, const AABB<T>& b) {
    return a.bound_union(b);
}

template <typename T>
inline AABB<T> bound_union(const AABB<T>& a, const Vec3<T>& b) {
    return a.bound_union(b);
}

using AABBf = AABB<float>;
using AABBd = AABB<double>;