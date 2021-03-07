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

    bool intersect(Ray& r) const {
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

            // inf situation
            if (inv_dir[i] != inv_dir[i]) {
                if (r.origin[i] > min[i] && r.origin[i] < max[i]) {
                    //std::cout << "inf and out of range index : " << i << std::endl;
                    return false;
                }
                else
                    continue;
            }

            if (inv_dir[i] < 0.f)
                std::swap(t0, t1);
            //std::cout << "t0 : " << t0 << ", t1 : " << t1 << std::endl;
            //std::cout << "tmin : " << r.tmin << ", tmax : " << r.tmax << std::endl;
            r.tmin = t0 > r.tmin ? t0 : r.tmin;
            r.tmax = t1 < r.tmax ? t1 : r.tmax;
            if (r.tmax <= r.tmin) {
                //std::cout << "index : " << i << ", min : " << r.tmin << ", max : " << r.tmax << std::endl;
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