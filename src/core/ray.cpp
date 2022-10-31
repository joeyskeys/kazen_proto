#include "ray.h"

bool plane_intersect(const Ray& r, const Vec3f& center, const Vec3f& n, float& t) {
    auto pos_vec = center - r.origin;
    auto projected_distance = base::dot(pos_vec, n);

    // Back facing hit ignored
    if (projected_distance >= 0)
        return false;

    // Plane direction vector is supposed to be normalized
    // No extra check here
    auto projected_direction = base::dot(n, r.direction);
    
    // Avoid zero division when ray is parallel to the plane
    if (projected_direction == 0.f)
        return false;

    t = projected_distance / projected_direction;
    if (t < 0)
        return false;

    return true;
};