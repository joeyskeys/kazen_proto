
#include "shape.h"
#include "ray.h"

bool Sphere::intersect(const Ray& r, Intersection& isect) const {
    auto r_local = world_to_local.apply(r);

    auto oc = r_local.origin - center;
    auto a = r_local.direction.dot(r_local.direction);
    auto half_b = oc.dot(r_local.direction);
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant < 0.f)
        return false;

    auto t0 = (-half_b - sqrtf(discriminant)) / a;
    auto t1 = (-half_b + sqrtf(discriminant)) / a;

    if (t0 > r.tmax || t1 < r.tmin) return false;
    float t = t0;
    bool is_backface = false;
    if (t <= 0) {
        t = t1;
        is_backface = true;
        if (t > r.tmax) return false;
    }

    isect.p = r.at(t);
    isect.n = (isect.p - center) / radius;
    if (is_backface)
        isect.n = -isect.n;

    if (isect.n == Vec3f(0.f, 1.f, 0.f))
        isect.t = Vec3f{0.f, 0.f, -1.f}.cross(isect.n);
    else
        isect.t = Vec3f{0.f, 1.f, 0.f}.cross(isect.n);

    isect.b = isect.t.cross(isect.n);
    
    return true;
}

bool Triangle::intersect(const Ray& r, Intersection& isect) const {
    return true;
}

bool TriangleMesh::intersect(const Ray& r, Intersection& isect) const {
    return true;
}