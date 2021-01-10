
#include "shape.h"
#include "ray.h"

bool Sphere::intersect(Ray& r, Intersection& isect) const {
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

    //if (t0 > r.tmax || t1 < r.tmin) return false;
    if (t0 > r.tmax || t1 < r.t) return false;
    float t = t0;
    bool is_backface = false;
    //if (t <= 0) {
    if (t <= r.t) {
        t = t1;
        is_backface = true;
        if (t > r.tmax) return false;
    }

    r.t = t;
    isect.position = r.at(t);
    isect.normal = (isect.position - center) / radius;
    if (is_backface)
        isect.normal = -isect.normal;

    if (isect.normal == Vec3f(0.f, 1.f, 0.f))
        isect.tangent = Vec3f{0.f, 0.f, -1.f}.cross(isect.normal);
    else
        isect.tangent = Vec3f{0.f, 1.f, 0.f}.cross(isect.normal);

    isect.bitangent = isect.tangent.cross(isect.normal);
    isect.mat = mat;
    
    return true;
}

bool Triangle::intersect(Ray& r, Intersection& isect) const {
    return true;
}

bool TriangleMesh::intersect(Ray& r, Intersection& isect) const {
    return true;
}