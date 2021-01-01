
#include "shape.h"
#include "ray.h"

bool Sphere::intersect(const Ray& r, Intersection& isect) const {
    auto oc = r.origin - center;
    auto a = r.direction.dot(r.direction);
    auto half_b = oc.dot(r.direction);
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant < 0.f)
        return false;

    auto t = (-half_b - sqrtf(discriminant)) / a;
    isect.p = r.at(t);
    isect.n = (isect.p - center) / radius;

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