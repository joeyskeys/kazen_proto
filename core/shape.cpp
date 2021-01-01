
#include "shape.h"

bool Sphere::intersect(const Ray& r, Intersection& isect) const {
    auto oc = r.origin - center;
    auto a = dot(r.direction, r.direction);
    auto b = 2.f * dot(oc, r.direction);
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant = b * b - 4 * a * c;
    return discriminant > 0.f;
}

bool Triangle::intersect(const Ray& r, Intersection& isect) const {

}

bool TriangleMesh::intersect(const Ray& r, Intersection& isect) const {

}