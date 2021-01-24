
#include "shape.h"
#include "ray.h"

bool Sphere::intersect(Ray& r, Intersection& isect) const {
    auto r_local = world_to_local.apply(r);

    auto oc = r_local.origin - center;
    auto a = r_local.direction.length_squared();
    auto half_b = dot(oc, r_local.direction);
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant < 0.f)
        return false;

    auto t0 = (-half_b - sqrtf(discriminant)) / a;
    auto t1 = (-half_b + sqrtf(discriminant)) / a;

    if (t0 > r.tmax || t1 < r.tmin) return false;
    float t = t0;
    isect.backface = false;
    //if (t <= 0) {
    if (t <= r.tmin) {
        t = t1;
        isect.backface = true;
        if (t > r.tmax) return false;
    }

    // Calculate in local space
    isect.ray_t = t;
    // small bias to avoid self intersection
    isect.position = r_local.at(t) * 1.00001f;
    isect.normal = (r_local.at(t) - center) / radius;
    if (isect.backface)
        isect.normal = -isect.normal;

    if (isect.normal == Vec3f(0.f, 1.f, 0.f))
        isect.tangent = normalize(Vec3f{0.f, 0.f, -1.f}.cross(isect.normal));
    else
        isect.tangent = normalize(Vec3f{0.f, 1.f, 0.f}.cross(isect.normal));

    isect.bitangent = normalize(isect.tangent.cross(isect.normal));
    isect.mat = mat;
    isect.obj_id = obj_id;

    // Transform back to world space
    isect = local_to_world.apply(isect);
    
    return true;
}

bool Triangle::intersect(Ray& r, Intersection& isect) const {
    return true;
}

bool TriangleMesh::intersect(Ray& r, Intersection& isect) const {
    return true;
}