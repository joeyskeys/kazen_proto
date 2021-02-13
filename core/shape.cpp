
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

static bool moller_trumbore_intersect(const Ray& r, const Vec3f* verts, Intersection& isect) {
    Vec3f v1v0 = verts[1] - verts[0];
    Vec3f v2v0 = verts[2] - verts[0];
    Vec3f pvec = cross(r.direction, v2v0);
    float det = dot(v1v0, pvec);

    if (det < 0.000001) return false;

    float det_inv = 1.f / det;
    Vec3f tvec = r.origin - verts[0];
    float u = dot(tvec, pvec) * det_inv;
    if (u < 0.f || u > 1.f) return false;

    Vec3f qvec = cross(tvec, v1v0);
    float v = dot(r.direction, qvec) * det_inv; 
    if (v < 0.f || u + v > 1.f) return false;

    float t = dot(v2v0, qvec) * det_inv;

    isect.position = r.at(t);
    isect.normal = cross(v1v0, v2v0).normalized();
    isect.ray_t = t;
    isect.backface = t < 0.f ? true : false;

    return true;
}

bool Triangle::intersect(Ray& r, Intersection& isect) const {
    auto local_r = world_to_local.apply(r);

    auto ret = moller_trumbore_intersect(local_r, verts, isect);
    if (ret) {
        isect.mat = mat;
        isect = local_to_world.apply(isect);
    }

    return ret;
}

bool TriangleMesh::intersect(Ray& r, Intersection& isect) const {
    auto local_r = world_to_local.apply(r);
    bool hit = false;

    for (auto& idx : indice) {
        Vec3f tri[3];
        tri[0] = verts[idx[0]];
        tri[1] = verts[idx[1]];
        tri[2] = verts[idx[2]];
        hit = moller_trumbore_intersect(local_r, tri, isect);
        if (hit) {
            isect.mat = mat;
            isect = local_to_world.apply(isect);
            return true;
        }
    }

    return false;
}