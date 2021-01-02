#pragma once

#include "transform.h"
#include "hitable.h"

class Ray;

class Sphere : public Hitable {
public:
    Sphere(const Transform& l2w, const float r, const Vec3f& c=Vec3f{0.f, 0.f, 0.f})
        : Shape(l2w)
        , radius(r)
        , center(c)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override;

    // Members
    float radius;
    Vec3f center;
};

class Triangle : public Hitable {
public:
    Triangle(const Transform& l2w, const Vec3f& a, const Vec3f& b, const Vec3f& c)
        : Shape(l2w)
    {
        verts[0] = a;
        verts[1] = b;
        verts[2] = c;
    }

    bool intersect(const Ray& r, Intersection& isect) const override;

    Vec3f verts[3];
};

class TriangleMesh : public Hitable {
public:
    TriangleMesh(const Transform& l2w)
        : Shape(l2w)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override;
};