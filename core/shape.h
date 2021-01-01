#pragma once

#include "transform.h"
#include "intersection.h"

class Shape {
public:
    Shape(const Transform& l2w)
        : local_to_world(l2w)
        , world_to_local(l2w.inverse())
    {}

    virtual bool intersect(const Ray& r, Intersection& isect) const = 0;
    
    // Members
    Transform local_to_world, world_to_local;
};

class Sphere : public Shape {
public:
    Sphere(const Transform& l2w, const Vec3f c, const float r)
        : Shape(l2w)
        , center(c)
        , radius(r)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override;

    // Members
    Vec3f center;
    float radius;
};

class Triangle : public Shape {
public:
    Triangle(const Transform& l2w, const Vec3f& a, const Vec3& b, const Vec3& c)
        : Shape(l2w)
    {
        verts[0] = a;
        verts[1] = b;
        verts[2] = c;
    }

    bool intersect(const Ray& r, Intersection& isect) const override;

    Vec3f verts[3]
};

class TriangleMesh : public Shape {
public:
    TriangleMesh(const Transform& l2w)
        : Shape(l2w)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override;
};