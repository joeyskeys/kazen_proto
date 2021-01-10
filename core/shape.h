#pragma once

#include "transform.h"
#include "hitable.h"
#include "material.h"

class Ray;

class Shape : public Hitable {
public:
    Shape(const Transform& l2w)
        : Hitable(l2w)
    {}

public:
    MaterialPtr mat;
};

class Sphere : public Shape {
public:
    Sphere(const Transform& l2w, const float r, const Vec3f& c=Vec3f{0.f, 0.f, 0.f})
        : Shape(l2w)
        , radius(r)
        , center(c)
    {}

    bool intersect(Ray& r, Intersection& isect) const override;

    // Members
    float radius;
    Vec3f center;
};

class Triangle : public Shape {
public:
    Triangle(const Transform& l2w, const Vec3f& a, const Vec3f& b, const Vec3f& c)
        : Shape(l2w)
    {
        verts[0] = a;
        verts[1] = b;
        verts[2] = c;
    }

    bool intersect(Ray& r, Intersection& isect) const override;

    Vec3f verts[3];
};

class TriangleMesh : public Shape {
public:
    TriangleMesh(const Transform& l2w)
        : Shape(l2w)
    {}

    bool intersect(Ray& r, Intersection& isect) const override;
};