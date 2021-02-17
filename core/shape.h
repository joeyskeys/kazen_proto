#pragma once

#include <vector>
#include <optional>

#include "transform.h"
#include "hitable.h"
#include "material.h"

class Ray;

class Shape : public Hitable {
public:
    Shape(const Transform& l2w, const uint& id)
        : Hitable(l2w)
        , obj_id(id)
    {}

public:
    MaterialPtr mat;
    uint obj_id;
};

class Sphere : public Shape {
public:
    Sphere(const Transform& l2w, const uint& id, const float r, const Vec3f& c=Vec3f{0.f, 0.f, 0.f})
        : Shape(l2w, id)
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
        : Shape(l2w, 0)
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
    TriangleMesh(const Transform& l2w, std::vector<Vec3f>&& vs, std::vector<Vec3i>&& idx)
        : Shape(l2w, 0)
        , verts(vs)
        , indice(idx)
    {}

    bool intersect(Ray& r, Intersection& isect) const override;

    std::vector<Vec3f> verts;
    std::vector<Vec3i> indice;
};

std::vector<TriangleMesh> load_triangle_mesh(const std::string& file_path);