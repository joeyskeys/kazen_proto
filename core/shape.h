#pragma once

#include <vector>
#include <optional>
#include <memory>

#include "transform.h"
#include "hitable.h"
#include "material.h"

class Ray;

class Shape : public Hitable {
public:
    Shape(const Transform& l2w, const MaterialPtr m, const uint& id)
        : Hitable(l2w)
        , mat(m)
        , obj_id(id)
    {}

    void print_bound() const override;

public:
    MaterialPtr mat;
    uint obj_id;
};

class Sphere : public Shape {
public:
    Sphere(const Transform& l2w, const uint& id, const float r, const MaterialPtr m=nullptr, const Vec3f& c=Vec3f{0.f, 0.f, 0.f})
        : Shape(l2w, m, id)
        , radius(r)
        , center(c)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    AABBf bbox() const override;

    // Members
    float radius;
    Vec3f center;
};

class Triangle : public Shape {
public:
    Triangle(const Transform& l2w, const Vec3f& a, const Vec3f& b, const Vec3f& c, const MaterialPtr m=nullptr)
        : Shape(l2w, m, 1)
    {
        verts[0] = a;
        verts[1] = b;
        verts[2] = c;
    }

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    AABBf bbox() const override;

    Vec3f verts[3];
};

class TriangleMesh : public Shape {
public:
    TriangleMesh(const Transform& l2w, std::vector<Vec3f>&& vs, std::vector<Vec3i>&& idx, const MaterialPtr m=nullptr)
        : Shape(l2w, m, 2)
        , verts(vs)
        , indice(idx)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    AABBf bbox() const override;

    std::vector<Vec3f> verts;
    std::vector<Vec3i> indice;
};

std::vector<std::shared_ptr<Hitable>> load_triangle_mesh(const std::string& file_path, const MaterialPtr m=nullptr);