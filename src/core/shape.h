#pragma once

#include <vector>
#include <optional>
#include <memory>

#include "base/dictlike.h"
#include "core/transform.h"
#include "core/hitable.h"
#include "core/material.h"

class Ray;
class Light;

class Shape : public Hitable, public DictLike {
public:
    Shape()
        : Hitable(Transform())
        , obj_id(0)
    {}

    Shape(const Transform& l2w, std::string m, bool is_l, const uint& id)
        : Hitable(l2w)
        , shader_name(m)
        , is_light(is_l)
        , obj_id(id)
    {}

    virtual void sample(Vec3f& p, Vec3f& n, float& pdf) const = 0;
    void print_bound() const override;

public:
    std::string shader_name;
    bool        is_light = false;
    std::weak_ptr<Light> light;
    uint        obj_id;
};

class Sphere : public Shape {
public:
    Sphere()
        : Shape()
        , radius(1)
        , center(Vec3f(0, 0, 0))
    {}

    Sphere(const Transform& l2w, const uint& id, const float r, const std::string& m="", bool is_l=false, const Vec3f& c=Vec3f{0.f, 0.f, 0.f})
        : Shape(l2w, m, is_l, id)
        , radius(r)
        , center(c)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    AABBf bbox() const override;

    void* address_of(const std::string& name) override;
    void sample(Vec3f& p, Vec3f& n, float& pdf) const override;

    // Members
    float radius;
    Vec3f center;
};

class Triangle : public Shape {
public:
    Triangle()
        : Shape()
    {
        verts[0] = Vec3f(0, 0, 0);
        verts[1] = Vec3f(1, 0, 0);
        verts[2] = Vec3f(0, 0, -1);
        normal = cross(verts[1] - verts[0], verts[2] - verts[0]).normalized();
    }

    Triangle(const Transform& l2w, const Vec3f& a, const Vec3f& b, const Vec3f& c, const std::string m="", bool is_l=false)
        : Shape(l2w, m, is_l, 1)
    {
        verts[0] = a;
        verts[1] = b;
        verts[2] = c;
        normal = cross(verts[1] - verts[0], verts[2] - verts[0]).normalized();
    }

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    AABBf bbox() const override;

    void* address_of(const std::string& name) override;
    void sample(Vec3f& p, Vec3f& n, float& pdf) const override;

    Vec3f verts[3];
    Vec3f normal;
};

class TriangleMesh : public Shape {
public:
    TriangleMesh(const Transform& l2w, std::vector<Vec3f>&& vs, std::vector<Vec3i>&& idx, const std::string m="", bool is_l=false)
        : Shape(l2w, m, is_l, 2)
        , verts(vs)
        , indice(idx)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    AABBf bbox() const override;

    //void* address_of(const std::string& name) override
    void sample(Vec3f& p, Vec3f& n, float& pdf) const override;

    std::vector<Vec3f> verts;
    std::vector<Vec3i> indice;
};

std::vector<std::shared_ptr<Hitable>> load_triangle_mesh(const std::string& file_path, const std::string& m="");