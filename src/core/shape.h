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
    //std::weak_ptr<Light> light;
    Light*      light;
    uint        obj_id;
};

class Sphere : public Shape {
public:
    Sphere()
        : Shape()
        , center_n_radius(Vec4f(0, 0, 0, 1))
    {}

    Sphere(const Transform& l2w, const uint& id, const Vec4f cnr, const std::string& m="", bool is_l=false)
        : Shape(l2w, m, is_l, id)
        , center_n_radius(cnr)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    AABBf bbox() const override;

    void* address_of(const std::string& name) override;
    void sample(Vec3f& p, Vec3f& n, float& pdf) const override;

    void print_info() const override;

    void bake();

    // Members
    // For embree data structure
    Vec4f center_n_radius;
};

class Quad : public Shape {
public:
    Quad()
        : Shape()
        , center(Vec3f{0, 0, 0})
        , dir(Vec3f{0, 1, 0})
        , vertical_vec(Vec3f{0, 0, 1})
        , horizontal_vec(cross(vertical_vec, dir))
        , half_width(1)
        , half_height(1)
    {
        down_left = center - horizontal_vec * half_width - vertical_vec * half_height;
    }

    Quad(const Transform& l2w, const Vec3f& c, const Vec3f& d, const Vec3f& u, const float w, const float h, const std::string& m="", bool is_l=false)
        : Shape(l2w, m, is_l, 2)
        , center(c)
        , dir(d.normalized())
        , vertical_vec(u.normalized())
        , horizontal_vec(cross(vertical_vec, dir))
        , half_width(w)
        , half_height(h)
    {
        down_left = center - horizontal_vec * half_width - vertical_vec * half_height;
    }

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    AABBf bbox() const override;

    void* address_of(const std::string& name) override;
    void sample(Vec3f& p, Vec3f& n, float& pdf) const override;

    void print_info() const override;

    void get_verts(void* verts) const;

    Vec3f center;
    Vec3f dir;
    Vec3f vertical_vec;
    Vec3f horizontal_vec;
    float half_width;
    float half_height;
    Vec3f down_left;
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

    void print_info() const override;

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

    void print_info() const override;

    std::vector<Vec3f> verts;
    std::vector<Vec3i> indice;
};

std::vector<std::shared_ptr<Hitable>> load_triangle_mesh(const std::string& file_path, const std::string& m="");