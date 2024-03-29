#pragma once

#include <vector>
#include <optional>
#include <memory>

#include "base/dictlike.h"
#include "base/dpdf.h"
#include "core/transform.h"
#include "core/hitable.h"
#include "core/material.h"

class Ray;
class Light;

class Shape : public Hitable, public DictLike {
public:
    Shape(const size_t id)
        : Hitable(Transform())
        , geom_id(id)
    {}

    Shape(const Transform& l2w, const std::string& n, const std::string& m,
        bool is_l, const uint& id)
        : Hitable(l2w)
        , name(n)
        , shader_name(m)
        , is_light(is_l)
        , geom_id(id)
    {}

    virtual void sample(Vec3f& p, Vec3f& n, Vec2f& uv, float& pdf) const = 0;

    // This function is added to handle tangent & bitangent calculation
    // in embree intersection
    virtual void post_hit(Intersection& isect) const;
    virtual float area() const = 0;
    void print_bound() const override;

public:
    std::string name;
    std::string shader_name;
    bool        is_light = false;
    //std::weak_ptr<Light> light;
    Light*      light;
    size_t      geom_id;
};

class Sphere : public Shape {
public:
    Sphere(const size_t id)
        : Shape(id)
        , center_n_radius(Vec4f(0, 0, 0, 1))
    {}

    Sphere(const Transform& l2w, const std::string& n, const uint& id, const Vec4f cnr, const std::string& m="", bool is_l=false)
        : Shape(l2w, n, m, is_l, id)
        , center_n_radius(cnr)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    AABBf bbox() const override;

    void* address_of(const std::string& name) override;
    void sample(Vec3f& p, Vec3f& n, Vec2f& uv, float& pdf) const override;
    void post_hit(Intersection& isect) const override;
    float area() const override;
    void print_info() const override;

    void get_world_position(Vec4f* cnr) const;

    // Members
    // For embree data structure
    Vec4f center_n_radius;
};

class Quad : public Shape {
public:
    Quad(const size_t id)
        : Shape(id)
        , center(Vec3f{0, 0, 0})
        , dir(Vec3f{0, 1, 0})
        , vertical_vec(Vec3f{0, 0, 1})
        , horizontal_vec(cross(vertical_vec, dir))
        , half_width(1)
        , half_height(1)
    {
        down_left = center - horizontal_vec * half_width - vertical_vec * half_height;
    }

    Quad(const Transform& l2w, const Vec3f& c, const Vec3f& d, const Vec3f& u, const float w, const float h, const std::string& n, const std::string& m="", bool is_l=false)
        : Shape(l2w, n, m, is_l, 2)
        , center(c)
        , dir(normalize(d))
        , vertical_vec(normalize(u))
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
    void sample(Vec3f& p, Vec3f& n, Vec2f& uv, float& pdf) const override;
    void post_hit(Intersection& isect) const override;
    float area() const override;
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
    Triangle(size_t id)
        : Shape(id)
    {
        verts[0] = Vec3f(0, 0, 0);
        verts[1] = Vec3f(1, 0, 0);
        verts[2] = Vec3f(0, 0, -1);
        normal = base::normalize(base::cross(verts[1] - verts[0], verts[2] - verts[0]));
    }

    Triangle(const Transform& l2w, const Vec3f& a, const Vec3f& b, const Vec3f& c, const std::string& n, const std::string m="", bool is_l=false)
        : Shape(l2w, n, m, is_l, 1)
    {
        verts[0] = a;
        verts[1] = b;
        verts[2] = c;
        normal = base::normalize(base::cross(verts[1] - verts[0], verts[2] - verts[0]));
    }

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    AABBf bbox() const override;

    void* address_of(const std::string& name) override;
    void sample(Vec3f& p, Vec3f& n, Vec2f& uv, float& pdf) const override;
    void post_hit(Intersection& isect) const override;
    float area() const override;
    void print_info() const override;

    Vec3f verts[3];
    Vec3f normal;
};

class TriangleArray: public Shape {
public:
    TriangleArray(const size_t id)
        : Shape(id)
    {}

    TriangleArray(const Transform& l2w, std::vector<Vec3f>&& vs, const std::string& n,
        const std::string& m = "", const bool is_light = false)
        : Shape(l2w, n, m, is_light, 0)
        , verts(vs)
    {}

    TriangleArray(const Transform& l2w, const std::vector<Vec4f>& vs, const std::string& n,
        const std::string& m = "", const bool is_light = false)
        : Shape(l2w, n, m, is_light, 0)
        , converted_verts(vs)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override { return false; }
    bool intersect(const Ray& r, float& t) const override { return false; }
    void sample(Vec3f& p, Vec3f& n, Vec2f& uv, float& pdf) const override {}
    float area() const override { return 0.f; }
    void convert_to_4f_alignment();

    std::vector<Vec3f> verts;
    std::vector<Vec4f> converted_verts;
    bool use_v3f = true;
};

class TriangleMesh : public Shape {
public:
    TriangleMesh(const size_t id)
        : Shape(id)
    {}

    TriangleMesh(const Transform& l2w, std::vector<Vec3f>&& vs, std::vector<Vec3f>&& ns, std::vector<Vec2f>&& ts, std::vector<Vec3i>&& idx, const std::string n, const std::string m="", bool is_l=false)
        : Shape(l2w, n, m, is_l, 0)
        , verts(vs)
        , norms(ns)
        , uvs(ts)
        , indice(idx)
    {}

    TriangleMesh(const Transform& l2w, const std::vector<Vec3f>& vs,
        const std::vector<Vec3f>& ns, const std::vector<Vec2f>& ts,
        const std::vector<Vec3i>& idx, const std::string& n,
        const std::string& m="", bool is_l=false)
        : Shape(l2w, n, m, is_l, 0)
        , verts(vs)
        , norms(ns)
        , uvs(ts)
        , indice(idx)
    {}

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    AABBf bbox() const override;

    //void* address_of(const std::string& name) override
    void sample(Vec3f& p, Vec3f& n, Vec2f& uv, float& pdf) const override;
    void post_hit(Intersection& isect) const override;
    float area() const override;
    void print_info() const override;

    float surface_area(uint32_t i) const;
    void setup_dpdf();
    void convert_to_4f_alignment();

    std::vector<Vec3f>  verts;
    std::vector<Vec4f>  converted_verts;
    std::vector<Vec3f>  norms;
    std::vector<Vec2f>  uvs;
    std::vector<Vec3i>  uv_idx;
    std::vector<Vec3i>  indice;
    DiscrectPDF         m_dpdf;
    float               m_area;
    bool                use_v3f = true;
};

std::vector<std::shared_ptr<TriangleMesh>> load_triangle_mesh(const std::string& file_path, const size_t start_id, const std::string& m="");