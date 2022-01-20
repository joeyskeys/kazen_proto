#pragma once

#include <vector>
#include <memory>

#include <embree3/rtcore.h>

#include "hitable.h"
#include "shape.h"

class Accelerator : public Hitable {
public:
    Accelerator(std::vector<std::shared_ptr<Hitable>>* hs)
        : hitables(hs)
    {}

    inline size_t size() const { return hitables->size(); }
    void add_hitable(std::shared_ptr<Hitable>&& h);
    void add_hitables(const std::vector<std::shared_ptr<Hitable>>& hs);

    virtual void add_sphere(std::shared_ptr<Sphere>& s);
    virtual void add_quad(std::shared_ptr<Quad>& q);
    virtual void add_triangle(std::shared_ptr<Triangle>& t);
    virtual void add_trianglemesh(std::shared_ptr<TriangleMesh>& t);
    virtual void build() {}

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    void print_info() const override;

public:
    std::vector<std::shared_ptr<Hitable>>* hitables;
};

class BVHNode;

class BVHAccel : public Accelerator {
public:
    BVHAccel(std::vector<std::shared_ptr<Hitable>>* hs)
        : Accelerator(hs)
        , root(nullptr)
    {}

    void build() override;
    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    void print_info() const override;

private:
    //std::shared_ptr<Hitable> children[2];
    std::shared_ptr<BVHNode> root;
};

class EmbreeAccel : public Accelerator {
public:
    EmbreeAccel(std::vector<std::shared_ptr<Hitable>>* hs);
    ~EmbreeAccel();

    void add_sphere(std::shared_ptr<Sphere>& s) override;
    void add_quad(std::shared_ptr<Quad>& q) override;
    void add_triangle(std::shared_ptr<Triangle>& t) override;
    void add_trianglemesh(std::shared_ptr<TriangleMesh>& t) override;
    void build() override;
    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    void print_info() const override;

private:
    RTCDevice   m_device = nullptr;
    RTCScene    m_scene = nullptr;
};
