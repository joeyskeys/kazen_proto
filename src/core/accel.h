#pragma once

#include <vector>
#include <memory>

#include <embree3/rtcore.h>

#include "hitable.h"
#include "shape.h"

class Scene;

class Accelerator : public Hitable {
public:
    Accelerator(const Scene* s)
        : hitables(s)
    {}

    inline size_t size() const { return scene->objects.size(); }
    void add_hitable(std::shared_ptr<Hitable>&& h);
    void add_hitables(const std::vector<std::shared_ptr<Hitable>>& hs);

    virtual void build();

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    void print_info() const override;

public:
    Scene* scene;
};

class BVHNode;

class BVHAccel : public Accelerator {
public:
    BVHAccel(const Scene* s)
        : Accelerator(s)
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
    EmbreeAccel(Scene* s);
    ~EmbreeAccel();

    void build() override;
    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    void print_info() const override;

private:
    RTCDevice   m_device = nullptr;
    RTCScene    m_scene = nullptr;
};
