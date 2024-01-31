#pragma once

#include <vector>
#include <memory>
#include <unordered_map>

#include <embree4/rtcore.h>

#include "core/hitable.h"
#include "core/shape.h"
#include "core/optix_utils.h"

class Accelerator : public Hitable {
public:
    Accelerator(std::vector<std::shared_ptr<Hitable>>* hs)
        : hitables(hs)
    {}

    inline size_t size() const { return hitables->size(); }
    void add_hitable(std::shared_ptr<Hitable>&& h);

    virtual void add_sphere(std::shared_ptr<Sphere>& s);
    virtual void add_quad(std::shared_ptr<Quad>& q);
    virtual void add_triangle(std::shared_ptr<Triangle>& t);
    virtual void add_trianglemesh(std::shared_ptr<TriangleMesh>& t);
    virtual void add_spheres(std::vector<std::shared_ptr<Sphere>& ss);
    virtual void add_trianglemeshes(std::vector<std::shared_ptr<TriangleMesh>& ts);
    virtual void build() {}

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    void print_info() const override;

public:
    std::vector<std::shared_ptr<Hitable>>*  hitables;
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
    std::unordered_map<uint32_t, std::pair<RTCScene, std::shared_ptr<Hitable>>> m_subscenes;
};

class OptixAccel : public Accelerator {
public:
    OptixAccel(const OptixDeviceContext&);
    ~OptixAccel();

    void add_sphere(std::shared_ptr<Sphere>& s) override;
    void add_quad(std::shared_ptr<Quad>& q) override;
    void add_triangle(std::shared_ptr<Triangle>& t) override;
    void add_trianglemesh(std::shared_ptr<TriangleMesh>& t) override;
    void add_spheres(std::vector<std::shared_ptr<Sphere>& ss) override;
    void add_trianglemeshes(std::vector<std::shared_ptr<TriangleMesh>>& ts) override;
    void build() override;
    //bool intersect(const Ray& r, Intersection& isect) const override;
    //bool intersect(const Ray& r, float& t) const override;
    void print_info() const override;

private:
    OptixDeviceContext                  ctx;
    CUdeviceptr                         d_sphere_data;
    std::vector<CUdeviceptr>            d_mesh_vertice_data;
    std::vector<OptixTraversableHandle> gas_handles;
    std::vector<CUdeviceptr>            gas_output_bufs;
    OptixTraversableHandle              ias_handle;
    CUdeviceptr                         ias_output_buf;
    std::vector<OptixInstance>          instances;
}