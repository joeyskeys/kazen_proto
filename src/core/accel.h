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
    Accelerator(std::vector<std::shared_ptr<Hitable>>* hs = nullptr)
        : hitables(hs)
    {}

    inline size_t size() const { return hitables->size(); }
    void add_hitable(std::shared_ptr<Hitable>&& h);

    virtual void add_sphere(std::shared_ptr<Sphere>& s);
    virtual void add_quad(std::shared_ptr<Quad>& q);
    virtual void add_triangle(std::shared_ptr<Triangle>& t);
    virtual void add_trianglearray(std::shared_ptr<TriangleArray>& t) {}
    virtual void add_trianglemesh(std::shared_ptr<TriangleMesh>& t);
    virtual void add_spheres(std::vector<std::shared_ptr<Sphere>>& ss) {}
    virtual void add_instances(const std::string& name,
        const std::vector<std::string>& instance_names,
        const Transform& trans = Transform(),
        bool root = false) {}
    virtual void build(const std::vector<std::string>&) {}

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    void print_info() const override;

public:
    std::vector<std::shared_ptr<Hitable>>*  hitables;
    std::vector<uint32_t>               inst_ids;
};

class BVHNode;

class BVHAccel : public Accelerator {
public:
    BVHAccel(std::vector<std::shared_ptr<Hitable>>* hs)
        : Accelerator(hs)
        , root(nullptr)
    {}

    void build(const std::vector<std::string>&) override;
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
    void add_trianglemesh(std::shared_ptr<TriangleMesh>&) override;
    void add_instances(const std::string& name,
        const std::vector<std::string>& instance_names,
        const Transform& trans = Transform(),
        bool root = false) override;
    inline void build(const std::vector<std::string>& instance_names) override {
        add_instances("root", instance_names, Transform(), true);
    }
    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    void print_info() const override;

private:
    RTCDevice   m_device = nullptr;
    RTCScene    m_scene = nullptr;
    std::unordered_map<std::string, RTCGeometry> m_geoms;
    std::unordered_map<uint32_t, std::shared_ptr<Hitable>> m_meshes;
};

class OptixAccel : public Accelerator {
public:
    using handle_map = std::unordered_map<std::string, std::pair<OptixTraversableHandle, CUdeviceptr>>;

    OptixAccel(const OptixDeviceContext&);
    ~OptixAccel();

    void add_sphere(std::shared_ptr<Sphere>& s) override;
    void add_quad(std::shared_ptr<Quad>& q) override;
    void add_triangle(std::shared_ptr<Triangle>& t) override;
    void add_trianglearray(std::shared_ptr<TriangleArray>&) override;
    void add_trianglemesh(std::shared_ptr<TriangleMesh>&) override;
    void add_spheres(std::vector<std::shared_ptr<Sphere>>& ss) override;
    void add_instances(const std::string& name,
        const std::vector<std::string>& instance_names,
        const Transform& trans = Transform(),
        bool root = false) override;
    inline void build(const std::vector<std::string>& instance_names) override {
        add_instances("root", instance_names, Transform(), true);
    }
    void print_info() const override;

    inline OptixTraversableHandle get_root_handle() const {
        return root_handle;
    }

private:
    OptixDeviceContext                  ctx;
    handle_map                          handles;
    OptixTraversableHandle              root_handle;
    CUdeviceptr                         root_buf;
    uint32_t                            inst_cnt = 0;
};