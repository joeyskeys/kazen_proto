#pragma once

#include <vector>
#include <memory>

#include "hitable.h"

class ListAccel : public Hitable {
public:
    void add_hitable(std::shared_ptr<Hitable>&& h);
    void add_hitables(const std::vector<std::shared_ptr<Hitable>>& hs);
    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    inline size_t size() { return hitables.size(); }

public:
    std::vector<std::shared_ptr<Hitable>> hitables;
};

class BVHNode;

class BVHAccel : public Hitable {
public:
    BVHAccel() {}
    BVHAccel(std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end);

    void reset(std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end);
    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    void print_info() const;

private:
    //std::shared_ptr<Hitable> children[2];
    std::shared_ptr<BVHNode> root;
};
