#pragma once

#include <vector>
#include <memory>

#include "hitable.h"

class ListAccel : public Hitable {
public:
    void add_hitable(std::shared_ptr<Hitable>&& h);
    void add_hitables(const std::vector<std::shared_ptr<Hitable>>& hs);
    bool intersect(const Ray& r, Intersection& isect) const override;
    inline size_t size() { hitables.size(); }

public:
    std::vector<std::shared_ptr<Hitable>> hitables;
};

class BVHAccel : public Hitable {
public:
    BVHAccel() {}
    BVHAccel(std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end);

    bool intersect(const Ray& r, Intersection& isect) const override;
    void print_bound() const override;

private:
    std::shared_ptr<Hitable> children[2];
};
