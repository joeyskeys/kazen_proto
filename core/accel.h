#pragma once

#include <vector>
#include <memory>

#include "hitable.h"

class ListAccel : public Hitable {
public:
    void add_hitable(Hitable&& h);
    bool intersect(Ray& r, Intersection& isect) const override;

private:
    std::vector<std::shared_ptr<Hitable>> hitables;
};

class BVHAccel : public Hitable {
public:
    BVHAccel(const std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end);
    bool intersect(Ray& r, Intersection& isect) const override;

private:
    std::shared_ptr<Hitable> children[2];
};