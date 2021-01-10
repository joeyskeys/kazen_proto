#pragma once

#include <vector>

#include "hitable.h"

class Accel : public HitableInterface {
public:

};

class ListAccel : public Accel {
public:
    void add_hitable(const HitablePtr h);
    bool intersect(Ray& r, Intersection& isect) const override;

private:
    std::vector<HitablePtr> hitables;
};