
#include "accel.h"

void ListAccel::add_hitable(const HitablePtr h) {
    hitables.emplace_back(h);
}

bool ListAccel::intersect(Ray& r, Intersection& isect) const {
    bool hit = false;
    for (auto& h : hitables) {
        hit |= h->intersect(r, isect);
    }

    return hit;
}