#include <limits>

#include "accel.h"
#include "material.h"

void ListAccel::add_hitable(const HitablePtr h) {
    hitables.emplace_back(h);
}

bool ListAccel::intersect(Ray& r, Intersection& isect) const {
    bool hit = false;
    Intersection tmp_sect;
    tmp_sect.ray_t = std::numeric_limits<float>::max();
    float curr_t;

    for (auto& h : hitables) {
        if (h->intersect(r, tmp_sect) && tmp_sect.ray_t < isect.ray_t) {
            hit = true;
            r.tmax = tmp_sect.ray_t;
            isect = tmp_sect;
        }
    }

    return hit;
}