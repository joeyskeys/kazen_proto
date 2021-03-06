#include <limits>

#include "accel.h"
#include "material.h"
#include "sampling.h"

void ListAccel::add_hitable(std::shared_ptr<Hitable>&& h) {
    bound_union(bound, h->bbox());
    hitables.emplace_back(h);
}

void ListAccel::add_hitables(const std::vector<std::shared_ptr<Hitable>>& hs) {
    for (auto& h : hs) {
        bound_union(bound, h->bbox());
        hitables.emplace_back(h);
    }
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

inline bool box_compare(const std::shared_ptr<Hitable>& a, const std::shared_ptr<Hitable>& b, int axis) {
    AABBf box_a;
    AABBf box_b;

    return a->bbox().min[axis] < b->bbox().min[axis];
}

bool x_compare(const std::shared_ptr<Hitable>& a, const std::shared_ptr<Hitable>& b) {
    return box_compare(a, b, 0);
}

bool y_compare(const std::shared_ptr<Hitable>& a, const std::shared_ptr<Hitable>& b) {
    return box_compare(a, b, 1);
}

bool z_compare(const std::shared_ptr<Hitable>& a, const std::shared_ptr<Hitable>& b) {
    return box_compare(a, b, 2);
}

BVHAccel::BVHAccel(const std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end) {
    auto local_hitables = hitables;

    int axis = randomi(2);
    auto comparator = axis == 0 ? x_compare :
                      axis == 1 ? y_compare :
                      z_compare;

    size_t object_span = hitables.size();

    if (object_span == 1) {
        children[0] = children[1] = local_hitables[start];
    }
    else if (object_span == 2) {
        if (comparator(hitables[start], hitables[start + 1])) {
            children[0] = local_hitables[start];
            children[1] = hitables[start + 1];
        }
        else {
            children[0] = hitables[start + 1];
            children[1] = hitables[start];
        }
    }
    else {
        std::sort(local_hitables.begin() + start, local_hitables.begin() + end, comparator);
        auto mid = start + object_span / 2;
        children[0] = std::make_shared<BVHAccel>(local_hitables, start, mid);
        children[1] = std::make_shared<BVHAccel>(local_hitables, mid, end);
    }

    bound = bound_union(children[0]->bbox(), children[1]->bbox());
}

bool BVHAccel::intersect(Ray& r, Intersection& isect) const {
    if (!bound.intersect(r))
        return false;

    bool hit_0 = children[0]->intersect(r, isect);
    bool hit_1 = children[1]->intersect(r, isect);

    return hit_0 || hit_1;
}