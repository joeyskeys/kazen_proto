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

bool ListAccel::intersect(const Ray& r, Intersection& isect) const {
    bool hit = false;
    Intersection tmp_sect;
    tmp_sect.ray_t = std::numeric_limits<float>::max();
    float curr_t;
    auto tmax = r.tmax;
    auto tmin = r.tmin;

    for (auto& h : hitables) {
        if (h->intersect(r, tmp_sect) && tmp_sect.ray_t < isect.ray_t) {
            hit = true;
            tmax = tmp_sect.ray_t;
            isect = tmp_sect;
        }
    }

    return hit;
}

bool ListAccel::intersect(const Ray& r, float& t) const {
    bool hit = false;
    float tmp_t = std::numeric_limits<float>::max();

    for (auto& h : hitables) {
        if (h->intersect(r, tmp_t) && tmp_t < t) {
            hit = true;
            t = tmp_t;
        }
    }

    return hit;
}

inline bool box_compare(const std::shared_ptr<Hitable>& a, const std::shared_ptr<Hitable>& b, int axis) {
    AABBf box_a;
    AABBf box_b;

    return a->bbox().min[axis] < b->bbox().min[axis];
}

/*
bool x_compare(const std::shared_ptr<Hitable>& a, const std::shared_ptr<Hitable>& b) {
    return box_compare(a, b, 0);
}

bool y_compare(const std::shared_ptr<Hitable>& a, const std::shared_ptr<Hitable>& b) {
    return box_compare(a, b, 1);
}

bool z_compare(const std::shared_ptr<Hitable>& a, const std::shared_ptr<Hitable>& b) {
    return box_compare(a, b, 2);
}
*/

class BVHNode : public Hitable {
public:
    BVHNode(std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end);

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    void print_bound() const override;

private:
    std::shared_ptr<Hitable> children[2];
};

//BVHAccel::BVHAccel(std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end) {
BVHNode::BVHNode(std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end) {
    int axis = randomi(2);

    /*
    auto comparator = axis == 0 ? x_compare :
                      axis == 1 ? y_compare :
                      z_compare;
    */

    auto comparator = [&axis](auto a, auto b) {
        return box_compare(a, b, axis);
    };

    size_t object_span = end - start;
    assert(object_span > 0);

    if (object_span == 1) {
        children[0] = children[1] = hitables[start];
    }
    else if (object_span == 2) {
        if (comparator(hitables[start], hitables[start + 1])) {
            children[0] = hitables[start];
            children[1] = hitables[start + 1];
        }
        else {
            children[0] = hitables[start + 1];
            children[1] = hitables[start];
        }
    }
    else {
        std::sort(hitables.begin() + start, hitables.begin() + end, comparator);
        auto mid = start + object_span / 2;
        children[0] = std::make_shared<BVHNode>(hitables, start, mid);
        children[1] = std::make_shared<BVHNode>(hitables, mid, end);
    }

    bound = bound_union(children[0]->bbox(), children[1]->bbox());
}

bool BVHNode::intersect(const Ray& r, Intersection& isect) const {
    if (!bound.intersect(r))
        return false;

    bool hit_0 = children[0]->intersect(r, isect);
    bool hit_1 = children[1]->intersect(r, isect);

    return hit_0 || hit_1;
}

bool BVHNode::intersect(const Ray& r, float& t) const {
    if (!bound.intersect(r))
        return false;
    
    bool hit_0 = children[0]->intersect(r, t);
    bool hit_1 = children[1]->intersect(r, t);

    return hit_0 || hit_1;
}

void BVHNode::print_bound() const {
    std::cout << "bvh node bound : " << bound;

    children[0]->print_bound();
    children[1]->print_bound();
}

BVHAccel::BVHAccel(std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end)
    : root(std::make_shared<BVHNode>(hitables, start, end))
{}

void BVHAccel::reset(std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end) {
    root = std::make_shared<BVHNode>(hitables, start, end);
}

bool BVHAccel::intersect(const Ray& r, Intersection& isect) const {
    return root->intersect(r, isect);
}

bool BVHAccel::intersect(const Ray& r, float& t) const {
    return root->intersect(r, t);
}
