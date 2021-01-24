#include "base/vec.h"
#include "core/shape.h"
#include "core/transform.h"
#include "core/material.h"

#include <iostream>

int main() {
    Transform t;
    t.translate(Vec3f{0.f, 0.f, -5.f});
    Sphere s{t, 0, 1.f};
    Ray r{Vec3f{0.f, 0.2f, 5.f}, Vec3f{0.f, 0.05f, -1.f}};
    Intersection isect;
    bool hit = s.intersect(r, isect);

    LambertianBxDF lamb{RGBSpectrum{1.f, 1.f, 1.f}};
    MetalBxDF metal{RGBSpectrum{1.f, 1.f, 1.f}};
    Vec3f wi;
    Vec3f local_wo = world_to_tangent(-r.direction, isect.normal, isect.tangent, isect.bitangent);
    float p;
    //lamb.sample_f(local_wo, wi, isect, Vec2f{0.8f, 0.4f}, p);
    metal.sample_f(local_wo, wi, isect, Vec2f{0.1f, 0.1f}, p);

    std::cout << "hit : " << hit << std::endl;
    std::cout << "position : " << isect.position;
    std::cout << "normal : " << isect.normal;
    std::cout << "tangent : " << isect.tangent;
    std::cout << "bitangent : " << isect.bitangent;
    std::cout << "local wo : " << local_wo;
    std::cout << "world wo : " << -r.direction;
    std::cout << "local wi : " << wi;
    std::cout << "world wi : " << tangent_to_world(wi, isect.normal, isect.tangent, isect.bitangent);
    
    return 0;
}