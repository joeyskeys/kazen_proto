#include "base/vec.h"
#include "core/shape.h"
#include "core/transform.h"
#include "core/material.h"
#include "core/accel.h"

#include <iostream>
#include <memory>

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
    Vec3f local_wo = world_to_local(-r.direction, isect.N, isect.tangent, isect.bitangent);
    float p;
    //lamb.sample_f(local_wo, wi, isect, Vec2f{0.8f, 0.4f}, p);
    metal.sample_f(local_wo, wi, isect, Vec2f{0.1f, 0.1f}, p);

    /*
    std::cout << "hit : " << hit << std::endl;
    std::cout << "position : " << isect.position;
    std::cout << "normal : " << isect.normal;
    std::cout << "tangent : " << isect.tangent;
    std::cout << "bitangent : " << isect.bitangent;
    std::cout << "local wo : " << local_wo;
    std::cout << "world wo : " << -r.direction;
    std::cout << "local wi : " << wi;
    std::cout << "world wi : " << local_to_world(wi, isect.normal, isect.tangent, isect.bitangent);
    */

    auto tt = Transform();
    tt.translate(Vec3f{0.f, 0.f, -10.f});
    Triangle tri{tt, Vec3f{0.f, 0.f, 0.f}, Vec3f{2.f, 0.f, 0.f}, Vec3f{0.f, 2.f, 0.f}};
    Ray r2{Vec3f{2.f, 1.5f, 2.f}, Vec3f{0.f, 0.f, -1.f}};
    hit = tri.intersect(r2, isect);

    std::cout << "hit : " << hit << std::endl;
    std::cout << "position : " << isect.P;
    std::cout << "normal : " << isect.N;
    std::cout << "t : " << isect.ray_t << std::endl;

    std::vector<std::shared_ptr<Hitable>> hitables;
    Accelerator list{&hitables};
    list.add_hitable(std::make_shared<Sphere>(t, 0, 1.f));

    auto sh = std::make_shared<Transform>();
    auto acc = std::make_shared<BVHAccel>(&hitables);
    
    return 0;
}