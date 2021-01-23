#include "base/vec.h"
#include "base/utils.h"

#include <iostream>

int main() {
    Vec3f a{0.f, 0.f, 0.f};
    Vec3f b{0.f, 0.5f, 0.8f};
    Vec3f c{0.5f, 0.1f, 0.2f};

    std::cout << "a + b : " << a + b;

    a += b;
    std::cout << "a += b : " << a;

    std::cout << "a - c : " << a - c;

    a-= c; 
    std::cout << "a -= c : " << a;

    std::cout << "a / 2 : " << a / 2.f;

    Vec3f n{0.f, 1.f, -1.f};
    n.normalize();
    Vec3f t{1.f, 0.f, 0.f};
    t.normalize();
    auto bi = cross(t, n).normalized();

    Vec3f w{1.f, 0.f, 1.f};
    auto w2l = world_to_tangent(w, n, t, bi);
    auto l2w = tangent_to_world(w, n, t, bi);

    std::cout << "w2l : " << w2l;
    std::cout << "l2w : " << l2w;
}