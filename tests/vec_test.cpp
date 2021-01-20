#include "base/vec.h"

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
}