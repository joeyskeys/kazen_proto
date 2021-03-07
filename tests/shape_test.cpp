#include "core/shape.h"
#include "core/transform.h"
#include "core/ray.h"

int main() {
    Transform t;
    t.translate(Vec3f{0.f, 0.f, -5.f});
    Sphere s{t, 0, 1.f};

    Ray r{Vec3f{0.5f, 0.5f, 5.f}, Vec3f{0.f, 0.f, -1.f}};

    std::cout << "sphere bbox : " << s.bbox();
    std::cout << "intersect test : " << s.bbox().intersect(r) << std::endl;

    auto triangle_meshes = load_triangle_mesh("../resource/obj/cube.obj");
    auto triangle_mesh = triangle_meshes[0];

    Ray r2{Vec3f{0.5f, 0.5f, 5.f}, Vec3f{0.f, 0.f, -1.f}};
    
    std::cout << "triangle_mesh bbox : " << triangle_mesh->bbox();
    std::cout << "triangle bbox intersection : " << triangle_mesh->bbox().intersect(r2) << std::endl;

    return 0;
}